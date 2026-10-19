//! Allocation-count regression test: the borrowed-aware `reserve*` family must
//! **fuse** the copy-on-write with the reservation into a single allocation,
//! exactly like `try_reserve*`, instead of materializing at `len` and then
//! reallocating to the planned capacity.
//!
//! A counting global allocator is installed for this test binary. The file
//! compiles to **one** `#[test]` (the cow test with the `cow` feature, a
//! vec-only twin without it) so no test body can run in parallel with the
//! counting windows: with one test, libtest's own bookkeeping is quiet until
//! the test returns, and every allocation counted between two snapshots was
//! made by the test itself. `--test-threads=1` alone would not be enough:
//! the counter is process-global, so any allocation made by another test (or
//! by libtest on another thread) inside a snapshot window would be
//! indistinguishable from the test's own.
//!
//! # One binary, one allocator
//!
//! Cargo builds each `tests/*.rs` as its own binary, which is what lets both
//! this file and `alloc_fail.rs` install a `#[global_allocator]`. Never merge
//! them into a shared module or a single multi-test binary: duplicate global
//! allocators are a compile error. Each such file is also an extra full build
//! of the library.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cow")]
use covariant_vec::CovariantCow;
use covariant_vec::CovariantVec;

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static COUNTER: Counting = Counting;

/// Runs `f`, returning how many allocation operations (`alloc`/`realloc`/
/// `alloc_zeroed`) it performed and its result.
fn count<T>(f: impl FnOnce() -> T) -> (usize, T) {
    let before = ALLOCS.load(Ordering::SeqCst);
    let out = f();
    (ALLOCS.load(Ordering::SeqCst) - before, out)
}

// Only one test function exists per build: the counter is process-global, so
// the cow test and the vec-only twin must never run concurrently.
#[cfg(feature = "cow")]
#[test]
fn borrowed_reserve_fuses_copy_on_write() {
    let src: [i32; 16] = core::array::from_fn(|i| i as i32);

    // Control: `push` was already fused before this test existed, so it pins
    // the measurement itself — one allocation for a borrowed push past `N`.
    // (If this fails, the counter is wrong, not the code under test.)
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let (n, ()) = count(|| c.push(100));
    assert_eq!(n, 1, "borrowed push must allocate once");
    assert_eq!(&c[..16], &src[..]);

    // `reserve` on a borrowed cow: ONE allocation. The old
    // materialize-at-`len` + realloc shape counted two.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    assert!(c.is_borrowed());
    let (n, ()) = count(|| c.reserve(4));
    assert_eq!(n, 1, "borrowed reserve must fuse into one allocation");
    assert!(!c.is_borrowed());
    assert!(c.capacity() >= 20);
    assert_eq!(&*c, &src[..]);

    // The fused `reserve` and the fused `try_reserve` share a capacity
    // trajectory.
    let mut d = CovariantCow::<i32, 4>::from_borrowed(&src);
    d.try_reserve(4).unwrap();
    assert_eq!(c.capacity(), d.capacity());

    // `reserve_exact` on a borrowed cow: ONE allocation, exact capacity.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let (n, ()) = count(|| c.reserve_exact(4));
    assert_eq!(n, 1, "borrowed reserve_exact must fuse into one allocation");
    assert_eq!(c.capacity(), 20);
    assert_eq!(&*c, &src[..]);
    let mut d = CovariantCow::<i32, 4>::from_borrowed(&src);
    d.try_reserve_exact(4).unwrap();
    assert_eq!(c.capacity(), d.capacity());

    // Zero additional capacity stays a no-op that allocates nothing and
    // leaves the cow borrowed.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let (n, ()) = count(|| c.reserve(0));
    assert_eq!(n, 0);
    let (n, ()) = count(|| c.reserve_exact(0));
    assert_eq!(n, 0);
    assert!(c.is_borrowed());

    // A request that already fits inline materializes inline, no allocation.
    let short = [1i32, 2, 3];
    let mut c = CovariantCow::<i32, 8>::from_borrowed(&short);
    let (n, ()) = count(|| c.reserve(3));
    assert_eq!(n, 0, "fits-inline reserve must not allocate");
    assert!(!c.is_borrowed());
    assert_eq!(c.capacity(), 8);
    assert_eq!(&*c, &[1, 2, 3]);

    // `reserve_exact` on a borrowed cow whose request exactly fills the inline
    // buffer: also no allocation (exact = needed <= N).
    let mut c = CovariantCow::<i32, 8>::from_borrowed(&short);
    let (n, ()) = count(|| c.reserve_exact(5));
    assert_eq!(n, 0);
    assert_eq!(c.capacity(), 8);

    // Both-borrowed `append` fuses the copy-on-write with the room for
    // `other` into one allocation.
    let left = [1i32, 2, 3];
    let right = [4i32, 5, 6];
    let mut a = CovariantCow::<i32, 4>::from_borrowed(&left);
    let mut b = CovariantCow::<i32, 4>::from_borrowed(&right);
    let (n, ()) = count(|| a.append(&mut b));
    assert_eq!(n, 1, "both-borrowed append must allocate once");
    assert_eq!(&*a, &[1, 2, 3, 4, 5, 6]);

    // A borrowed `splice` that grows past the hole folds the replacement's
    // lower size hint into the materialization: ONE allocation, and the
    // phase-2 inserts fit without a realloc. (Materializing at exactly `len`
    // first would make the first insert realloc.)
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    // Drop the returned `CowSplice` immediately (it applies the replacement);
    // collecting the removed elements would allocate and pollute the count.
    let (n, ()) = count(|| drop(c.splice(1..3, 0..5)));
    assert_eq!(
        n, 1,
        "borrowed splice must fuse growth into materialization"
    );
    assert_eq!(c.len(), 19); // 16 - 2 + 5
                             // The fused materialization asks for 16 + (5 - 2) = 19 slots; the
                             // amortized branch of `planned_cap` doubles the current length instead,
                             // giving max(2 * 16, 19) = 32. That is the same curve `Vec::reserve`
                             // uses, so 32 is derivable rather than incidental; the `*_exact` twins
                             // (pinned above) are the exact-fit alternative.
    assert_eq!(
        c.capacity(),
        32,
        "materialization must follow the amortized curve (max(2 * len, needed))"
    );
    assert!(!c.is_borrowed());
    assert_eq!(&c[..6], &[0, 0, 1, 2, 3, 4]);
    assert_eq!(c[6], 3);

    // `extend_from_slice` on a borrowed cow fuses the copy-on-write with the
    // room for the slice: ONE allocation.
    let more = [100i32, 101, 102, 103];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let (n, ()) = count(|| c.extend_from_slice(&more));
    assert_eq!(n, 1, "borrowed extend_from_slice must allocate once");
    assert_eq!(c.len(), 20);
    assert_eq!(&c[..16], &src[..]);
    assert_eq!(&c[16..], &more[..]);

    // An inline vector `extend_from_slice` that fits allocates nothing.
    let mut v = CovariantVec::<i32, 8>::new();
    let (n, ()) = count(|| v.extend_from_slice(&[1, 2, 3]));
    assert_eq!(n, 0, "inline extend_from_slice must not allocate");
    assert_eq!(&*v, &[1, 2, 3]);

    // `try_to_mut` materializes at exactly `len`, like `to_mut`: ONE
    // allocation for a borrowed heap case, none when it fits inline, and none
    // when the cow is already owned.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let (n, ok) = count(|| c.try_to_mut().is_ok());
    assert!(ok);
    assert_eq!(n, 1, "borrowed try_to_mut must allocate once");
    assert_eq!(c.capacity(), 16, "try_to_mut sizes the block exactly");
    assert_eq!(&*c, &src[..]);

    let short = [1i32, 2, 3];
    let mut c = CovariantCow::<i32, 8>::from_borrowed(&short);
    let (n, ok) = count(|| c.try_to_mut().is_ok());
    assert!(ok);
    assert_eq!(n, 0, "fits-inline try_to_mut must not allocate");
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());

    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned(0..10);
    let ptr = c.as_ptr();
    let (n, ok) = count(|| c.try_to_mut().is_ok());
    assert!(ok);
    assert_eq!(n, 0, "owned try_to_mut must not allocate");
    assert_eq!(c.as_ptr(), ptr);
}

/// Cow-less twin: the `CovariantVec` half of the allocation assertions, which
/// the cow test would otherwise hide in a `--no-default-features` build.
#[cfg(not(feature = "cow"))]
#[test]
fn vec_bulk_ops_allocation_counts() {
    // An inline vector `extend_from_slice` that fits allocates nothing.
    let mut v = CovariantVec::<i32, 8>::new();
    let (n, ()) = count(|| v.extend_from_slice(&[1, 2, 3]));
    assert_eq!(n, 0, "inline extend_from_slice must not allocate");
    assert_eq!(&*v, &[1, 2, 3]);

    // A spill is exactly one allocation, and heap growth is one realloc.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4]);
    assert!(!v.is_heap());
    let (n, ()) = count(|| v.push(5));
    assert_eq!(n, 1, "inline spill must allocate once");
    assert!(v.is_heap());
    let (n, ()) = count(|| v.reserve(100));
    assert_eq!(n, 1, "heap growth must realloc once");
}
