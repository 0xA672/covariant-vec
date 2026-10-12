//! Allocation-count regression test: the borrowed-aware `reserve*` family must
//! **fuse** the copy-on-write with the reservation into a single allocation,
//! exactly like `try_reserve*`, instead of materializing at `len` and then
//! reallocating to the planned capacity.
//!
//! A counting global allocator is installed for this test binary. The whole
//! file is a **single** `#[test]` so no test body can run in parallel with the
//! counting windows: with one test, libtest's own bookkeeping is quiet until
//! the test returns, and every allocation counted between two snapshots was
//! made by the test itself.

#![cfg(feature = "cow")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use covariant_vec::CovariantCow;

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
}
