//! Allocation-failure regression test: the `AllocError` arm of the fallible
//! capacity paths (`try_materialize`, `try_grow`) must leave the container
//! untouched — a borrowed cow stays borrowed with its data intact, an owned
//! container keeps its contents and capacity, and an inline vector stays
//! inline. A fault-injecting global allocator fails exactly one allocation at
//! a time.
//!
//! Like `alloc_count.rs`, this file is a **single** `#[test]`: the failure
//! flag is process-global, so a second test body running in parallel could
//! steal (or absorb) the injected failure.

#![cfg(feature = "cow")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use covariant_vec::{CovariantCow, CovariantVec, TryReserveErrorKind};

struct Faulty;

/// When set, the next allocation operation returns null (and is recorded);
/// the flag then clears itself.
static FAIL_NEXT: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Faulty {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT.swap(false, Ordering::SeqCst) {
            FAILED.fetch_add(1, Ordering::SeqCst);
            return ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if FAIL_NEXT.swap(false, Ordering::SeqCst) {
            FAILED.fetch_add(1, Ordering::SeqCst);
            return ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT.swap(false, Ordering::SeqCst) {
            FAILED.fetch_add(1, Ordering::SeqCst);
            return ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static FAULTY: Faulty = Faulty;

/// Arms the fault and resets the injection counter.
fn fail_next() {
    FAILED.store(0, Ordering::SeqCst);
    FAIL_NEXT.store(true, Ordering::SeqCst);
}

/// How many failures were injected since the last [`fail_next`].
fn injected() -> usize {
    FAILED.load(Ordering::SeqCst)
}

#[test]
fn alloc_error_leaves_state_untouched() {
    // 1. Borrowed cow, `try_reserve`: the fused materialization's `alloc`
    //    fails -> `Err(AllocError)`, still borrowed, same borrow, same data.
    let src: [i32; 8] = core::array::from_fn(|i| i as i32);
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let borrow_before = c.as_borrowed().unwrap().as_ptr();
    fail_next();
    let e = c.try_reserve(100).unwrap_err();
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(c.is_borrowed());
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), borrow_before);
    assert_eq!(&*c, &src[..]);

    // 2. Owned heap cow, `try_reserve_owned`: the `realloc` fails -> contents
    //    and capacity intact (a failed `realloc` leaves the old block alive).
    let mut c = CovariantCow::<i32, 2>::new();
    c.extend_owned(0..5);
    assert!(c.is_heap());
    let cap = c.capacity();
    fail_next();
    let e = c.try_reserve_owned(1000).unwrap_err();
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), cap);
    assert_eq!(&*c, &[0, 1, 2, 3, 4]);

    // 3. Inline vec, `try_reserve` (the spill's `alloc` fails): stays inline,
    //    contents intact.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    fail_next();
    let e = v.try_reserve(100).unwrap_err();
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert_eq!(&*v, &[1, 2, 3]);

    // 4. Heap vec, `try_reserve` (the `realloc` fails): contents and capacity
    //    intact.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..5);
    assert!(v.is_heap());
    let cap = v.capacity();
    fail_next();
    let e = v.try_reserve(1000).unwrap_err();
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    // The containers are still fully usable after the failed requests.
    v.push(5);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);
}
