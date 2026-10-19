//! Allocation-failure regression test: the `AllocError` arm of the fallible
//! capacity paths (`try_materialize`, `try_grow`) must leave the container
//! untouched — a borrowed cow stays borrowed with its data intact, an owned
//! container keeps its contents and capacity, and an inline vector stays
//! inline. A fault-injecting global allocator fails exactly one allocation at
//! a time.
//!
//! Like `alloc_count.rs`, this file has **one** `#[test]` per build: the
//! failure flag is process-global, so a second test body running in parallel
//! could steal (or absorb) the injected failure. In a `--no-default-features`
//! build the cow sections are compiled out and a vec-only twin runs instead,
//! so the vec assertions are not hidden by the `cow` feature.
//!
//! # One binary, one allocator
//!
//! This file installs a `#[global_allocator]`. Cargo builds each
//! `tests/*.rs` as its own binary, which is what makes that legal: never
//! merge this file into a shared module or a single multi-test binary, or
//! the second `#[global_allocator]` will be a compile error. (It also means
//! this file is an extra full build of the library, like `alloc_count.rs`.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "cow")]
use covariant_vec::CovariantCow;
use covariant_vec::{CovariantVec, TryReserveErrorKind};

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

/// Extracts the error from a fallible capacity call **without allocating**.
///
/// `unwrap_err()` formats the `Ok` value through `Debug` when the call
/// wrongly succeeds; that formatting allocates while the fault flag may still
/// be armed, so the harness would abort with "memory allocation of N bytes
/// failed" instead of reporting the assertion failure. This helper panics with
/// a `&'static str` payload (no allocation) and disarms the fault first, so
/// the subsequent failure formatting is safe too.
fn expect_alloc_error<T>(
    r: Result<T, covariant_vec::TryReserveError>,
) -> covariant_vec::TryReserveError {
    match r {
        Ok(_) => {
            FAIL_NEXT.store(false, Ordering::SeqCst);
            panic!("try_reserve* must fail under injected AllocError");
        }
        Err(e) => e,
    }
}

#[cfg(feature = "cow")]
#[test]
fn alloc_error_leaves_state_untouched() {
    // 1. Borrowed cow, `try_reserve`: the fused materialization's `alloc`
    //    fails -> `Err(AllocError)`, still borrowed, same borrow, same data.
    let src: [i32; 8] = core::array::from_fn(|i| i as i32);
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let borrow_before = c.as_borrowed().unwrap().as_ptr();
    fail_next();
    let e = expect_alloc_error(c.try_reserve(100));
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
    let e = expect_alloc_error(c.try_reserve_owned(1000));
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
    let e = expect_alloc_error(v.try_reserve(100));
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
    let e = expect_alloc_error(v.try_reserve(1000));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    // The containers are still fully usable after the failed requests.
    v.push(5);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);

    // 5. Borrowed cow, `try_reserve_exact`: the exact fused
    //    materialization's `alloc` fails -> `Err(AllocError)`, still
    //    borrowed, same borrow, same data.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let borrow_before = c.as_borrowed().unwrap().as_ptr();
    fail_next();
    let e = expect_alloc_error(c.try_reserve_exact(100));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(c.is_borrowed());
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), borrow_before);
    assert_eq!(&*c, &src[..]);

    // 6. Owned heap cow, `try_reserve_exact_owned`: the `realloc` fails ->
    //    contents and capacity intact.
    let mut c = CovariantCow::<i32, 2>::new();
    c.extend_owned(0..5);
    assert!(c.is_heap());
    let cap = c.capacity();
    fail_next();
    let e = expect_alloc_error(c.try_reserve_exact_owned(1000));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), cap);
    assert_eq!(&*c, &[0, 1, 2, 3, 4]);

    // 7. Inline vec, `try_reserve_exact` (the spill's `alloc` fails): stays
    //    inline, contents intact.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    fail_next();
    let e = expect_alloc_error(v.try_reserve_exact(100));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert_eq!(&*v, &[1, 2, 3]);

    // 8. Heap vec, `try_reserve_exact` (the `realloc` fails): contents and
    //    capacity intact.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..5);
    assert!(v.is_heap());
    let cap = v.capacity();
    fail_next();
    let e = expect_alloc_error(v.try_reserve_exact(1000));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    // 9. Borrowed cow, `try_to_mut`: the exact materialization's `alloc`
    //    fails -> `Err(AllocError)`, still borrowed, same borrow, same data.
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&src);
    let borrow_before = c.as_borrowed().unwrap().as_ptr();
    fail_next();
    let e = expect_alloc_error(c.try_to_mut());
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(c.is_borrowed());
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), borrow_before);
    assert_eq!(&*c, &src[..]);

    // The infallible capacity methods (`reserve*` and the bulk operations
    // that materialize) are deliberately absent: on `AllocError` they call
    // `handle_alloc_error` and abort the process (std parity), so there is no
    // post-failure state to assert in-process. Their fallible twins above —
    // including `try_to_mut` for `to_mut` — cover the same allocation sites.
}

/// Cow-less twin: the `CovariantVec` half of the failure matrix, which the
/// cow test would otherwise hide in a `--no-default-features` build.
#[cfg(not(feature = "cow"))]
#[test]
fn vec_alloc_error_leaves_state_untouched() {
    // 1. Inline vec, `try_reserve` (the spill's `alloc` fails): stays inline,
    //    contents intact.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    fail_next();
    let e = expect_alloc_error(v.try_reserve(100));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert_eq!(&*v, &[1, 2, 3]);

    // 2. Heap vec, `try_reserve` (the `realloc` fails): contents and capacity
    //    intact.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..5);
    assert!(v.is_heap());
    let cap = v.capacity();
    fail_next();
    let e = expect_alloc_error(v.try_reserve(1000));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    // 3. Inline vec, `try_reserve_exact` (the spill's `alloc` fails): stays
    //    inline, contents intact.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    fail_next();
    let e = expect_alloc_error(v.try_reserve_exact(100));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert_eq!(&*v, &[1, 2, 3]);

    // 4. Heap vec, `try_reserve_exact` (the `realloc` fails): contents and
    //    capacity intact.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..5);
    assert!(v.is_heap());
    let cap = v.capacity();
    fail_next();
    let e = expect_alloc_error(v.try_reserve_exact(1000));
    assert!(matches!(e.kind(), TryReserveErrorKind::AllocError { .. }));
    assert_eq!(injected(), 1);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    // The container is still fully usable after the failed requests.
    v.push(5);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);
}
