//! Tests for the nightly-only `specialization` feature: the `T: Copy`
//! bulk-copy fast path in `Clone` (both containers) and in
//! `CovariantCow::to_mut` / `CovariantCow::try_materialize`.
//!
//! The *only* observable difference between the two paths is whether
//! `T::clone` is called, so these tests key on exactly that:
//!
//! * a `Copy` type whose `Clone` counts its calls — a contract-violating but
//!   entirely sound test double — proves the fast path ran (zero calls) or did
//!   not (one call per element);
//! * a `!Copy` type that needs no drop but has an observable `Clone` proves the
//!   element-wise loop still runs. That is the case the rejected
//!   `!needs_drop::<T>()` dispatch would have silently skipped, so it is
//!   pinned explicitly.
//!
//! Every value assertion (contents, length, capacity, state) must hold in
//! *both* configurations; only the call counts differ.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use covariant_vec::CovariantVec;

static COPY_CLONES: AtomicUsize = AtomicUsize::new(0);
/// Serializes the call-count assertions below: `COPY_CLONES`/
/// `NONCOPY_CLONES` are process-global, so two count-checking tests running
/// concurrently on separate threads would pollute each other's tallies.
static COUNT_LOCK: Mutex<()> = Mutex::new(());
static NONCOPY_CLONES: AtomicUsize = AtomicUsize::new(0);

/// `Copy`, but with an observable `Clone`.
///
/// The `Copy` contract says `clone` must be a bitwise copy; the only thing this
/// type adds is a counter side effect, so it stays sound — it exists purely to
/// observe which path ran. (If the contract were violated *semantically*, the
/// test double itself would be the bug, not the crate.)
#[derive(Copy)]
struct CopyCounter(u32);

// The non-canonical body *is* the point: the counter is the only way to
// observe which clone path ran. (Clippy's `non_canonical_clone_impl` exists to
// catch this shape by accident, not on purpose.)
#[allow(clippy::non_canonical_clone_impl)]
impl Clone for CopyCounter {
    fn clone(&self) -> Self {
        COPY_CLONES.fetch_add(1, Ordering::SeqCst);
        *self
    }
}

/// `!Copy` and `needs_drop::<Self>() == false`, but with an observable `Clone`.
///
/// This is the exact type the review flagged: a `needs_drop`-keyed dispatch
/// would have bitwise-copied it and never run `Clone::clone`. Keying on
/// `T: Copy` must not.
#[derive(Debug, PartialEq)]
struct NoDropNotCopy(u32);

impl Clone for NoDropNotCopy {
    fn clone(&self) -> Self {
        NONCOPY_CLONES.fetch_add(1, Ordering::SeqCst);
        NoDropNotCopy(self.0)
    }
}

const IS_SPECIALIZED: bool = cfg!(feature = "specialization");

// -------------------------------------------------------------------------- //
// CovariantVec::clone                                                         //
// -------------------------------------------------------------------------- //

#[test]
fn vec_clone_copy_heap_is_value_exact() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<CopyCounter, 4>::new();
    for i in 0..10 {
        v.push(CopyCounter(i));
    }
    assert!(v.is_heap());

    let c = v.clone();
    // Value fidelity, length, and the "allocate exactly len" normalization.
    assert_eq!(c.len(), 10);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), 10);
    for (i, x) in c.iter().enumerate() {
        assert_eq!(x.0, i as u32);
    }
    // The source is untouched.
    for (i, x) in v.iter().enumerate() {
        assert_eq!(x.0, i as u32);
    }

    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the memcpy path");
    } else {
        assert_eq!(calls, 10, "stable must run the element-wise loop");
    }
}

#[test]
fn vec_clone_copy_inline_is_value_exact() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<CopyCounter, 8>::new();
    for i in 0..5 {
        v.push(CopyCounter(i));
    }
    assert!(!v.is_heap());

    let c = v.clone();
    assert_eq!(c.len(), 5);
    assert!(!c.is_heap());
    let vals: Vec<u32> = c.iter().map(|x| x.0).collect();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);

    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the memcpy path");
    } else {
        assert_eq!(calls, 5, "stable must run the element-wise loop");
    }
}

#[test]
fn vec_clone_non_copy_still_runs_elementwise() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    NONCOPY_CLONES.store(0, Ordering::SeqCst);
    // Heap case.
    let mut v = CovariantVec::<NoDropNotCopy, 4>::new();
    for i in 0..10 {
        v.push(NoDropNotCopy(i));
    }
    let c = v.clone();
    assert_eq!(c.len(), 10);
    assert_eq!(c.capacity(), 10);
    // Inline case.
    let mut w = CovariantVec::<NoDropNotCopy, 8>::new();
    for i in 0..5 {
        w.push(NoDropNotCopy(i));
    }
    let d = w.clone();
    assert_eq!(d.len(), 5);
    assert!(!d.is_heap());

    // 10 + 5 clones, in *both* configurations: no-drop `!Copy` types keep the
    // element-wise loop (this is the `needs_drop` trap the review called out).
    assert_eq!(NONCOPY_CLONES.load(Ordering::SeqCst), 15);
}

#[test]
fn vec_clone_copy_zst_never_heap() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    // A ZST clone of length > N stays inline (ZSTs never spill).
    let mut v = CovariantVec::<(), 4>::new();
    for _ in 0..10 {
        v.push(());
    }
    assert_eq!(v.len(), 10);
    assert!(!v.is_heap());
    let c = v.clone();
    assert_eq!(c.len(), 10);
    assert!(!c.is_heap());
    assert_eq!(c.iter().count(), 10);
}

#[test]
fn vec_clone_copy_empty() {
    let v = CovariantVec::<CopyCounter, 4>::new();
    let c = v.clone();
    assert!(c.is_empty());
    assert!(!c.is_heap());
}

// -------------------------------------------------------------------------- //
// CovariantCow                                                                //
// -------------------------------------------------------------------------- //

#[cfg(feature = "cow")]
mod cow {
    use super::*;
    use covariant_vec::CovariantCow;

    #[test]
    fn cow_clone_copy_owned() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        COPY_CLONES.store(0, Ordering::SeqCst);
        let mut c = CovariantCow::<CopyCounter, 4>::new();
        for i in 0..10 {
            c.push(CopyCounter(i));
        }
        assert!(c.is_heap());
        let d = c.clone();
        assert_eq!(d.len(), 10);
        assert_eq!(d.capacity(), 10);
        assert!(d.is_heap());
        let vals: Vec<u32> = d.iter().map(|x| x.0).collect();
        assert_eq!(vals, (0..10).collect::<Vec<u32>>());

        let calls = COPY_CLONES.load(Ordering::SeqCst);
        if IS_SPECIALIZED {
            assert_eq!(calls, 0, "T: Copy must take the memcpy path");
        } else {
            assert_eq!(calls, 10, "stable must run the element-wise loop");
        }
    }

    #[test]
    fn cow_clone_borrowed_is_still_zero_copy() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        COPY_CLONES.store(0, Ordering::SeqCst);
        let data = [CopyCounter(1), CopyCounter(2), CopyCounter(3)];
        let c = CovariantCow::<CopyCounter, 4>::from_borrowed(&data);
        let d = c.clone();
        assert!(d.is_borrowed());
        assert_eq!(d.as_ptr(), data.as_ptr());
        // A borrowed clone is a zero-copy share, never an element clone.
        assert_eq!(COPY_CLONES.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cow_to_mut_copy_inline_skips_clone() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        COPY_CLONES.store(0, Ordering::SeqCst);
        let data = [CopyCounter(1), CopyCounter(2), CopyCounter(3)];
        let mut c = CovariantCow::<CopyCounter, 4>::from_borrowed(&data);
        assert!(c.is_borrowed());

        c.to_mut()[0] = CopyCounter(9);
        assert!(!c.is_borrowed());
        assert!(!c.is_heap());
        let vals: Vec<u32> = c.iter().map(|x| x.0).collect();
        assert_eq!(vals, vec![9, 2, 3]);
        // The borrowed source is untouched.
        assert_eq!(data[0].0, 1);

        let calls = COPY_CLONES.load(Ordering::SeqCst);
        if IS_SPECIALIZED {
            assert_eq!(calls, 0, "T: Copy must take the memcpy path");
        } else {
            assert_eq!(calls, 3, "stable must run the element-wise loop");
        }
    }

    #[test]
    fn cow_to_mut_copy_heap_skips_clone() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        COPY_CLONES.store(0, Ordering::SeqCst);
        let data: Vec<CopyCounter> = (0..10).map(CopyCounter).collect();
        let mut c = CovariantCow::<CopyCounter, 2>::from_borrowed(&data);
        assert!(c.is_borrowed());

        c.to_mut()[0] = CopyCounter(99);
        assert!(c.is_heap());
        assert_eq!(c.len(), 10);
        assert_eq!(c.capacity(), 10);
        let vals: Vec<u32> = c.iter().map(|x| x.0).collect();
        let mut expect: Vec<u32> = (0..10).collect();
        expect[0] = 99;
        assert_eq!(vals, expect);
        assert_eq!(data[0].0, 0);

        let calls = COPY_CLONES.load(Ordering::SeqCst);
        if IS_SPECIALIZED {
            assert_eq!(calls, 0, "T: Copy must take the memcpy path");
        } else {
            assert_eq!(calls, 10, "stable must run the element-wise loop");
        }
    }

    #[test]
    fn cow_try_materialize_copy_skips_clone() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        COPY_CLONES.store(0, Ordering::SeqCst);
        let data: Vec<CopyCounter> = (0..10).map(CopyCounter).collect();
        let mut c = CovariantCow::<CopyCounter, 2>::from_borrowed(&data);
        // `try_reserve` fuses the materialization with the growth, so this
        // exercises `try_materialize`'s heap branch directly.
        c.try_reserve(5).unwrap();
        assert!(!c.is_borrowed());
        assert!(c.is_heap());
        assert!(c.capacity() >= 15);
        let vals: Vec<u32> = c.iter().map(|x| x.0).collect();
        assert_eq!(vals, (0..10).collect::<Vec<u32>>());

        let calls = COPY_CLONES.load(Ordering::SeqCst);
        if IS_SPECIALIZED {
            assert_eq!(calls, 0, "T: Copy must take the memcpy path");
        } else {
            assert_eq!(calls, 10, "stable must run the element-wise loop");
        }
    }

    #[test]
    fn cow_to_mut_non_copy_still_runs_elementwise() {
        let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        NONCOPY_CLONES.store(0, Ordering::SeqCst);
        let data: Vec<NoDropNotCopy> = (0..3).map(NoDropNotCopy).collect();
        let mut c = CovariantCow::<NoDropNotCopy, 8>::from_borrowed(&data);
        c.to_mut()[0] = NoDropNotCopy(42);
        assert_eq!(c.iter().map(|x| x.0).collect::<Vec<u32>>(), vec![42, 1, 2]);
        assert_eq!(NONCOPY_CLONES.load(Ordering::SeqCst), 3);
    }
}

// -------------------------------------------------------------------------- //
// from_elem / resize bulk fill (CloneRepeat)                                  //
// -------------------------------------------------------------------------- //

#[test]
fn vec_from_elem_copy_clone_counts() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    // Inline and heap; the pairing pins both `CloneRepeat` paths.
    let a = CovariantVec::<CopyCounter, 4>::from_elem(CopyCounter(7), 4);
    assert_eq!(a.iter().map(|x| x.0).collect::<Vec<u32>>(), vec![7; 4]);
    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the bulk-copy path");
    } else {
        assert_eq!(calls, 3, "stable must run the element-wise loop");
    }

    COPY_CLONES.store(0, Ordering::SeqCst);
    let b = CovariantVec::<CopyCounter, 2>::from_elem(CopyCounter(9), 5);
    assert!(b.is_heap());
    assert_eq!(b.iter().map(|x| x.0).collect::<Vec<u32>>(), vec![9; 5]);
    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the bulk-copy path");
    } else {
        assert_eq!(calls, 4, "stable must run the element-wise loop");
    }
}

#[test]
fn vec_resize_grow_copy_clone_counts() {
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<CopyCounter, 8>::new();
    v.push(CopyCounter(1));
    v.push(CopyCounter(2));
    v.push(CopyCounter(3));
    assert!(!v.is_heap());
    // 3 -> 6: `Vec::resize` parity = 2 clones + 1 move.
    v.resize(6, CopyCounter(7));
    assert_eq!(
        v.iter().map(|x| x.0).collect::<Vec<u32>>(),
        vec![1, 2, 3, 7, 7, 7]
    );
    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the bulk-copy path");
    } else {
        assert_eq!(calls, 2, "stable must run the element-wise loop");
    }
}

#[cfg(feature = "cow")]
#[test]
fn cow_from_elem_copy_clone_counts() {
    use covariant_vec::CovariantCow;
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    let c = CovariantCow::<CopyCounter, 4>::from_elem(CopyCounter(7), 4);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<u32>>(), vec![7; 4]);
    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the bulk-copy path");
    } else {
        assert_eq!(calls, 3, "stable must run the element-wise loop");
    }
}

#[cfg(feature = "cow")]
#[test]
fn cow_resize_grow_copy_clone_counts() {
    use covariant_vec::CovariantCow;
    let _count_guard = COUNT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    COPY_CLONES.store(0, Ordering::SeqCst);
    let mut c: CovariantCow<CopyCounter, 8> = (0..3).map(CopyCounter).collect();
    c.resize(6, CopyCounter(7));
    assert_eq!(
        c.iter().map(|x| x.0).collect::<Vec<u32>>(),
        vec![0, 1, 2, 7, 7, 7]
    );
    let calls = COPY_CLONES.load(Ordering::SeqCst);
    if IS_SPECIALIZED {
        assert_eq!(calls, 0, "T: Copy must take the bulk-copy path");
    } else {
        assert_eq!(calls, 2, "stable must run the element-wise loop");
    }
}
