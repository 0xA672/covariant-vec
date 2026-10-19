//! Bulk-operation tests for the optional `CovariantCow` (`cow` feature):
//! append / split_off / splice / extract_if, including the borrowed-aware vs
//! owned-only split and drop accounting.

#![cfg(feature = "cow")]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantCow;

mod common;

use common::Tracked;

/// Deliberately `!Clone`: the `_owned` methods and borrowed `split_off` must
/// compile and work without a `Clone` bound.
#[derive(Debug)]
struct NoClone {
    counter: Arc<AtomicUsize>,
    val: i32,
}

impl NoClone {
    fn new(counter: &Arc<AtomicUsize>, val: i32) -> Self {
        NoClone {
            counter: Arc::clone(counter),
            val,
        }
    }
}

impl Drop for NoClone {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}

/// Replacement iterator with an exact `size_hint` (so `splice` takes the
/// batched phase-2 path) that panics on the `panic_at`-th call to `next`.
struct Hinted {
    next: i32,
    remaining: usize,
    panic_at: usize,
    calls: usize,
}

impl Hinted {
    fn new(next: i32, remaining: usize, panic_at: usize) -> Self {
        Self {
            next,
            remaining,
            panic_at,
            calls: 0,
        }
    }
}

impl Iterator for Hinted {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.calls == self.panic_at {
            panic!("boom");
        }
        if self.remaining == 0 {
            return None;
        }
        self.calls += 1;
        self.remaining -= 1;
        let v = self.next;
        self.next += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Wraps an iterator and reports a fixed (deliberately lying) lower bound, to
/// exercise the batch path's partially-filled-gap rollback.
struct LyingHint<I: Iterator> {
    inner: I,
    hint: usize,
}

impl<I: Iterator> Iterator for LyingHint<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.hint, None)
    }
}

/// `Clone` that panics on a chosen call, counting calls in a shared counter.
/// Not `Copy`, so the element-wise `clone_fill_range` path runs on every
/// feature set.
struct PanicClone {
    calls: Arc<AtomicUsize>,
    panic_at: usize,
    val: i32,
}

impl Clone for PanicClone {
    fn clone(&self) -> Self {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == self.panic_at {
            panic!("clone boom");
        }
        PanicClone {
            calls: Arc::clone(&self.calls),
            panic_at: self.panic_at,
            val: self.val,
        }
    }
}

// ---- append ---------------------------------------------------------------

#[test]
fn append_borrowed_other_clones_directly_and_narrows_it() {
    let src = [3, 4, 5];
    let mut other = CovariantCow::<i32, 4>::from_borrowed(&src);
    let mut a = CovariantCow::<i32, 4>::new();
    a.push(1);
    a.push(2);
    a.append(&mut other);
    assert_eq!(&*a, &[1, 2, 3, 4, 5]);
    assert!(!a.is_borrowed());
    assert!(other.is_borrowed() && other.is_empty());
    assert_eq!(other.as_borrowed().unwrap(), &[]);
    assert_eq!(src, [3, 4, 5]); // the borrowed source is untouched
}

#[test]
fn append_both_borrowed_fuses_materialization() {
    let left = [1, 2];
    let right = [3, 4, 5];
    let mut a = CovariantCow::<i32, 4>::from_borrowed(&left);
    let mut b = CovariantCow::<i32, 4>::from_borrowed(&right);
    a.append(&mut b);
    assert_eq!(&*a, &[1, 2, 3, 4, 5]);
    assert!(!a.is_borrowed());
    assert!(b.is_borrowed() && b.is_empty());
    assert_eq!(left, [1, 2]);
    assert_eq!(right, [3, 4, 5]);
}

#[test]
fn append_owned_needs_no_clone() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut a = CovariantCow::<NoClone, 4>::new();
    a.push_owned(NoClone::new(&c, 0));
    let mut b = CovariantCow::<NoClone, 4>::new();
    b.push_owned(NoClone::new(&c, 1));
    b.push_owned(NoClone::new(&c, 2));
    a.append_owned(&mut b);
    assert_eq!(a.iter().map(|x| x.val).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!(b.is_empty());
    drop(a);
    assert_eq!(c.load(Ordering::SeqCst), 3);
}

#[test]
fn append_zst_counts_and_keeps_borrow_empty() {
    let src = [(), (), ()];
    let mut other = CovariantCow::<(), 4>::from_borrowed(&src);
    let mut a = CovariantCow::<(), 4>::new();
    for _ in 0..10 {
        a.push(());
    }
    a.append(&mut other);
    assert_eq!(a.len(), 13);
    assert!(other.is_borrowed() && other.is_empty());
}

#[test]
fn append_borrowed_panicking_clone_keeps_self_prefix_and_other_intact() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mk = |val| PanicClone {
        calls: Arc::clone(&calls),
        panic_at: 1,
        val,
    };
    let src = [mk(10), mk(11), mk(12)];
    let mut other = CovariantCow::<PanicClone, 4>::from_borrowed(&src);
    let mut a = CovariantCow::<PanicClone, 4>::new();
    a.push(mk(0));
    let res = catch_unwind(AssertUnwindSafe(|| a.append(&mut other)));
    assert!(res.is_err());
    // The first clone was published; the second panicked.
    assert_eq!(a.len(), 2);
    assert_eq!(a.iter().map(|x| x.val).collect::<Vec<_>>(), vec![0, 10]);
    // `other` was not narrowed: the borrow is still full-length.
    assert!(other.is_borrowed());
    assert_eq!(other.len(), 3);
}

#[test]
fn cow_extend_from_slice_owned_borrowed_and_zst() {
    // Owned inline, then spill.
    let mut c = CovariantCow::<u32, 4>::new();
    c.extend_from_slice(&[1, 2]);
    assert!(!c.is_borrowed() && !c.is_heap());
    c.extend_from_slice(&[3, 4, 5]);
    assert!(c.is_heap());
    assert_eq!(&*c, &[1, 2, 3, 4, 5]);

    // Borrowed: materializes once; the source borrow is untouched.
    let data = [10, 11, 12];
    let mut b = CovariantCow::<u32, 4>::from_borrowed(&data);
    b.extend_from_slice(&data);
    assert!(!b.is_borrowed());
    assert_eq!(&*b, &[10, 11, 12, 10, 11, 12]);
    assert_eq!(data, [10, 11, 12]);

    // An empty slice leaves a borrowed cow borrowed.
    let mut e = CovariantCow::<u32, 4>::from_borrowed(&data);
    e.extend_from_slice(&[]);
    assert!(e.is_borrowed());
    assert_eq!(e.as_borrowed().unwrap(), &data);

    // ZST: the count is the ownership.
    let mut z = CovariantCow::<(), 4>::new();
    z.extend_from_slice(&[(), (), ()]);
    assert_eq!(z.len(), 3);
    assert!(!z.is_borrowed());
}

// ---- split_off ------------------------------------------------------------

#[test]
fn split_off_borrowed_is_zero_copy() {
    let data = [10, 20, 30, 40, 50];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let tail = c.split_off(2);
    assert!(c.is_borrowed());
    assert!(tail.is_borrowed());
    assert_eq!(&*c, &[10, 20]);
    assert_eq!(&*tail, &[30, 40, 50]);
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), data.as_ptr());
    assert_eq!(
        tail.as_borrowed().unwrap().as_ptr(),
        data.as_ptr().wrapping_add(2)
    );
}

#[test]
fn split_off_borrowed_at_len_keeps_the_tail_borrowed() {
    let data = [10, 20, 30, 40];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let tail = c.split_off(data.len());
    assert!(c.is_borrowed() && tail.is_borrowed());
    // `split_off(len)` keeps the whole slice in `self`; only the tail is empty.
    assert_eq!(&*c, &data[..]);
    assert!(tail.is_empty());
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), data.as_ptr());
    assert_eq!(
        tail.as_borrowed().unwrap().as_ptr(),
        data.as_ptr().wrapping_add(data.len())
    );
}

#[test]
fn split_off_borrowed_needs_no_clone() {
    let data = [
        NoClone::new(&Arc::new(AtomicUsize::new(0)), 1),
        NoClone::new(&Arc::new(AtomicUsize::new(0)), 2),
        NoClone::new(&Arc::new(AtomicUsize::new(0)), 3),
    ];
    let mut c = CovariantCow::<NoClone, 4>::from_borrowed(&data);
    let tail = c.split_off(1);
    assert_eq!(c.len(), 1);
    assert_eq!(tail.len(), 2);
    assert!(c.is_borrowed() && tail.is_borrowed());
}

#[test]
fn split_off_owned_normalizes_like_clone() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let tail = v.split_off(4);
    assert_eq!(&*v, &[0, 1, 2, 3]);
    assert!(v.is_heap());
    assert_eq!(&*tail, &[4, 5]);
    assert!(!tail.is_borrowed() && !tail.is_heap());
    assert_eq!(tail.capacity(), 4);

    let mut w = CovariantCow::<i32, 2>::new();
    for i in 0..10 {
        w.push(i);
    }
    let tail = w.split_off(3);
    assert_eq!(&*w, &[0, 1, 2]);
    assert_eq!(&*tail, &[3, 4, 5, 6, 7, 8, 9]);
    assert!(tail.is_heap());
    assert_eq!(tail.capacity(), 7);
}

#[test]
fn split_off_owned_needs_no_clone() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<NoClone, 4>::new();
    for i in 0..6 {
        v.push_owned(NoClone::new(&c, i));
    }
    let tail = v.split_off(2);
    assert_eq!(v.iter().map(|x| x.val).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(
        tail.iter().map(|x| x.val).collect::<Vec<_>>(),
        vec![2, 3, 4, 5]
    );
    drop(v);
    drop(tail);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

#[test]
#[should_panic(expected = "`at` split index (is 2) should be <= len (is 1)")]
fn split_off_out_of_bounds_panics() {
    let mut v = CovariantCow::<i32, 4>::new();
    v.push(1);
    let _ = v.split_off(2);
}

// ---- splice ---------------------------------------------------------------

#[test]
fn splice_borrowed_materializes_and_splices() {
    let data = [0, 1, 2, 3, 4, 5];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(1..4, [10, 11]).collect();
    assert_eq!(drained, vec![1, 2, 3]);
    assert_eq!(&*c, &[0, 10, 11, 4, 5]);
    assert!(!c.is_borrowed());
    assert_eq!(data, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn splice_empty_range_with_empty_replacement_keeps_borrow() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(1..1, std::iter::empty()).collect();
    assert!(drained.is_empty());
    assert!(c.is_borrowed());
    assert_eq!(&*c, &[1, 2, 3]);
}

#[test]
fn splice_empty_range_with_replacement_materializes() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(1..1, [10, 11]).collect();
    assert!(drained.is_empty());
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[1, 10, 11, 2, 3]);
}

#[test]
fn cow_splice_empty_range_many_elements_keeps_tail() {
    // The upper hint is non-empty, so the borrowed cow is materialized first;
    // the fill then runs the chunked empty-range path on the owned cow.
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(1..1, 0..10_000).collect();
    assert!(drained.is_empty());
    assert!(!c.is_borrowed());
    assert_eq!(c.len(), 10_003);
    assert_eq!(c[0], 1);
    assert!(c[1..10_001].iter().copied().eq(0..10_000));
    assert_eq!(&c[10_001..], &[2, 3]);
    assert_eq!(data, [1, 2, 3]);
}

#[test]
fn cow_splice_empty_range_zero_hint_grows_in_chunks() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(2..2, (0..1000).filter(|_| true)).collect();
    assert!(drained.is_empty());
    assert!(!c.is_borrowed());
    assert_eq!(c.len(), 1003);
    assert_eq!(&c[..2], &[1, 2]);
    assert!(c[2..1002].iter().copied().eq(0..1000));
    assert_eq!(&c[1002..], &[3]);
}

#[test]
fn cow_splice_empty_range_replacement_panic_keeps_inserted_prefix() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let replacement = std::iter::from_fn(move || {
            n += 1;
            if n <= 4 {
                Some(100 + n)
            } else {
                panic!("boom");
            }
        });
        // Empty range at 2: the chunked path opens a gap, writes 101..104,
        // then the iterator panics; the guard keeps the prefix in place.
        drop(v.splice(2..2, replacement));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 1, 101, 102, 103, 104, 2, 3, 4, 5]);
}

#[test]
fn cow_splice_zero_hint_replacement_panic_keeps_prefix_tail_and_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    let inner = Arc::clone(&c);
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let replacement = std::iter::from_fn(move || {
            n += 1;
            if n <= 5 {
                Some(Tracked::new(&inner, 100 + n))
            } else {
                panic!("boom");
            }
        });
        drop(v.splice(1..3, replacement));
    }));
    assert!(res.is_err());
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 101, 102, 103, 104, 105, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 2 + 9);
}

/// The empty-range borrowed shortcut trusts `size_hint().1 == Some(0)`.
/// A replacement that reports zero and then yields is a logic error, not UB:
/// the `CowSplice` must panic with a message naming the contract and leave
/// the borrowed cow untouched (the `Drop` path has no `T: Clone`, so it
/// cannot materialize lazily).
#[test]
fn splice_borrowed_empty_range_lying_size_hint_panics() {
    struct Liar;

    impl Iterator for Liar {
        type Item = i32;

        fn next(&mut self) -> Option<i32> {
            Some(10)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(0))
        }
    }

    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let res = catch_unwind(AssertUnwindSafe(|| {
        let _ = c.splice(1..1, Liar);
    }));
    let payload = res.unwrap_err();
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("splice replacement yielded an element after reporting an empty size hint")
    );
    assert!(c.is_borrowed());
    assert_eq!(&*c, &[1, 2, 3]);
    assert_eq!(data, [1, 2, 3]);
}

#[test]
fn splice_owned_needs_no_clone() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<NoClone, 4>::new();
    for i in 0..6 {
        v.push_owned(NoClone::new(&c, i));
    }
    let drained: Vec<NoClone> = v.splice_owned(1..3, [NoClone::new(&c, 100)]).collect();
    assert_eq!(drained.len(), 2);
    assert_eq!(
        v.iter().map(|x| x.val).collect::<Vec<_>>(),
        vec![0, 100, 3, 4, 5]
    );
    drop(drained);
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 7);
}

#[test]
fn splice_replacement_panic_keeps_prefix_and_tail() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let replacement = std::iter::from_fn(move || {
            n += 1;
            if n <= 3 {
                Some(100 + n)
            } else {
                panic!("boom")
            }
        });
        let mut sp = v.splice(1..3, replacement);
        assert_eq!(sp.next(), Some(1));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 101, 102, 103, 3, 4, 5]);
}

#[test]
fn cow_splice_borrowed_longer_replacement_batches_after_materializing() {
    let data = [0, 1, 2, 3, 4, 5];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let drained: Vec<i32> = c.splice(1..3, 100..106).collect();
    assert_eq!(drained, vec![1, 2]);
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[0, 100, 101, 102, 103, 104, 105, 3, 4, 5]);
    assert_eq!(data, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn cow_splice_replacement_panic_phase2_batch_keeps_prefix_and_tail() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // Hole 1..3; phase 1 writes 101 and 102, phase 2 shifts the tail by
        // four, writes 103 and 104, then the iterator panics.
        let replacement = Hinted::new(101, 6, 4);
        let mut sp = v.splice(1..3, replacement);
        assert_eq!(sp.next(), Some(1));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 101, 102, 103, 104, 3, 4, 5]);
}

#[test]
fn cow_splice_huge_lying_lower_bound_panics_before_shifting() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // The first two elements fill the hole; the chunked path then pulls
        // the third (so the iterator is not exhausted) and consults the hint,
        // which claims `usize::MAX` more: the reservation panics before any
        // geometry changes.
        let replacement = LyingHint {
            inner: [101, 102, 103].into_iter(),
            hint: usize::MAX,
        };
        drop(v.splice(1..3, replacement));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 101, 102, 3, 4, 5]);
}

/// Non-fused replacement: ends phase 1 early (leaving one hole slot dead),
/// then yields again in phase 2 so the batch opens a gap wider than the hole.
struct NonFusedBatch {
    calls: usize,
}

impl Iterator for NonFusedBatch {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.calls += 1;
        match self.calls {
            1 => Some(101),
            2 => None,
            3 => Some(103),
            4 => Some(104),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (3, None)
    }
}

#[test]
fn cow_splice_non_fused_batch_compacts_partial_gap() {
    let mut v = CovariantCow::<i32, 4>::new();
    for i in 0..6 {
        v.push(i);
    }
    let drained: Vec<i32> = v.splice(1..3, NonFusedBatch { calls: 0 }).collect();
    assert_eq!(drained, vec![1, 2]);
    assert_eq!(&*v, &[0, 101, 103, 104, 3, 4, 5]);
}

#[test]
fn cow_splice_lying_lower_bound_batch_keeps_unfilled_gap_dead() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    let replacement = LyingHint {
        inner: vec![
            Tracked::new(&c, 100),
            Tracked::new(&c, 101),
            Tracked::new(&c, 102),
        ]
        .into_iter(),
        hint: 5,
    };
    {
        let drained: Vec<Tracked> = v.splice(1..3, replacement).collect();
        assert_eq!(
            drained.iter().map(|t| t.val).collect::<Vec<_>>(),
            vec![1, 2]
        );
        drop(drained);
    }
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 100, 101, 102, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 9);
}

#[test]
fn splice_panic_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    let inner_c = Arc::clone(&c);
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let replacement = std::iter::from_fn(move || {
            n += 1;
            if n <= 3 {
                Some(Tracked::new(&inner_c, 100 + n))
            } else {
                panic!("boom")
            }
        });
        let mut sp = v.splice(1..3, replacement);
        let t = sp.next().unwrap();
        assert_eq!(t.val, 1);
        drop(t);
    }));
    assert!(res.is_err());
    // Yielded 1 + drained 2 were dropped; the three replacements live in v.
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 101, 102, 103, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 2 + 7);
}

// ---- extract_if -----------------------------------------------------------

#[test]
fn extract_if_borrowed_materializes() {
    let data = [0, 1, 2, 3, 4, 5];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let removed: Vec<i32> = c.extract_if(|x| *x % 2 == 0).collect();
    assert_eq!(removed, vec![0, 2, 4]);
    assert_eq!(&*c, &[1, 3, 5]);
    assert!(!c.is_borrowed());
    assert_eq!(data, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn extract_if_owned_needs_no_clone() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<NoClone, 4>::new();
    for i in 0..6 {
        v.push_owned(NoClone::new(&c, i));
    }
    let removed: Vec<NoClone> = v.extract_if_owned(|x| x.val % 2 == 0).collect();
    assert_eq!(removed.len(), 3);
    assert_eq!(v.iter().map(|x| x.val).collect::<Vec<_>>(), vec![1, 3, 5]);
    drop(removed);
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

#[test]
fn cow_splice_zst_counts() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPS.store(0, Ordering::SeqCst);
    let mut v = CovariantCow::<Z, 4>::new();
    for _ in 0..6 {
        v.push_owned(Z);
    }
    let drained: Vec<Z> = v.splice_owned(1..3, [Z, Z, Z]).collect();
    assert_eq!(drained.len(), 2);
    drop(drained);
    assert_eq!(v.len(), 7);
    assert!(!v.is_heap());
    drop(v);
    assert_eq!(DROPS.load(Ordering::SeqCst), 9);
}

#[test]
fn cow_extract_if_zst_counts() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPS.store(0, Ordering::SeqCst);
    let mut v = CovariantCow::<Z, 4>::new();
    for _ in 0..10 {
        v.push_owned(Z);
    }
    let mut i = 0;
    let removed: Vec<Z> = v
        .extract_if_owned(|_| {
            let m = i % 2 == 0;
            i += 1;
            m
        })
        .collect();
    assert_eq!(removed.len(), 5);
    assert_eq!(v.len(), 5);
    drop(removed);
    drop(v);
    assert_eq!(DROPS.load(Ordering::SeqCst), 10);
}

#[test]
fn extract_if_pred_panic_leaves_cow_consistent() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantCow::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut it = v.extract_if(|t| {
            if t.val == 3 {
                panic!("boom")
            }
            t.val % 2 == 1
        });
        let first = it.next().unwrap();
        assert_eq!(first.val, 1);
        drop(first);
        let _ = it.next(); // examines 2, then panics at 3
    }));
    assert!(res.is_err());
    assert_eq!(c.load(Ordering::SeqCst), 1);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 2, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}
