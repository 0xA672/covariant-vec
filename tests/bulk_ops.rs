//! Runtime tests for the native bulk operations (append / split_off /
//! splice / extract_if), including drop accounting. Uses the shared
//! `Tracked` double from `common`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantVec;

mod common;

use common::Tracked;

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

// ---- append ---------------------------------------------------------------

#[test]
fn append_inline_to_inline_allocates_nothing() {
    let mut a = CovariantVec::<i32, 4>::new();
    a.extend([1, 2]);
    let mut b = CovariantVec::<i32, 4>::new();
    b.extend([3, 4]);
    a.append(&mut b);
    assert!(!a.is_heap());
    assert_eq!(a.capacity(), 4);
    assert_eq!(&*a, &[1, 2, 3, 4]);
    assert!(b.is_empty() && !b.is_heap());
    assert_eq!(b.capacity(), 4); // storage kept, std parity
}

#[test]
fn append_spills_self_once_and_keeps_other_state() {
    let mut a = CovariantVec::<i32, 2>::new();
    a.extend([1, 2]); // full inline
    let mut b = CovariantVec::<i32, 2>::new();
    b.extend([3, 4, 5, 6]); // heap
    let cap = b.capacity();
    a.append(&mut b);
    assert!(a.is_heap());
    assert_eq!(&*a, &[1, 2, 3, 4, 5, 6]);
    assert!(b.is_heap() && b.is_empty());
    assert_eq!(b.capacity(), cap);
}

#[test]
fn append_zst_counts() {
    let mut a = CovariantVec::<(), 4>::new();
    let mut b = CovariantVec::<(), 4>::new();
    for _ in 0..10 {
        a.push(());
    }
    for _ in 0..5 {
        b.push(());
    }
    a.append(&mut b);
    assert_eq!(a.len(), 15);
    assert!(!a.is_heap());
    assert!(b.is_empty());
}

#[test]
fn append_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut a = CovariantVec::<Tracked, 4>::new();
    let mut b = CovariantVec::<Tracked, 4>::new();
    for i in 0..3 {
        a.push(Tracked::new(&c, i));
    }
    for i in 3..6 {
        b.push(Tracked::new(&c, i));
    }
    a.append(&mut b);
    assert_eq!(c.load(Ordering::SeqCst), 0); // moved, not dropped
    drop(a);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

// ---- split_off ------------------------------------------------------------

#[test]
fn split_off_inline_tail_normalizes() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6); // heap
    let cap = v.capacity();
    let tail = v.split_off(4);
    assert_eq!(&*v, &[0, 1, 2, 3]);
    assert!(v.is_heap()); // self keeps its state
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*tail, &[4, 5]);
    assert!(!tail.is_heap()); // fits inline: normalized
    assert_eq!(tail.capacity(), 4);
}

#[test]
fn split_off_heap_tail_allocates_exactly() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..10);
    let tail = v.split_off(3);
    assert_eq!(&*v, &[0, 1, 2]);
    assert_eq!(&*tail, &[3, 4, 5, 6, 7, 8, 9]);
    assert!(tail.is_heap());
    assert_eq!(tail.capacity(), 7); // exactly the tail length
}

#[test]
fn split_off_zst_and_boundaries() {
    let mut v = CovariantVec::<(), 4>::new();
    for _ in 0..10 {
        v.push(());
    }
    let tail = v.split_off(7);
    assert_eq!(v.len(), 7);
    assert_eq!(tail.len(), 3);
    assert!(!tail.is_heap());

    // `at == len`: an empty result, self untouched.
    let empty = v.split_off(7);
    assert!(empty.is_empty());
    assert_eq!(v.len(), 7);

    // `at == 0`: everything moves out, self is left empty.
    let all = v.split_off(0);
    assert_eq!(all.len(), 7);
    assert!(v.is_empty());
    assert!(v.split_off(0).is_empty());
}

#[test]
fn split_off_non_zst_boundaries() {
    // Inline: both boundary splits stay inline and allocate nothing.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2]);
    let empty = v.split_off(2);
    assert!(empty.is_empty());
    assert!(!empty.is_heap());
    assert_eq!(&*v, &[1, 2]);
    let all = v.split_off(0);
    assert_eq!(&*all, &[1, 2]);
    assert!(!all.is_heap()); // fits inline: normalized
    assert!(v.is_empty());
    assert!(!v.is_heap());

    // Heap: `at == len` leaves self's block and capacity untouched.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    assert!(v.is_heap());
    let cap = v.capacity();
    let empty = v.split_off(6);
    assert!(empty.is_empty());
    assert!(!empty.is_heap());
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);

    // `at == 0`: the whole contents move out; the tail is longer than `N`,
    // so it normalizes to a heap block sized exactly its length.
    let all = v.split_off(0);
    assert_eq!(&*all, &[0, 1, 2, 3, 4, 5]);
    assert!(all.is_heap());
    assert_eq!(all.capacity(), 6);
    assert!(v.is_empty());
    assert!(v.is_heap()); // self keeps its state
    assert_eq!(v.capacity(), cap);
}

#[test]
#[should_panic(expected = "`at` split index (is 2) should be <= len (is 1)")]
fn split_off_out_of_bounds_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.push(1);
    let _ = v.split_off(2);
}

#[test]
fn split_off_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    let tail = v.split_off(4);
    assert_eq!(c.load(Ordering::SeqCst), 0);
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 4);
    drop(tail);
    assert_eq!(c.load(Ordering::SeqCst), 10);
}

// ---- splice ---------------------------------------------------------------

#[test]
fn splice_same_size_refills_in_place() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let cap = v.capacity();
    let drained: Vec<i32> = v.splice(1..5, [10, 11, 12, 13]).collect();
    assert_eq!(drained, vec![1, 2, 3, 4]);
    assert_eq!(&*v, &[0, 10, 11, 12, 13, 5]);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap); // refilled in place: no reallocation
}

#[test]
fn splice_longer_replacement() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let drained: Vec<i32> = v.splice(2..4, 100..106).collect();
    assert_eq!(drained, vec![2, 3]);
    assert_eq!(&*v, &[0, 1, 100, 101, 102, 103, 104, 105, 4, 5]);
}

#[test]
fn splice_non_fused_replacement_still_rolls_back_correctly() {
    // A non-fused replacement returns `None` once (ending phase 1 with an
    // unfilled hole), then yields again (phase 2). The geometry must follow
    // the inserts so the tail is shifted from the *updated* end, not the
    // original drained end.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let mut calls = 0;
    let replacement = std::iter::from_fn(move || {
        calls += 1;
        match calls {
            1 => None,
            2 => Some(102),
            3 => Some(103),
            _ => None,
        }
    });
    let drained: Vec<i32> = v.splice(1..4, replacement).collect();
    assert_eq!(drained, vec![1, 2, 3]);
    assert_eq!(&*v, &[0, 102, 103, 4, 5]);
}

#[test]
fn splice_shorter_replacement() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let cap = v.capacity();
    let drained: Vec<i32> = v.splice(1..5, [9]).collect();
    assert_eq!(drained, vec![1, 2, 3, 4]);
    assert_eq!(&*v, &[0, 9, 5]);
    assert_eq!(v.capacity(), cap); // shrinking the range does not reallocate
}

#[test]
fn splice_lazy_then_drop_keeps_tail() {
    // Yield one removed element, then drop the Splice: the rest of the range
    // is dropped, the replacement is spliced in, and the tail survives.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    {
        let mut sp = v.splice(2..5, [7, 7]);
        assert_eq!(sp.next(), Some(2));
    }
    assert_eq!(&*v, &[0, 1, 7, 7, 5]);
}

#[test]
fn splice_empty_range_is_pure_insert() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let drained: Vec<i32> = v.splice(1..1, [10, 11]).collect();
    assert!(drained.is_empty());
    assert_eq!(&*v, &[1, 10, 11, 2, 3]);
}

#[test]
fn splice_empty_range_many_elements_keeps_tail() {
    // An empty range used to take the one-slot-insert path, which is O(k · n).
    // It now runs the same chunked fill as a longer replacement: one reserve,
    // one tail shift, and `k` writes for an exact hint.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let drained: Vec<i32> = v.splice(1..1, 0..10_000).collect();
    assert!(drained.is_empty());
    assert_eq!(v.len(), 10_003);
    assert_eq!(v[0], 1);
    assert!(v[1..10_001].iter().copied().eq(0..10_000));
    assert_eq!(&v[10_001..], &[2, 3]);
}

#[test]
fn splice_empty_range_zero_hint_grows_in_chunks() {
    // `filter` reports a zero lower bound, so the chunk schedule doubles:
    // 8, 16, 32, … until the iterator is exhausted. The tail must still land
    // after every inserted element.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let drained: Vec<i32> = v.splice(2..2, (0..1000).filter(|_| true)).collect();
    assert!(drained.is_empty());
    assert_eq!(v.len(), 1003);
    assert_eq!(&v[..2], &[1, 2]);
    assert!(v[2..1002].iter().copied().eq(0..1000));
    assert_eq!(&v[1002..], &[3]);
}

#[test]
fn splice_empty_range_replacement_panic_keeps_inserted_prefix() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
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
        // then the iterator panics. The guard compacts the unfilled gap and
        // keeps the written prefix in place.
        drop(v.splice(2..2, replacement));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 1, 101, 102, 103, 104, 2, 3, 4, 5]);
}

#[test]
fn splice_zero_hint_replacement_panic_keeps_prefix_tail_and_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
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
        // Hole 1..3: 101 and 102 refill it in place; the chunked growth path
        // then writes 103..105 into an opened gap before the panic.
        drop(v.splice(1..3, replacement));
    }));
    assert!(res.is_err());
    // The drained elements 1 and 2 were dropped by the Splice; the written
    // replacements are owned by the vector.
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 101, 102, 103, 104, 105, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 2 + 9);
}

#[test]
fn splice_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    {
        let mut sp = v.splice(1..4, [Tracked::new(&c, 100)]);
        let t = sp.next().unwrap();
        assert_eq!(t.val, 1);
        drop(t); // caller owns yielded elements
        assert_eq!(c.load(Ordering::SeqCst), 1);
    }
    // The un-yielded removed elements (2, 3) were dropped by the Splice.
    assert_eq!(c.load(Ordering::SeqCst), 3);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 100, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 7);
}

#[test]
fn splice_replacement_panic_phase1_keeps_written_prefix() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let replacement = std::iter::from_fn(move || {
            n += 1;
            if n == 1 {
                Some(101)
            } else {
                panic!("boom")
            }
        });
        // Hole is 1..4 (three slots); the first replacement is written into
        // the first dead slot, then the iterator panics.
        let mut sp = v.splice(1..4, replacement);
        assert_eq!(sp.next(), Some(1));
    }));
    assert!(res.is_err());
    // The drained remainder (2, 3) was dropped by the Splice; the written
    // replacement is preserved and the tail is shifted over the dead slots.
    assert_eq!(&*v, &[0, 101, 4, 5]);
}

#[test]
fn splice_replacement_panic_phase2_keeps_prefix_and_tail() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
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
        // Hole is 1..3 (two slots): 101 and 102 refill it in place, 103 is
        // inserted past the hole (moving the tail right), then the iterator
        // panics. The rollback must use the *updated* geometry.
        let mut sp = v.splice(1..3, replacement);
        assert_eq!(sp.next(), Some(1));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 101, 102, 103, 3, 4, 5]);
}

#[test]
fn splice_replacement_panic_phase2_batch_keeps_prefix_and_tail() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let res = catch_unwind(AssertUnwindSafe(|| {
        // Hole 1..3 (two slots), six replacement elements: 101 and 102 refill
        // the hole in phase 1; phase 2 batches the remaining four (one tail
        // shift), writes 103 and 104 into the opened gap, then panics. The
        // rollback must drop the unfilled gap and keep the written prefix.
        let replacement = Hinted::new(101, 6, 4);
        let mut sp = v.splice(1..3, replacement);
        assert_eq!(sp.next(), Some(1));
    }));
    assert!(res.is_err());
    assert_eq!(&*v, &[0, 101, 102, 103, 104, 3, 4, 5]);
}

#[test]
fn splice_huge_lying_lower_bound_panics_before_shifting() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let res = catch_unwind(AssertUnwindSafe(|| {
        // The first two elements fill the hole. The chunked path then pulls
        // the third (so the iterator is not exhausted) and consults the hint,
        // which claims `usize::MAX` more: the reservation panics before any
        // geometry changes, and the pulled element is dropped by the unwind.
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
fn splice_non_fused_batch_compacts_partial_gap() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..6);
    let drained: Vec<i32> = v.splice(1..3, NonFusedBatch { calls: 0 }).collect();
    assert_eq!(drained, vec![1, 2]);
    assert_eq!(&*v, &[0, 101, 103, 104, 3, 4, 5]);
}

#[test]
fn splice_lying_lower_bound_batch_keeps_unfilled_gap_dead() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    // Hole 1..3; the replacement claims five more elements but yields three.
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
        drop(drained); // the caller owns the yielded elements
    }
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 100, 101, 102, 3, 4, 5]
    );
    drop(v);
    // Every original and replacement element is dropped exactly once: the
    // unfilled gap holds only stale duplicates, which the rollback reclaims
    // without running destructors.
    assert_eq!(c.load(Ordering::SeqCst), 9);
}

#[test]
fn splice_replacement_panic_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
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
    // Yielded 1 + drained 2 => 2 drops. The three inserted replacements are
    // owned by the vector; nothing was dropped twice.
    assert_eq!(c.load(Ordering::SeqCst), 2);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 101, 102, 103, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 2 + 7);
}

#[test]
fn splice_zst_counts() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPS.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<Z, 4>::new();
    for _ in 0..6 {
        v.push(Z);
    }
    let drained: Vec<Z> = v.splice(1..3, [Z, Z, Z]).collect();
    assert_eq!(drained.len(), 2);
    drop(drained);
    assert_eq!(v.len(), 7);
    assert!(!v.is_heap());
    drop(v);
    assert_eq!(DROPS.load(Ordering::SeqCst), 9);
}

// ---- extract_if -----------------------------------------------------------

#[test]
fn extract_if_keeps_rest_in_order_and_state() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    let cap = v.capacity();
    let removed: Vec<i32> = v.extract_if(|x| *x % 2 == 0).collect();
    assert_eq!(removed, vec![0, 2, 4, 6, 8]);
    assert_eq!(&*v, &[1, 3, 5, 7, 9]);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap); // never reallocates
}

#[test]
fn extract_if_early_drop_keeps_unprocessed() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    {
        let mut it = v.extract_if(|t| t.val >= 2 && t.val < 5);
        let t = it.next().unwrap(); // val 2 extracted
        assert_eq!(t.val, 2);
        drop(t);
        assert_eq!(c.load(Ordering::SeqCst), 1);
    } // dropped without exhausting
    assert_eq!(c.load(Ordering::SeqCst), 1); // only the extracted one dropped
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 1, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

#[test]
fn extract_if_pred_panic_leaves_container_consistent() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
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
        let first = it.next().unwrap(); // yields 1
        assert_eq!(first.val, 1);
        drop(first);
        let _ = it.next(); // examines 2, then panics at 3
    }));
    assert!(res.is_err());
    // The element being examined (3) is kept; 1 stays with the unwound caller.
    assert_eq!(c.load(Ordering::SeqCst), 1);
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 2, 3, 4, 5]
    );
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

#[test]
fn extract_if_zst_counts() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPS.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<Z, 4>::new();
    for _ in 0..10 {
        v.push(Z);
    }
    let mut i = 0;
    let removed: Vec<Z> = v
        .extract_if(|_| {
            let m = i % 2 == 0;
            i += 1;
            m
        })
        .collect();
    assert_eq!(removed.len(), 5);
    assert_eq!(v.len(), 5);
    assert!(!v.is_heap());
    drop(removed);
    drop(v);
    assert_eq!(DROPS.load(Ordering::SeqCst), 10); // one destructor per element
}
