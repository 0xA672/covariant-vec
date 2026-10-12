//! Tests for `retain`/`retain_mut` and `drain`, with drop accounting and
//! panic-safety checks.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantVec;

#[derive(Clone, Debug)]
struct Tracked {
    counter: Arc<AtomicUsize>,
    val: i32,
}

impl Tracked {
    fn new(counter: &Arc<AtomicUsize>, val: i32) -> Self {
        Tracked {
            counter: Arc::clone(counter),
            val,
        }
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}

/// An element whose destructor panics when `boom` is set.
struct PanicDrop {
    val: i32,
    ctr: Arc<AtomicUsize>,
    boom: bool,
}

impl Drop for PanicDrop {
    fn drop(&mut self) {
        assert!(!self.boom, "intentional drop panic");
        self.ctr.fetch_add(1, Ordering::SeqCst);
    }
}

// -------------------------------------------------------------------------- //
// retain / retain_mut                                                        //
// -------------------------------------------------------------------------- //

#[test]
fn retain_inline_and_heap() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    assert!(v.is_heap());
    v.retain(|x| x % 2 == 0);
    assert_eq!(&*v, &[0, 2, 4, 6, 8]);
    assert!(v.is_heap()); // in-place: never reallocates, never shrinks

    let mut w = CovariantVec::<i32, 8>::new();
    w.extend([1, 9, 2, 8, 3]);
    w.retain(|x| *x < 5);
    assert_eq!(&*w, &[1, 2, 3]);
    assert!(!w.is_heap());
}

#[test]
fn retain_mut_visits_each_once_and_can_mutate() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4, 5, 6]);
    let visits = std::cell::Cell::new(0);
    v.retain_mut(|x| {
        visits.set(visits.get() + 1);
        *x *= 10;
        *x != 40 // removes the element that used to be 4
    });
    assert_eq!(visits.get(), 6);
    assert_eq!(&*v, &[10, 20, 30, 50, 60]);
}

#[test]
fn retain_all_false_keeps_state_and_capacity() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    v.retain(|_| false);
    assert!(v.is_empty());
    assert!(v.is_heap());
    assert!(v.capacity() >= 10);
}

#[test]
fn retain_drop_accounting() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    v.retain(|t| t.val % 3 != 0); // removes 0, 3, 6, 9
    assert_eq!(c.load(Ordering::SeqCst), 4);
    assert!(v.iter().map(|t| t.val).eq([1, 2, 4, 5, 7, 8]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 10); // each element dropped exactly once
}

#[test]
fn retain_zst_and_n_zero() {
    let mut v = CovariantVec::<(), 4>::new();
    for _ in 0..100 {
        v.push(());
    }
    let mut seen = 0;
    v.retain(|_| {
        seen += 1;
        seen % 2 == 0
    });
    assert_eq!(v.len(), 50);
    assert!(!v.is_heap());

    let mut w = CovariantVec::<i32, 0>::new();
    w.extend(0..10);
    w.retain(|x| x % 2 == 1);
    assert_eq!(&*w, &[1, 3, 5, 7, 9]);
}

#[test]
fn retain_closure_panic() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        // Examines 0..=4: keeps 0, 2, 4; drops 1, 3; panics at 5.
        v.retain(|t| {
            if t.val == 5 {
                panic!("closure panic");
            }
            t.val % 2 == 0
        });
    }));
    assert!(r.is_err());
    // The guard shifted the untouched tail [5..10) down behind the kept
    // prefix, and the length is consistent.
    assert_eq!(c.load(Ordering::SeqCst), 2); // only 1 and 3 were dropped
    assert_eq!(v.len(), 8);
    assert!(v.iter().map(|t| t.val).eq([0, 2, 4, 5, 6, 7, 8, 9]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 10); // every element dropped exactly once
}

#[test]
fn retain_destructor_panic() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<PanicDrop, 4>::new();
    for i in 0..10 {
        v.push(PanicDrop {
            val: i,
            ctr: Arc::clone(&c),
            boom: i == 5,
        });
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        v.retain(|t| t.val != 5); // removing 5 makes its destructor panic
    }));
    assert!(r.is_err());
    // The half-dropped element was overwritten (leaked), never re-dropped;
    // the vector holds the other nine, valid, exactly once each.
    assert_eq!(c.load(Ordering::SeqCst), 0); // 5 never finished dropping
    assert_eq!(v.len(), 9);
    assert!(v.iter().map(|t| t.val).eq([0, 1, 2, 3, 4, 6, 7, 8, 9]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 9);
}

// -------------------------------------------------------------------------- //
// drain                                                                      //
// -------------------------------------------------------------------------- //

#[test]
fn drain_range_and_tail_restore() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    assert!(v.is_heap());
    {
        let mut d = v.drain(3..7);
        assert_eq!(d.len(), 4);
        assert_eq!(
            d.as_slice().iter().map(|t| t.val).collect::<Vec<_>>(),
            vec![3, 4, 5, 6]
        );
        assert_eq!(d.next_back().map(|t| t.val), Some(6));
        assert_eq!(d.next().map(|t| t.val), Some(3));
        assert_eq!(d.len(), 2);
        assert_eq!(d.map(|t| t.val).collect::<Vec<_>>(), vec![4, 5]);
    } // exhausted: the tail is restored
    assert_eq!(c.load(Ordering::SeqCst), 4);
    assert_eq!(v.len(), 6);
    assert!(v.is_heap()); // capacity preserved
    assert!(v.iter().map(|t| t.val).eq([0, 1, 2, 7, 8, 9]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 10);
}

#[test]
fn drain_partial_drop_restores_tail() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    {
        let mut d = v.drain(2..6);
        assert_eq!(d.next().map(|t| t.val), Some(2));
        assert_eq!(d.next_back().map(|t| t.val), Some(5));
        // dropped here with [3, 4] unconsumed: `Drain::drop` drops them
    }
    assert_eq!(c.load(Ordering::SeqCst), 4); // 2, 5 (caller) + 3, 4 (drain drop)
    assert_eq!(v.len(), 6);
    assert!(v.iter().map(|t| t.val).eq([0, 1, 6, 7, 8, 9]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 10);
}

#[test]
fn drain_stays_inline() {
    let mut v = CovariantVec::<i32, 8>::new();
    v.extend([1, 2, 3, 4, 5]);
    let rest: Vec<i32> = v.drain(1..4).collect();
    assert_eq!(rest, vec![2, 3, 4]);
    assert_eq!(&*v, &[1, 5]);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 8);
}

#[test]
fn drain_all_keeps_capacity() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    assert_eq!(v.drain(..).count(), 10);
    assert!(v.is_empty());
    assert!(v.is_heap());
    assert!(v.capacity() >= 10);
    v.extend(100..105); // still usable
    assert_eq!(&*v, &[100, 101, 102, 103, 104]);
}

#[test]
fn drain_empty_range_is_noop() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 8>::new();
    for i in 0..6 {
        v.push(Tracked::new(&c, i));
    }
    {
        let d = v.drain(2..2);
        assert_eq!(d.len(), 0);
        assert_eq!(d.count(), 0);
    }
    assert_eq!(c.load(Ordering::SeqCst), 0);
    assert_eq!(v.len(), 6);
    assert!(v.iter().map(|t| t.val).eq(0..6));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 6);
}

#[test]
fn drain_as_mut_slice() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4, 5]);
    {
        let mut d = v.drain(1..4);
        assert_eq!(d.as_mut_slice(), &mut [2, 3, 4]);
        d.as_mut_slice()[0] = 20;
        assert_eq!(d.next(), Some(20));
        assert_eq!(d.next_back(), Some(4));
        assert_eq!(d.next(), Some(3));
        assert_eq!(d.next(), None);
    }
    assert_eq!(&*v, &[1, 5]);
}

#[test]
fn drain_zst() {
    let mut v = CovariantVec::<(), 4>::new();
    v.extend(core::iter::repeat(()).take(100));
    assert_eq!(v.drain(10..90).count(), 80);
    assert_eq!(v.len(), 20);
    assert!(!v.is_heap());
}

#[test]
fn drain_n_zero() {
    let mut v = CovariantVec::<i32, 0>::new();
    v.extend([1, 2, 3]);
    let d = v.drain(1..);
    assert_eq!(d.as_slice(), &[2, 3]);
    drop(d);
    assert_eq!(&*v, &[1]);
}

#[test]
#[should_panic(expected = "slice index starts at 2 but ends at 1")]
#[allow(clippy::reversed_empty_ranges)] // the reversed range is the point
fn drain_start_after_end_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let _ = v.drain(2..1);
}

#[test]
#[should_panic(expected = "range end index 4 out of range for slice of length 3")]
fn drain_end_out_of_bounds_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let _ = v.drain(..4);
}

/// `drain(5..)` on a len-3 vector: the *start* is past the end, which std
/// reports as a start-out-of-range (not as a start-vs-end ordering error).
#[test]
#[should_panic(expected = "range start index 5 out of range for slice of length 3")]
fn drain_start_out_of_bounds_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let _ = v.drain(5..);
}

/// This test exists to make the *reason* for the move-out `Drain` design a
/// compile-visible artifact. It asserts the user-observable consequence that a
/// borrowed implementation could not provide: `mem::forget(drain)` leaves the
/// caller's vector in a valid, empty, usable state.
///
/// If a future refactor changes `Drain` to borrow the container, this test
/// will fail and force the refactor to either re-earn the guarantee or
/// explicitly document its removal.
#[test]
fn drain_forget_leaves_a_valid_placeholder() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let d = v.drain(1..2);
    std::mem::forget(d); // leaks the taken container (documented behaviour)
                         // The caller's vector is the empty placeholder — valid and usable.
    assert!(v.is_empty());
    assert!(!v.is_heap());
    v.push(9);
    assert_eq!(&*v, &[9]);
}

/// An **empty** range is a true no-op: it never takes the container, so
/// forgetting it leaks nothing and leaves the vector fully intact — the
/// opposite of the non-empty case above. This also pins the "no container-sized
/// memcpy for an empty drain" optimization.
#[test]
fn drain_empty_range_forget_leaves_vector_intact() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let d = v.drain(1..1);
    std::mem::forget(d);
    assert_eq!(&*v, &[1, 2, 3]);
    assert!(!v.is_heap());
}

#[test]
fn drain_destructor_panic_during_unwind() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<PanicDrop, 4>::new();
    for i in 0..10 {
        v.push(PanicDrop {
            val: i,
            ctr: Arc::clone(&c),
            boom: i == 7,
        });
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut d = v.drain(5..9); // yields 5, 6, 7, 8
        assert_eq!(d.next().unwrap().val, 5); // dropped at end of statement
        assert_eq!(d.next().unwrap().val, 6);
        let seven = d.next().unwrap();
        assert_eq!(seven.val, 7);
        drop(seven); // panics; `d` is then dropped *during unwinding*
    }));
    assert!(r.is_err());
    // `d`'s emergency drop: dropped 8 (the only unconsumed element), rolled
    // the tail [9] back behind [0..5], handed the container back.
    assert_eq!(c.load(Ordering::SeqCst), 3); // 5, 6 (caller) + 8 (drain drop); 7 leaked
    assert_eq!(v.len(), 6);
    assert!(v.iter().map(|t| t.val).eq([0, 1, 2, 3, 4, 9]));
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 9); // 10 originals - 1 leaked
}

// -------------------------------------------------------------------------- //
// truncate / clear destructor-panic safety                                   //
// -------------------------------------------------------------------------- //

/// An element that increments its *own* slot in a shared table on drop (before
/// an optional panic), so a test can detect any element being dropped twice.
/// The increment precedes the panic, so even the panicking element is counted.
struct PerElemDrop {
    idx: usize,
    counts: Arc<Vec<AtomicUsize>>,
    boom: bool,
}

impl Drop for PerElemDrop {
    fn drop(&mut self) {
        self.counts[self.idx].fetch_add(1, Ordering::SeqCst);
        assert!(!self.boom, "intentional drop panic");
    }
}

fn per_elem_vec(
    len: usize,
    boom_at: usize,
) -> (CovariantVec<'static, PerElemDrop, 4>, Arc<Vec<AtomicUsize>>) {
    let counts = Arc::new((0..len).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let mut v = CovariantVec::new();
    for i in 0..len {
        v.push(PerElemDrop {
            idx: i,
            counts: Arc::clone(&counts),
            boom: i == boom_at,
        });
    }
    (v, counts)
}

#[test]
fn truncate_destructor_panic_leaks_not_double_drops() {
    let (mut v, counts) = per_elem_vec(10, 7);
    let r = catch_unwind(AssertUnwindSafe(|| v.truncate(3)));
    assert!(r.is_err());
    // The new length was published *before* the destructors ran, so the
    // container no longer claims the truncated slots.
    assert_eq!(v.len(), 3);
    drop(v);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    assert!(seen.iter().all(|&n| n <= 1), "double drop: {seen:?}");
    // The kept prefix is dropped exactly once by `drop(v)`.
    assert!(seen[..3].iter().all(|&n| n == 1), "{seen:?}");
}

#[test]
fn clear_destructor_panic_leaks_not_double_drops() {
    let (mut v, counts) = per_elem_vec(10, 7);
    let r = catch_unwind(AssertUnwindSafe(|| v.clear()));
    assert!(r.is_err());
    assert_eq!(v.len(), 0);
    drop(v);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    assert!(seen.iter().all(|&n| n <= 1), "double drop: {seen:?}");
}

/// A panic from a **not-yet-yielded** element's destructor *inside* `Drain`'s
/// own drop loop (as opposed to the caller's loop) must still roll the
/// container back during unwinding: the tail is shifted over the drained region
/// and the elements whose destructors have not run yet are *leaked* (their
/// destructors skipped), never dropped twice. This pins the documented
/// unwind-safety of `Drain::drop_impl`.
///
/// The element owns **no allocation** (its counter lives in a function-local
/// `static`), so the deliberate leak-on-panic does not trip Miri's leak checker
/// — the leaked slot's bytes are reclaimed when the vector's buffer is freed;
/// only the destructor is skipped.
#[test]
fn drain_drop_loop_destructor_panic_rolls_back() {
    static COUNTS: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];

    struct El {
        idx: usize,
        boom: bool,
    }
    impl Drop for El {
        fn drop(&mut self) {
            COUNTS[self.idx].fetch_add(1, Ordering::SeqCst);
            assert!(!self.boom, "intentional drop panic");
        }
    }

    let mut v = CovariantVec::<El, 4>::new();
    for i in 0..10 {
        v.push(El {
            idx: i,
            boom: i == 7,
        }); // element 7 panics on drop
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        // Consume nothing: dropping `d` at the end of the closure runs the loop
        // over 5, 6, 7 (panic!), leaving 8 unreached.
        let _d = v.drain(5..9);
    }));
    assert!(r.is_err());
    // The rollback ran during unwinding: tail [9] shifted behind [0..5].
    assert_eq!(v.len(), 6);
    assert!(v.iter().map(|t| t.idx).eq([0, 1, 2, 3, 4, 9]));
    drop(v);
    let seen: Vec<usize> = COUNTS[..10]
        .iter()
        .map(|c| c.load(Ordering::SeqCst))
        .collect();
    assert!(seen.iter().all(|&n| n <= 1), "double drop: {seen:?}");
    assert_eq!(&seen[..5], &[1, 1, 1, 1, 1]); // kept prefix dropped once
    assert_eq!(seen[5], 1); // dropped by the loop before the panic
    assert_eq!(seen[6], 1); // dropped by the loop before the panic
    assert_eq!(seen[7], 1); // the panicking element counts itself before panicking
    assert_eq!(seen[8], 0); // never reached -> leaked (no double drop)
    assert_eq!(seen[9], 1); // shifted tail, dropped once
}

// -------------------------------------------------------------------------- //
// drain: heap-state fast path                                                 //
// -------------------------------------------------------------------------- //
// `drain` hands a heap-backed container over by swapping only `(ptr, cap)` +
// the tagged length (and swaps it back on rollback) instead of `mem::take`-ing
// the whole struct. These tests pin every *observable* consequence of that
// state swap: the state, the capacity, the contents, and the exact destructor
// accounting.

#[test]
fn drain_heap_state_swap_preserves_capacity_and_drop_accounting() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&ctr, i));
    }
    assert!(v.is_heap());
    let cap = v.capacity();

    let mut d = v.drain(3..7);
    // Consume from both ends, then drop the rest: every drained element must
    // be dropped exactly once, by whoever ends up owning it.
    let first = d.next().unwrap();
    let last = d.next_back().unwrap();
    assert_eq!(first.val, 3);
    assert_eq!(last.val, 6);
    assert_eq!(d.as_slice().len(), 2);
    assert_eq!((d.as_slice()[0].val, d.as_slice()[1].val), (4, 5));
    // The yielded pair is still alive in `first`/`last`; dropping the drain
    // drops only the two it still holds.
    drop(d);
    assert_eq!(ctr.load(Ordering::SeqCst), 2);
    drop(first);
    drop(last);
    assert_eq!(ctr.load(Ordering::SeqCst), 4);

    // Rollback restored the heap state *and* the original block (capacity).
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap);
    let vals: Vec<i32> = v.iter().map(|t| t.val).collect();
    assert_eq!(vals, vec![0, 1, 2, 7, 8, 9]);

    drop(v);
    assert_eq!(ctr.load(Ordering::SeqCst), 10);
}

// The placeholder a heap-state `drain` leaves behind is only observable after
// `mem::forget`-ing the iterator (while it lives, it holds `&mut` to the slot),
// and that *intentionally* leaks the container's heap block. Miri's leak
// checker flags any leaked allocation, so this test is skipped there; the
// leak-free rollback paths are covered by the tests around it.
#[cfg(not(miri))]
#[test]
fn drain_heap_forget_leaves_a_valid_placeholder() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    assert!(v.is_heap());

    let d = v.drain(2..5);
    std::mem::forget(d); // leaks the taken container (documented behaviour)

    // The caller's slot is the empty *inline* placeholder — valid and usable.
    assert!(v.is_empty());
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    v.push(9);
    assert_eq!(&*v, &[9]);
}

#[test]
fn drain_heap_prefix_and_suffix_ranges() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..8);
    assert!(v.is_heap());

    assert_eq!(v.drain(..3).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!(v.is_heap());
    assert_eq!(&*v, &[3, 4, 5, 6, 7]);

    // Suffix range: the shift-back is a no-op, the write-back still happens.
    assert_eq!(v.drain(3..5).collect::<Vec<_>>(), vec![6, 7]);
    assert!(v.is_heap());
    assert_eq!(&*v, &[3, 4, 5]);

    // Empty range on a heap vector: never takes the container.
    assert_eq!(v.drain(1..1).count(), 0);
    assert!(v.is_heap());
    assert_eq!(&*v, &[3, 4, 5]);

    assert_eq!(v.drain(..).collect::<Vec<_>>(), vec![3, 4, 5]);
    assert!(v.is_heap());
    assert!(v.is_empty());
}

// -------------------------------------------------------------------------- //
// `Drain::keep_rest`                                                         //
// -------------------------------------------------------------------------- //

#[test]
fn drain_keep_rest_keeps_unyielded() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4, 5, 6]);
    let mut d = v.drain(1..4);
    assert_eq!(d.next(), Some(2)); // yielded: the caller owns it now
    d.keep_rest(); // keeps 3 and 4, removes only the yielded 2
    assert_eq!(d.len(), 0); // nothing left to yield
    drop(d); // must be a no-op after `keep_rest`
    assert_eq!(&*v, &[1, 3, 4, 5, 6]);
    assert_eq!(v.len(), 5);
}

#[test]
fn drain_keep_rest_after_next_back() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4, 5, 6, 7]);
    let mut d = v.drain(1..6); // [2, 3, 4, 5, 6]
    assert_eq!(d.next(), Some(2));
    assert_eq!(d.next_back(), Some(6)); // unyielded: [3, 4, 5]
    d.keep_rest();
    drop(d);
    assert_eq!(&*v, &[1, 3, 4, 5, 7]);
}

#[test]
fn drain_keep_rest_heap() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3, 4, 5, 6]);
    assert!(v.is_heap());
    let cap = v.capacity();
    let mut d = v.drain(1..5);
    assert_eq!(d.next(), Some(2));
    d.keep_rest();
    drop(d);
    assert!(v.is_heap());
    assert_eq!(&*v, &[1, 3, 4, 5, 6]);
    assert_eq!(v.capacity(), cap); // capacity preserved
}

#[test]
fn drain_keep_rest_drop_accounting() {
    // Each element is destroyed exactly once: the yielded one by the caller,
    // the survivors by the container. `keep_rest` must neither drop the kept
    // elements nor double-drop the yielded ones.
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut v: CovariantVec<Tracked, 4> = (0..6).map(|i| Tracked::new(&ctr, i)).collect();
    let mut d = v.drain(1..4); // [1, 2, 3]
    let yielded = d.next().unwrap(); // caller-owned
    d.keep_rest(); // keeps 2 and 3 in the container
    drop(d);
    drop(yielded);
    assert_eq!(ctr.load(Ordering::SeqCst), 1); // only the yielded one so far
    assert_eq!(
        &v.iter().map(|t| t.val).collect::<Vec<_>>(),
        &[0, 2, 3, 4, 5]
    );
    drop(v);
    assert_eq!(ctr.load(Ordering::SeqCst), 6); // every element exactly once
}

#[test]
fn drain_keep_rest_zst() {
    let mut v = CovariantVec::<(), 4>::new();
    v.extend([(), (), (), (), ()]); // len 5
    let mut d = v.drain(1..4);
    d.next(); // one yielded (ZST: no observable drop)
    d.keep_rest(); // keeps two
    drop(d);
    assert_eq!(v.len(), 4); // prefix(1) + kept(2) + tail(1)
}

#[test]
fn drain_keep_rest_yield_everything() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4, 5]);
    let mut d = v.drain(1..4);
    assert_eq!(d.len(), 3);
    let _ = d.next();
    let _ = d.next();
    let _ = d.next();
    d.keep_rest(); // nothing unyielded: pure removal, like a finished drain
    drop(d);
    assert_eq!(&*v, &[1, 5]);
}

#[test]
fn drain_keep_rest_empty_range_is_noop() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let mut d = v.drain(1..1); // empty: container never taken
    d.keep_rest();
    drop(d);
    assert_eq!(&*v, &[1, 2, 3]);
}

#[test]
fn drain_keep_rest_after_next_back_only() {
    // Consume only from the back, then keep the rest: the front elements were
    // never yielded, so nothing but the yielded tail is dropped.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4]);
    let mut d = v.drain(1..4);
    assert_eq!(d.next_back(), Some(4));
    d.keep_rest();
    assert_eq!(d.len(), 0);
    drop(d);
    assert_eq!(&*v, &[1, 2, 3]);
}

#[test]
fn drain_keep_rest_nothing_yielded_is_a_full_keep() {
    // `keep_rest` before any yield: `pos == begin`, so the shift is skipped
    // and the rollback range `begin + kept == end` trips `DrainRollback`'s
    // empty-range skip too. Both skip paths are subtle; the observable
    // contract is that this is a pure no-op, not a partial removal.
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut v: CovariantVec<Tracked, 4> = (0..6).map(|i| Tracked::new(&ctr, i)).collect();
    let mut d = v.drain(1..4);
    d.keep_rest();
    assert_eq!(d.len(), 0);
    drop(d);
    assert_eq!(ctr.load(Ordering::SeqCst), 0); // nothing was dropped
    assert_eq!(
        &v.iter().map(|t| t.val).collect::<Vec<_>>(),
        &[0, 1, 2, 3, 4, 5]
    );
    drop(v);
    assert_eq!(ctr.load(Ordering::SeqCst), 6); // each element exactly once
}

#[test]
fn drain_keep_rest_nothing_yielded_heap_keeps_capacity() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3, 4, 5, 6]);
    assert!(v.is_heap());
    let cap = v.capacity();
    let mut d = v.drain(1..5);
    d.keep_rest();
    drop(d);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), cap); // the state-swap write-back still happened
    assert_eq!(&*v, &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn drain_partial_both_ends_inline() {
    // Partial consumption from both ends of an *inline* vector: the rollback
    // must shift the tail behind the kept prefix inside the union's own
    // storage, staying inline.
    let mut v = CovariantVec::<i32, 8>::new();
    v.extend([1, 2, 3, 4, 5, 6]);
    assert!(!v.is_heap());
    {
        let mut d = v.drain(1..5); // [2, 3, 4, 5]
        assert_eq!(d.next(), Some(2));
        assert_eq!(d.next_back(), Some(5));
        // [3, 4] are still unconsumed: `Drain::drop` drops them.
    }
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 8);
    assert_eq!(&*v, &[1, 6]);
}

/// Inline-state twin of the heap drain tests, with drop glue: the elements
/// live inside the `Drain` struct, so `next` derives a slot pointer into the
/// drain's own allocation and only then bumps `pos`. Sound — borrow stacks are
/// per-location, and the cursor is byte-disjoint from the container — but
/// inline-only, so this pins both the `next` read and `Drain::drop`'s
/// per-element drop loop on inline storage under Miri, with exact drop
/// accounting.
#[test]
fn drain_inline_non_zst_next_and_drop_loop_accounting() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..3 {
        v.push(Tracked::new(&ctr, i));
    }
    assert!(!v.is_heap());
    {
        let mut d = v.drain(1..3); // [1, 2]
        assert_eq!(d.next().map(|t| t.val), Some(1)); // caller-owned from here
                                                      // [2] is still unconsumed: `Drain::drop`'s loop drops it inline.
    }
    assert_eq!(ctr.load(Ordering::SeqCst), 2); // 1 (caller) + 2 (drop loop)
    assert!(!v.is_heap());
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].val, 0);
    drop(v);
    assert_eq!(ctr.load(Ordering::SeqCst), 3); // every element exactly once
}
