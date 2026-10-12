//! Runtime behaviour tests, including drop-accounting.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantVec;

// --------------------------------------------------------------------------
// A drop-accounting element type.
// --------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Tracked {
    counter: Arc<AtomicUsize>,
    val: i32,
}

impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}
impl Eq for Tracked {}

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

// --------------------------------------------------------------------------
// Basic inline behaviour
// --------------------------------------------------------------------------

#[test]
fn inline_push_pop() {
    let mut v = CovariantVec::<i32, 4>::new();
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert!(v.is_empty());

    v.push(1);
    v.push(2);
    v.push(3);
    assert_eq!(v.len(), 3);
    assert!(!v.is_heap());
    assert_eq!(&*v, &[1, 2, 3]);
    assert_eq!(v[0], 1);
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.pop(), Some(2));
    assert_eq!(v.pop(), Some(1));
    assert_eq!(v.pop(), None);
    assert!(v.is_empty());
}

#[test]
fn spill_at_boundary_and_grow() {
    let mut v = CovariantVec::<i32, 4>::new();
    for i in 0..4 {
        v.push(i);
    }
    assert!(!v.is_heap());
    v.push(4); // 5th element -> spill
    assert!(v.is_heap());
    assert!(v.capacity() >= 5);
    assert_eq!(&*v, &[0, 1, 2, 3, 4]);

    for i in 5..100 {
        v.push(i);
    }
    assert_eq!(v.len(), 100);
    assert!(v.is_heap());
    assert!(v.iter().copied().eq(0..100));
}

#[test]
fn reserve_spills_early() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.push(1);
    assert!(!v.is_heap());
    v.reserve(50);
    assert!(v.is_heap());
    assert!(v.capacity() >= 51);
    assert_eq!(&*v, &[1]);
}

// --------------------------------------------------------------------------
// pop never shrinks; explicit shrink_to_inline
// --------------------------------------------------------------------------

#[test]
fn pop_does_not_shrink() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3]);
    assert!(v.is_heap());
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.pop(), Some(2));
    assert_eq!(v.pop(), Some(1));
    assert_eq!(v.pop(), None);
    // Still on the heap even though it would fit inline.
    assert!(v.is_heap());

    v.push(9);
    v.shrink_to_inline();
    assert!(!v.is_heap());
    assert_eq!(&*v, &[9]);

    // shrink_to_inline is a no-op when it does not fit.
    let mut w = CovariantVec::<i32, 2>::new();
    w.extend([1, 2, 3, 4]);
    assert!(w.is_heap());
    w.shrink_to_inline();
    assert!(w.is_heap());
    assert_eq!(&*w, &[1, 2, 3, 4]);
}

// --------------------------------------------------------------------------
// insert / remove / swap_remove / truncate / clear
// --------------------------------------------------------------------------

#[test]
fn insert_remove_swap_remove() {
    let mut v = CovariantVec::<i32, 2>::new();
    for i in 0..6 {
        v.push(i);
    }
    assert!(v.is_heap());

    v.insert(0, 100);
    assert_eq!(&*v, &[100, 0, 1, 2, 3, 4, 5]);
    v.insert(7, 200);
    assert_eq!(&*v, &[100, 0, 1, 2, 3, 4, 5, 200]);

    assert_eq!(v.remove(0), 100);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 200]);
    assert_eq!(v.remove(6), 200);
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);

    assert_eq!(v.swap_remove(0), 0);
    assert_eq!(&*v, &[5, 1, 2, 3, 4]);
    assert_eq!(v.swap_remove(4), 4);
    assert_eq!(&*v, &[5, 1, 2, 3]);

    v.truncate(2);
    assert_eq!(&*v, &[5, 1]);
    v.clear();
    assert!(v.is_empty());
    assert!(v.is_heap());
}

#[test]
#[should_panic]
fn insert_out_of_bounds_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.push(1);
    v.insert(5, 2);
}

// --------------------------------------------------------------------------
// Drop accounting
// --------------------------------------------------------------------------

#[test]
fn drop_inline_elements() {
    let c = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 4>::new();
        for i in 0..3 {
            v.push(Tracked::new(&c, i));
        }
        assert!(!v.is_heap());
        assert_eq!(c.load(Ordering::SeqCst), 0);
    }
    assert_eq!(c.load(Ordering::SeqCst), 3);
}

#[test]
fn drop_heap_elements() {
    let c = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 4>::new();
        for i in 0..100 {
            v.push(Tracked::new(&c, i));
        }
        assert!(v.is_heap());
        assert_eq!(c.load(Ordering::SeqCst), 0);
    }
    assert_eq!(c.load(Ordering::SeqCst), 100);
}

#[test]
fn drop_after_truncate_and_pop() {
    let c = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<Tracked, 4>::new();
    for i in 0..10 {
        v.push(Tracked::new(&c, i));
    }
    assert!(v.is_heap());
    let popped = v.pop().unwrap();
    assert_eq!(popped.val, 9);
    drop(popped);
    assert_eq!(c.load(Ordering::SeqCst), 1);
    v.truncate(5);
    assert_eq!(c.load(Ordering::SeqCst), 5); // 9 (popped) + 8,7,6,5
    drop(v);
    assert_eq!(c.load(Ordering::SeqCst), 10);
}

#[test]
fn drop_partially_consumed_into_iter() {
    let c = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 4>::new();
        for i in 0..10 {
            v.push(Tracked::new(&c, i));
        }
        let mut it = v.into_iter();
        let a = it.next().unwrap();
        assert_eq!(a.val, 0);
        drop(a);
        assert_eq!(c.load(Ordering::SeqCst), 1);
        // Dropping the iterator drops the remaining 9.
    }
    assert_eq!(c.load(Ordering::SeqCst), 10);
}

/// Inline twin of `drop_partially_consumed_into_iter`: with `len <= N` the
/// elements live *inside* the iterator (`ManuallyDrop<CovariantVec>`), so
/// `next`/`next_back` derive a slot pointer into the iterator's own allocation
/// and only *then* write the cursor field. That is sound — borrow stacks are
/// per-location, and the cursor is byte-disjoint from the container, so the
/// write does not pop the pointer's tag — but the paths are inline-only, so
/// this test pins them under Miri (the heap-state tests never reach them).
#[test]
fn into_iter_inline_non_zst_both_ends_and_drop_remaining() {
    let c = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 4>::new();
        for i in 0..3 {
            v.push(Tracked::new(&c, i));
        }
        assert!(!v.is_heap());
        let mut it = v.into_iter();
        assert_eq!(it.as_slice().len(), 3);
        assert_eq!(it.next().map(|t| t.val), Some(0));
        assert_eq!(it.next_back().map(|t| t.val), Some(2));
        // The yielded pair was dropped by the caller.
        assert_eq!(c.load(Ordering::SeqCst), 2);
        // One element is left: dropping the iterator runs `drop_remaining`
        // over the inline storage.
    }
    assert_eq!(c.load(Ordering::SeqCst), 3);
}

// --------------------------------------------------------------------------
// ZST
// --------------------------------------------------------------------------

#[test]
fn zst_never_spills() {
    let mut v = CovariantVec::<(), 4>::new();
    for _ in 0..100_000 {
        v.push(());
    }
    assert_eq!(v.len(), 100_000);
    assert!(!v.is_heap());
    assert_eq!(v.pop(), Some(()));
    assert_eq!(v.len(), 99_999);
    v.clear();
    assert!(v.is_empty());
}

static ZDROP_COUNTS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn zst_with_drop_counts() {
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            ZDROP_COUNTS.fetch_add(1, Ordering::SeqCst);
        }
    }
    ZDROP_COUNTS.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<Z, 4>::new();
    for _ in 0..10 {
        v.push(Z);
    }
    assert_eq!(v.len(), 10);
    assert!(!v.is_heap());
    // A ZST owns no bytes: `push` `mem::forget`s the value and the count owns
    // the conceptual destructor, so nothing has been dropped yet.
    assert_eq!(ZDROP_COUNTS.load(Ordering::SeqCst), 0);
    drop(v);
    // Exactly one destructor per counted element.
    assert_eq!(ZDROP_COUNTS.load(Ordering::SeqCst), 10);
}

// --------------------------------------------------------------------------
// Clone
// --------------------------------------------------------------------------

#[test]
fn clone_inline_and_heap() {
    let c = Arc::new(AtomicUsize::new(0));

    let mut inline = CovariantVec::<Tracked, 4>::new();
    inline.push(Tracked::new(&c, 1));
    inline.push(Tracked::new(&c, 2));
    let inline2 = inline.clone();
    assert_eq!(inline, inline2);
    assert!(!inline2.is_heap());

    let mut heap = CovariantVec::<Tracked, 2>::new();
    for i in 0..6 {
        heap.push(Tracked::new(&c, i));
    }
    assert!(heap.is_heap());
    let heap2 = heap.clone();
    assert!(heap2.is_heap());
    assert_eq!(heap, heap2);

    drop(inline);
    drop(inline2);
    drop(heap);
    drop(heap2);
    // 2 (inline) + 2 (clone) + 6 (heap) + 6 (clone)
    assert_eq!(c.load(Ordering::SeqCst), 16);
}

#[test]
fn clone_normalizes_small_heap_to_inline() {
    // Spill to the heap, then shrink to a length that would fit inline.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    assert!(v.is_heap());
    v.truncate(2);
    assert!(v.is_heap()); // `truncate` never shrinks back to inline

    // The clone normalizes to the inline state: no allocation.
    let c = v.clone();
    assert!(!c.is_heap());
    assert_eq!(&*c, &[0, 1]);
    assert_eq!(c.capacity(), 4);
}

#[test]
fn clone_heap_allocates_exactly_len() {
    let mut v = CovariantVec::<i32, 2>::with_capacity(8);
    v.extend(0..6);
    assert!(v.is_heap());
    assert_eq!(v.capacity(), 8); // source has growth slack
    let c = v.clone();
    assert!(c.is_heap());
    assert_eq!(c.capacity(), 6); // clone allocates exactly `len`, not the source cap
    assert_eq!(&*c, &[0, 1, 2, 3, 4, 5]);
}

// --------------------------------------------------------------------------
// Vec conversions
// --------------------------------------------------------------------------

#[test]
fn from_vec_inline() {
    let cv = CovariantVec::<i32, 8>::from(vec![1, 2, 3]);
    assert!(!cv.is_heap());
    assert_eq!(&*cv, &[1, 2, 3]);
}

#[test]
fn from_vec_heap() {
    let mut v: Vec<i32> = Vec::with_capacity(100);
    v.extend([1, 2, 3]);
    assert!(v.capacity() > 4);
    let cv = CovariantVec::<i32, 4>::from(v);
    assert!(cv.is_heap());
    assert_eq!(&*cv, &[1, 2, 3]);
}

#[test]
fn into_vec_roundtrip() {
    let mut cv = CovariantVec::<i32, 2>::new();
    for i in 0..10 {
        cv.push(i);
    }
    let v: Vec<i32> = cv.into();
    assert_eq!(v, (0..10).collect::<Vec<_>>());
}

#[test]
fn from_vec_zst() {
    let cv = CovariantVec::<(), 4>::from(vec![(); 100]);
    assert_eq!(cv.len(), 100);
    assert!(!cv.is_heap());
}

#[test]
fn into_vec_zst() {
    let mut cv = CovariantVec::<(), 4>::new();
    for _ in 0..100 {
        cv.push(());
    }
    let v: Vec<()> = cv.into();
    assert_eq!(v.len(), 100);
}

#[test]
#[should_panic(expected = "capacity overflow")]
#[allow(clippy::uninit_vec)] // the whole point is a bogus ZST length
fn from_vec_zst_len_overflow_panics() {
    // A ZST `Vec` can have an arbitrarily large length; `From<Vec>` must
    // reject a length that would pollute `HEAP_TAG`.
    let mut v: Vec<()> = Vec::new();
    // The smallest length that exceeds the container's bound, computed
    // portably (`isize::MAX + 1` is `2^63` on 64-bit, `2^31` on 32-bit).
    let bogus_len = isize::MAX as usize + 1;
    // Safety: `()` is a ZST; this length is representable and the allocation is
    // empty, so `set_len` is sound.
    unsafe { v.set_len(bogus_len) };
    let _ = CovariantVec::<(), 4>::from(v);
}

#[test]
fn insert_triggers_realloc() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(0..4); // spills; capacity is now 4
    assert!(v.is_heap());
    let old_cap = v.capacity();
    while v.len() < old_cap {
        let n = v.len() as i32;
        v.push(n);
    }
    assert_eq!(v.len(), old_cap);
    v.insert(0, -1); // forces a reallocation
    assert!(v.capacity() > old_cap);
    assert_eq!(v.len(), old_cap + 1);
    assert_eq!(v[0], -1);
}

#[test]
fn shrink_then_regrow() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend(0..10);
    assert!(v.is_heap());
    v.truncate(2);
    v.shrink_to_inline();
    assert!(!v.is_heap());
    assert_eq!(&*v, &[0, 1]);

    v.extend(2..10); // re-spill
    assert!(v.is_heap());
    assert_eq!(v.len(), 10);
    assert!(v.iter().copied().eq(0..10));
}

#[test]
fn as_ptr_and_resize() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3]);
    assert_eq!(unsafe { *v.as_ptr() }, 1);
    unsafe { *v.as_mut_ptr() = 10 };
    assert_eq!(v[0], 10);

    v.resize(6, 7);
    assert_eq!(&*v, &[10, 2, 3, 7, 7, 7]);
    v.resize(2, 0);
    assert_eq!(&*v, &[10, 2]);
}

#[test]
fn with_capacity_goes_heap() {
    let v = CovariantVec::<i32, 4>::with_capacity(64);
    assert!(v.is_heap());
    assert!(v.capacity() >= 64);
    assert!(v.is_empty());

    let w = CovariantVec::<i32, 4>::with_capacity(2);
    assert!(!w.is_heap());
    assert_eq!(w.capacity(), 4);
}

#[test]
fn set_len_shrink() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    // Safety: shrinking over `Copy` elements (no destructors to run).
    unsafe { v.set_len(1) };
    assert_eq!(v.len(), 1);
    assert_eq!(v[0], 1);
    assert_eq!(v.pop(), Some(1));
    assert_eq!(v.pop(), None);
}

// --------------------------------------------------------------------------
// Iterators
// --------------------------------------------------------------------------

#[test]
fn iterators() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3, 4, 5]);

    assert!(v.iter().copied().eq([1, 2, 3, 4, 5]));
    for x in v.iter_mut() {
        *x *= 10;
    }
    assert_eq!(&*v, &[10, 20, 30, 40, 50]);

    // By-reference IntoIterator
    let mut sum = 0;
    for x in &v {
        sum += *x;
    }
    assert_eq!(sum, 150);

    // Consuming iterator, both ends
    let mut it = v.into_iter();
    assert_eq!(it.next(), Some(10));
    assert_eq!(it.next_back(), Some(50));
    assert_eq!(it.len(), 3);
    let rest: Vec<i32> = it.collect();
    assert_eq!(rest, vec![20, 30, 40]);
}

#[test]
fn into_iter_as_slice() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([1, 2, 3, 4, 5]);
    let mut it = v.into_iter();
    assert_eq!(it.as_slice(), &[1, 2, 3, 4, 5]);
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.as_slice(), &[2, 3, 4, 5]);
    assert_eq!(it.next_back(), Some(5));
    assert_eq!(it.as_slice(), &[2, 3, 4]);
}

// --------------------------------------------------------------------------
// Trait forwarding
// --------------------------------------------------------------------------

#[test]
fn trait_forwarding() {
    let a = CovariantVec::<i32, 4>::from_iter([1, 2, 3]);
    let b: CovariantVec<i32, 4> = [1, 2, 3].into_iter().collect();
    assert_eq!(a, b);
    assert_eq!(format!("{:?}", a), "[1, 2, 3]");
    assert!(a < CovariantVec::<i32, 4>::from_iter([1, 2, 4]));

    let mut h1 = DefaultHasher::new();
    a.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    CovariantVec::<i32, 4>::from_iter([1, 2, 3]).hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());

    let mut e = CovariantVec::<i32, 4>::new();
    e.extend([4, 5]);
    assert_eq!(&*e, &[4, 5]);
}

// --------------------------------------------------------------------------
// N = 0
// --------------------------------------------------------------------------

#[test]
fn n_zero_behaves_like_vec() {
    let mut v = CovariantVec::<i32, 0>::new();
    assert_eq!(v.capacity(), 0);
    v.push(1);
    assert!(v.is_heap());
    assert!(v.capacity() >= 1);
    v.push(2);
    assert_eq!(&*v, &[1, 2]);
}

// --------------------------------------------------------------------------
// Forwarding impls: PartialEq<Vec>, Extend<&T>/FromIterator<&T> for T: Copy
// --------------------------------------------------------------------------

#[test]
fn partial_eq_vec_and_copy_extend() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    assert_eq!(v, vec![1, 2, 3]);
    assert!(v != vec![1, 2, 4]);

    // `Extend<&T>` for `T: Copy`.
    let src = [10, 20, 30];
    v.extend(src.iter());
    assert_eq!(&*v, &[1, 2, 3, 10, 20, 30]);

    // `FromIterator<&T>` for `T: Copy`.
    let c: CovariantVec<i32, 4> = src.iter().collect();
    assert_eq!(&*c, &[10, 20, 30]);
}

// --------------------------------------------------------------------------
// Move-out conversions: drop accounting and inline-full insert
// --------------------------------------------------------------------------

#[test]
fn into_vec_drop_accounting() {
    let counter = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 2>::new();
        for i in 0..6 {
            v.push(Tracked::new(&counter, i));
        }
        assert!(v.is_heap());
        let vec: Vec<Tracked> = v.into();
        // The move-out must not run any destructor.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(vec); // exactly six drops
    }
    assert_eq!(counter.load(Ordering::SeqCst), 6);

    // Inline state too.
    let counter = Arc::new(AtomicUsize::new(0));
    {
        let mut v = CovariantVec::<Tracked, 4>::new();
        for i in 0..3 {
            v.push(Tracked::new(&counter, i));
        }
        assert!(!v.is_heap());
        let vec: Vec<Tracked> = v.into();
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(vec);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[test]
fn from_vec_drop_accounting() {
    let counter = Arc::new(AtomicUsize::new(0));
    {
        // Heap path (capacity > N): the block is taken over, not copied.
        let mut src: Vec<Tracked> = Vec::with_capacity(10);
        for i in 0..6 {
            src.push(Tracked::new(&counter, i));
        }
        let v = CovariantVec::<Tracked, 2>::from(src);
        assert!(v.is_heap());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(v);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 6);

    // Inline path (len <= N): elements are moved in, the `Vec` buffer freed
    // without dropping them.
    let counter = Arc::new(AtomicUsize::new(0));
    {
        let src: Vec<Tracked> = (0..3).map(|i| Tracked::new(&counter, i)).collect();
        let v = CovariantVec::<Tracked, 4>::from(src);
        assert!(!v.is_heap());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(v);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[test]
fn insert_into_full_inline_spills() {
    // `insert` on a full inline vector must spill to the heap rather than
    // overwrite, and preserve order.
    let mut v = CovariantVec::<i32, 3>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    assert_eq!(v.len(), 3);
    v.insert(1, 99); // len == N: triggers spill
    assert!(v.is_heap());
    assert_eq!(&*v, &[1, 99, 2, 3]);
    // Front and back inserts still work after the spill.
    v.insert(0, 0);
    v.insert(v.len(), 4);
    assert_eq!(&*v, &[0, 1, 99, 2, 3, 4]);
}

#[test]
fn insert_at_len_into_full_inline_appends() {
    // The append corner (`index == len`) on a *full* inline vector: `reserve(1)`
    // spills first, then the insert must write past the last slot without a
    // shift (`index < len` is false) — the boundary the mid-insert test above
    // does not hit.
    let mut v = CovariantVec::<i32, 3>::new();
    v.extend([1, 2, 3]);
    assert!(!v.is_heap());
    v.insert(3, 4); // append at len == N
    assert!(v.is_heap());
    assert_eq!(&*v, &[1, 2, 3, 4]);

    // Same corner on a vector that is still inline with room to spare: no
    // spill, no shift.
    let mut w = CovariantVec::<i32, 4>::new();
    w.extend([1, 2]);
    w.insert(2, 3);
    assert!(!w.is_heap());
    assert_eq!(&*w, &[1, 2, 3]);
}

/// An iterator that lies about its lower-bound `size_hint`, reporting `0`.
struct LyingIter {
    items: std::vec::IntoIter<i32>,
}

impl Iterator for LyingIter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        self.items.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Deliberately understate the lower bound.
        (0, self.items.size_hint().1)
    }
}

#[test]
fn extend_with_lying_size_hint() {
    // `extend` trusts `size_hint().0` only as a *reservation*; an understated
    // bound must not lose elements or corrupt state.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend(LyingIter {
        items: vec![1, 2, 3, 4, 5, 6, 7, 8].into_iter(),
    });
    assert_eq!(&*v, &[1, 2, 3, 4, 5, 6, 7, 8]);

    // Same for the `T: Copy` by-reference variant.
    v.extend(LyingIter {
        items: vec![9, 10, 11, 12].into_iter(),
    });
    assert_eq!(&*v, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

#[test]
fn extend_heap_fast_path_with_lying_size_hint_falls_through() {
    // A heap destination with *spare* capacity and an understated hint: the
    // heap fast path fills the spare slots, then the remainder falls through
    // to the general loop (which pushes, growing the block). Pins the
    // single-entry fast loop's fall-through, not just the push-at-entry case
    // covered above.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([0, 1, 2]); // spills: len 3, cap 4
    assert!(v.is_heap());
    assert_eq!(v.capacity(), 4);
    v.extend(LyingIter {
        items: vec![3, 4, 5, 6].into_iter(), // lo == 0
    });
    assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6]);
    assert!(v.capacity() >= 7);
}
