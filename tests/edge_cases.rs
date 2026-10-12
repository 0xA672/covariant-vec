//! Edge cases: overflow/panic paths, `From<Vec>` boundaries, ZST destructor
//! accounting, clone panic safety, and the "retain never spills" invariant.

use std::ops::Bound;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantVec;

// -------------------------------------------------------------------------- //
// reserve / with_capacity overflow                                         //
// -------------------------------------------------------------------------- //

#[test]
#[should_panic(expected = "capacity overflow")]
fn reserve_usize_max_panics() {
    let mut v = CovariantVec::<u8, 4>::new();
    v.reserve(usize::MAX);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn reserve_overflow_on_nonempty_panics() {
    // `len + additional` overflows `usize`, so the checked add fires first.
    let mut v = CovariantVec::<u8, 4>::new();
    v.push(1);
    v.reserve(usize::MAX);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn with_capacity_usize_max_panics() {
    let _ = CovariantVec::<u8, 4>::with_capacity(usize::MAX);
}

// -------------------------------------------------------------------------- //
// try_reserve / try_reserve_exact                                          //
// -------------------------------------------------------------------------- //

#[test]
fn try_reserve_grows_and_preserves() {
    let mut v = CovariantVec::<u32, 4>::new();
    v.extend([1, 2, 3]);
    assert!(v.try_reserve(50).is_ok());
    assert!(v.capacity() >= 3 + 50);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
    // Growing an already-heap vector (realloc path) also works.
    assert!(v.try_reserve(1000).is_ok());
    assert!(v.capacity() >= 3 + 1000);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn try_reserve_exact_is_exact() {
    let mut v = CovariantVec::<u32, 4>::new();
    assert!(v.try_reserve_exact(50).is_ok());
    assert_eq!(v.capacity(), 50);
    // A request already covered by the capacity is a no-op.
    assert!(v.try_reserve_exact(0).is_ok());
    assert_eq!(v.capacity(), 50);
}

#[test]
fn reserve_exact_is_exact() {
    let mut v = CovariantVec::<u32, 4>::new();
    v.reserve_exact(50);
    assert_eq!(v.capacity(), 50);
    v.reserve_exact(0); // no-op
    assert_eq!(v.capacity(), 50);
}

#[test]
fn try_reserve_capacity_overflow_is_err() {
    use covariant_vec::TryReserveErrorKind;
    let mut v = CovariantVec::<u8, 4>::new();
    assert_eq!(
        v.try_reserve(usize::MAX).unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
    // `len + additional` overflowing `usize` is also a capacity overflow.
    v.push(1);
    assert_eq!(
        v.try_reserve(usize::MAX).unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
    // The failed reservation left the vector untouched.
    assert_eq!(v.as_slice(), &[1]);
}

#[test]
fn try_reserve_zst_bounds() {
    use covariant_vec::TryReserveErrorKind;
    let mut v = CovariantVec::<(), 4>::new();
    assert!(v.try_reserve(isize::MAX as usize).is_ok());
    assert_eq!(
        v.try_reserve(isize::MAX as usize + 1).unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
}

#[test]
fn try_reserve_error_display() {
    // The error type renders something meaningful for both kinds.
    let mut v = CovariantVec::<u8, 4>::new();
    let e = v.try_reserve(usize::MAX).unwrap_err();
    assert_eq!(e.to_string(), "capacity overflow");
}

// -------------------------------------------------------------------------- //
// ZST with_capacity bound + first-spill floor                              //
// -------------------------------------------------------------------------- //

#[test]
fn zst_with_capacity_at_isize_max_ok() {
    // A representable ZST capacity is accepted (and never allocates).
    let v = CovariantVec::<(), 4>::with_capacity(isize::MAX as usize);
    assert_eq!(v.capacity(), isize::MAX as usize);
    assert!(v.is_empty());
    assert!(!v.is_heap());
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn zst_with_capacity_usize_max_panics() {
    // Symmetric with `reserve`: a ZST request above `isize::MAX` is rejected
    // rather than silently accepted (the length field cannot hold it).
    let _ = CovariantVec::<(), 4>::with_capacity(usize::MAX);
}

#[test]
fn n_zero_first_spill_is_floored() {
    // `N = 0` must not crawl 1 -> 2 -> 4 -> 8: the first heap block is floored
    // to std's `min_non_zero_cap` (8 for `u8`, 4 for a 4-byte element).
    let mut a = CovariantVec::<u8, 0>::new();
    a.push(1);
    assert!(a.is_heap());
    assert_eq!(a.capacity(), 8);

    let mut b = CovariantVec::<u32, 0>::new();
    b.push(1);
    assert!(b.is_heap());
    assert_eq!(b.capacity(), 4);
}

// -------------------------------------------------------------------------- //
// drain(..=usize::MAX)                                                     //
// -------------------------------------------------------------------------- //

#[test]
#[should_panic]
fn drain_inclusive_usize_max_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    // `end = usize::MAX.saturating_add(1) == usize::MAX > len` -> range panic.
    let _ = v.drain(..=usize::MAX);
}

#[test]
fn drain_excluded_start_usize_max_panics() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4]);
    // `Bound::Excluded(usize::MAX)` as the start saturates: `begin` stays at
    // `usize::MAX` and the start-range assertion (`begin <= len`) fires *before*
    // the container is touched.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = v.drain((Bound::Excluded(usize::MAX), Bound::Excluded(1)));
    }));
    assert!(r.is_err());
    // Nothing was taken: the panic happened during range validation.
    assert_eq!(&*v, &[1, 2, 3, 4]);
}

#[test]
fn drain_excluded_start_usize_max_unbounded_end_panics() {
    // Same saturating-start corner with an `Unbounded` end, so the failing
    // start-range comparison is against `len` rather than a bounded end.
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = v.drain((Bound::Excluded(usize::MAX), Bound::Unbounded));
    }));
    assert!(r.is_err());
    assert_eq!(&*v, &[1, 2, 3, 4]);
}

// -------------------------------------------------------------------------- //
// From<Vec> boundaries                                                     //
// -------------------------------------------------------------------------- //

#[test]
fn from_vec_len_eq_n_cap_le_n_stays_inline() {
    let v = vec![1, 2, 3, 4];
    assert!(v.capacity() <= 4, "precondition: cap <= N");
    let cv = CovariantVec::<i32, 4>::from(v);
    assert!(!cv.is_heap());
    assert_eq!(&*cv, &[1, 2, 3, 4]);
}

#[test]
fn from_vec_cap_gt_n_len_zero_is_empty_heap() {
    let v: Vec<i32> = Vec::with_capacity(5);
    assert!(v.capacity() > 4);
    let cv = CovariantVec::<i32, 4>::from(v);
    assert!(cv.is_heap());
    assert!(cv.is_empty());
    assert!(cv.capacity() >= 5);
}

#[test]
fn from_vec_cap_n_plus_one_len_n_takes_over_with_slack() {
    let mut v: Vec<i32> = Vec::with_capacity(5);
    v.extend([1, 2, 3, 4]);
    assert!(v.capacity() > 4 && v.len() == 4);
    let cv = CovariantVec::<i32, 4>::from(v);
    assert!(cv.is_heap());
    assert_eq!(&*cv, &[1, 2, 3, 4]);
    assert!(cv.capacity() > 4); // slack preserved by the takeover
}

#[test]
fn from_nonempty_vec_n_zero_takes_over_heap() {
    // With `N = 0` the inline buffer holds nothing, so any non-empty `Vec` must
    // be taken over as heap storage.
    let v = vec![1, 2, 3];
    assert!(v.capacity() > 0);
    let cv = CovariantVec::<i32, 0>::from(v);
    assert!(cv.is_heap());
    assert_eq!(&*cv, &[1, 2, 3]);
    assert!(cv.capacity() >= 3);
}

#[test]
fn from_empty_vec_n_zero_stays_inline() {
    let v: Vec<i32> = Vec::new();
    assert_eq!(v.capacity(), 0);
    let cv = CovariantVec::<i32, 0>::from(v);
    assert!(!cv.is_heap());
    assert!(cv.is_empty());
    assert_eq!(cv.capacity(), 0);
}

// -------------------------------------------------------------------------- //
// ZST-with-Drop destructor accounting                                      //
// -------------------------------------------------------------------------- //

// Separate statics so the two tests can run in parallel without interfering.
static ZDROP_RETAIN: AtomicUsize = AtomicUsize::new(0);

struct ZDropR;
impl Drop for ZDropR {
    fn drop(&mut self) {
        ZDROP_RETAIN.fetch_add(1, Ordering::SeqCst);
    }
}

static ZDROP_DRAIN: AtomicUsize = AtomicUsize::new(0);

struct ZDropD;
impl Drop for ZDropD {
    fn drop(&mut self) {
        ZDROP_DRAIN.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn zst_drop_accounting_through_retain() {
    let mut v = CovariantVec::<ZDropR, 4>::new();
    for _ in 0..10 {
        v.push(ZDropR);
    }
    assert!(!v.is_heap());
    let mut seen = 0;
    v.retain(|_| {
        seen += 1;
        seen % 2 == 0
    });
    assert_eq!(v.len(), 5);
    assert_eq!(ZDROP_RETAIN.load(Ordering::SeqCst), 5); // odd ones dropped
    drop(v);
    assert_eq!(ZDROP_RETAIN.load(Ordering::SeqCst), 10); // exactly one per element
}

#[test]
fn zst_drop_accounting_through_drain() {
    let mut v = CovariantVec::<ZDropD, 4>::new();
    for _ in 0..10 {
        v.push(ZDropD);
    }
    assert_eq!(v.drain(2..6).count(), 4);
    assert_eq!(ZDROP_DRAIN.load(Ordering::SeqCst), 4);
    assert_eq!(v.len(), 6);
    drop(v);
    assert_eq!(ZDROP_DRAIN.load(Ordering::SeqCst), 10);
}

static ZDROP_INTOITER: AtomicUsize = AtomicUsize::new(0);

struct ZDropI;
impl Drop for ZDropI {
    fn drop(&mut self) {
        ZDROP_INTOITER.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn zst_drop_accounting_through_into_iter() {
    // `IntoIter` is ZST-protocol site 3: it `ptr::read`s values out of slots
    // that were never written, and its `Drop` runs `drop_in_place` over the
    // remaining unwritten slots. Pin the one-destructor-per-count balance.
    let mut v = CovariantVec::<ZDropI, 4>::new();
    for _ in 0..10 {
        v.push(ZDropI);
    }
    assert!(!v.is_heap());
    let mut it = v.into_iter();
    assert_eq!(it.len(), 10);
    for _ in 0..3 {
        assert!(it.next().is_some()); // each yielded value is dropped here
    }
    drop(it); // drops the 7 remaining
    assert_eq!(ZDROP_INTOITER.load(Ordering::SeqCst), 10);
}

static ZDROP_TRUNC: AtomicUsize = AtomicUsize::new(0);

struct ZDropT;
impl Drop for ZDropT {
    fn drop(&mut self) {
        ZDROP_TRUNC.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn zst_truncate_and_clear_accounting() {
    let mut v = CovariantVec::<ZDropT, 4>::new();
    for _ in 0..10 {
        v.push(ZDropT);
    }
    v.truncate(4);
    assert_eq!(v.len(), 4);
    assert_eq!(ZDROP_TRUNC.load(Ordering::SeqCst), 6); // 6 truncated away
    v.clear();
    assert_eq!(v.len(), 0);
    // Exactly one destructor per counted element.
    assert_eq!(ZDROP_TRUNC.load(Ordering::SeqCst), 10);

    // Clearing an empty ZST vector is a no-op.
    v.clear();
    assert_eq!(ZDROP_TRUNC.load(Ordering::SeqCst), 10);
    drop(v);
    assert_eq!(ZDROP_TRUNC.load(Ordering::SeqCst), 10);
}

static ZDROP_MUT: AtomicUsize = AtomicUsize::new(0);

struct ZDropM;
impl Drop for ZDropM {
    fn drop(&mut self) {
        ZDROP_MUT.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn zst_insert_remove_swap_remove_accounting() {
    let mut v = CovariantVec::<ZDropM, 4>::new();
    for _ in 0..5 {
        v.push(ZDropM);
    }
    // `insert` moves the value into the count (ZST protocol site 1): no
    // destructor runs, length grows.
    v.insert(0, ZDropM);
    assert_eq!(v.len(), 6);
    assert_eq!(ZDROP_MUT.load(Ordering::SeqCst), 0);
    // `remove` reads a value out of an unwritten slot (site 3) and drops it.
    drop(v.remove(2));
    assert_eq!(v.len(), 5);
    assert_eq!(ZDROP_MUT.load(Ordering::SeqCst), 1);
    // `swap_remove` likewise.
    drop(v.swap_remove(0));
    assert_eq!(v.len(), 4);
    assert_eq!(ZDROP_MUT.load(Ordering::SeqCst), 2);
    drop(v);
    // 5 pushes + 1 insert = 6 conceptual elements; 2 already dropped + 4 here.
    assert_eq!(ZDROP_MUT.load(Ordering::SeqCst), 6);
}

static ZCLONE_CTR: AtomicUsize = AtomicUsize::new(0);
static ZDROP_FE: AtomicUsize = AtomicUsize::new(0);

struct ZSide;
impl Clone for ZSide {
    fn clone(&self) -> Self {
        ZCLONE_CTR.fetch_add(1, Ordering::SeqCst);
        ZSide
    }
}
impl Drop for ZSide {
    fn drop(&mut self) {
        ZDROP_FE.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn from_elem_zst_side_effects() {
    // `from_elem` moves the argument into slot 0 and clones *that* for the
    // rest, so a side-effectful `Clone` runs exactly `n - 1` times and there are
    // exactly `n` destructors (`vec![elem; n]` semantics).
    ZCLONE_CTR.store(0, Ordering::SeqCst);
    ZDROP_FE.store(0, Ordering::SeqCst);
    let v = CovariantVec::<ZSide, 4>::from_elem(ZSide, 5);
    assert_eq!(v.len(), 5);
    assert_eq!(ZCLONE_CTR.load(Ordering::SeqCst), 4);
    assert_eq!(ZDROP_FE.load(Ordering::SeqCst), 0);
    drop(v);
    assert_eq!(ZDROP_FE.load(Ordering::SeqCst), 5);
}

// -------------------------------------------------------------------------- //
// resize: `Vec::resize` parity (clone/drop accounting)                        //
// -------------------------------------------------------------------------- //

/// Element whose `Clone` and `Drop` are counted, and whose clones are
/// distinguishable from the original (a clone's `val` is shifted by 1000), so a
/// test can check *which* slot received the moved original.
struct ResizeTracked {
    val: i32,
    clones: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl ResizeTracked {
    fn new(val: i32, clones: &Arc<AtomicUsize>, drops: &Arc<AtomicUsize>) -> Self {
        ResizeTracked {
            val,
            clones: Arc::clone(clones),
            drops: Arc::clone(drops),
        }
    }
}

impl Clone for ResizeTracked {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        ResizeTracked {
            val: self.val + 1000,
            clones: Arc::clone(&self.clones),
            drops: Arc::clone(&self.drops),
        }
    }
}

impl Drop for ResizeTracked {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn resize_grow_inline_moves_original_into_last_slot() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut v =
        CovariantVec::<ResizeTracked, 4>::from_elem(ResizeTracked::new(0, &clones, &drops), 3);
    assert!(!v.is_heap());
    // 3 -> 6: two clones (plus the two `from_elem` clones), and the original
    // argument lands in slot 5.
    v.resize(6, ResizeTracked::new(7, &clones, &drops));
    assert_eq!(v.len(), 6);
    assert_eq!(clones.load(Ordering::SeqCst), 2 + 2); // 2 from `from_elem`
    assert_eq!(drops.load(Ordering::SeqCst), 0); // the moved original is not dropped
                                                 // Slots 0..=2 are the `from_elem` elements (slot 0 is the original, the rest
                                                 // are clones, so `val` is shifted); 3..=4 are the resize clones (`val`
                                                 // shifted); slot 5 is the moved original (unshifted).
    assert_eq!(v[0].val, 0);
    for i in 1..3 {
        assert_eq!(v[i].val, 1000);
    }
    for i in 3..5 {
        assert_eq!(v[i].val, 1007);
    }
    assert_eq!(v[5].val, 7);
    drop(v);
    assert_eq!(drops.load(Ordering::SeqCst), 6); // exactly `new_len` destructors
}

#[test]
fn resize_grow_heap_moves_original_into_last_slot() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut v =
        CovariantVec::<ResizeTracked, 2>::from_elem(ResizeTracked::new(0, &clones, &drops), 5);
    assert!(v.is_heap());
    v.resize(9, ResizeTracked::new(7, &clones, &drops));
    assert_eq!(v.len(), 9);
    assert_eq!(clones.load(Ordering::SeqCst), 4 + 3);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    for i in 5..8 {
        assert_eq!(v[i].val, 1007);
    }
    assert_eq!(v[8].val, 7);
    drop(v);
    assert_eq!(drops.load(Ordering::SeqCst), 9);
}

#[test]
fn resize_grow_zst_clone_drop_counts() {
    // `vec![elem; n]`-style accounting for a side-effectful ZST: growing 5 -> 9
    // clones 3 more times (4 came from `from_elem`), never drops an extra
    // original, and ends with exactly `new_len` destructors.
    ZCLONE_CTR.store(0, Ordering::SeqCst);
    ZDROP_FE.store(0, Ordering::SeqCst);
    let mut v = CovariantVec::<ZSide, 4>::from_elem(ZSide, 5);
    v.resize(9, ZSide);
    assert_eq!(v.len(), 9);
    assert_eq!(ZCLONE_CTR.load(Ordering::SeqCst), 4 + 3);
    assert_eq!(ZDROP_FE.load(Ordering::SeqCst), 0);
    drop(v);
    assert_eq!(ZDROP_FE.load(Ordering::SeqCst), 9);
}

// -------------------------------------------------------------------------- //
// Clone panic safety                                                       //
// -------------------------------------------------------------------------- //

struct CloneBomb {
    val: i32,
    clone_ctr: Arc<AtomicUsize>,
    drop_ctr: Arc<AtomicUsize>,
    panic_at: usize,
}

impl Clone for CloneBomb {
    fn clone(&self) -> Self {
        let n = self.clone_ctr.fetch_add(1, Ordering::SeqCst);
        assert!(n != self.panic_at, "intentional clone panic");
        CloneBomb {
            val: self.val,
            clone_ctr: Arc::clone(&self.clone_ctr),
            drop_ctr: Arc::clone(&self.drop_ctr),
            panic_at: self.panic_at,
        }
    }
}

impl Drop for CloneBomb {
    fn drop(&mut self) {
        self.drop_ctr.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn clone_panic_drops_partial_and_leaves_source_intact() {
    let clone_ctr = Arc::new(AtomicUsize::new(0));
    let drop_ctr = Arc::new(AtomicUsize::new(0));

    let mut src = CovariantVec::<CloneBomb, 2>::new();
    for i in 0..6 {
        src.push(CloneBomb {
            val: i,
            clone_ctr: Arc::clone(&clone_ctr),
            drop_ctr: Arc::clone(&drop_ctr),
            panic_at: 3,
        });
    }
    assert!(src.is_heap());

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = src.clone();
    }));
    assert!(r.is_err());
    // Three clones succeeded before the panic; the partially built clone was
    // dropped exactly once, and the source is untouched.
    assert_eq!(clone_ctr.load(Ordering::SeqCst), 4); // 0,1,2 ok; 3 panicked
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 3);
    assert_eq!(src.len(), 6);
    drop(src);
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 9);
}

#[test]
fn clone_panic_inline_drops_partial_and_leaves_source_intact() {
    // Same guarantee as the heap branch, but through the inline direct-write
    // path (which advances `out.len` per element rather than going via `push`).
    let clone_ctr = Arc::new(AtomicUsize::new(0));
    let drop_ctr = Arc::new(AtomicUsize::new(0));

    let mut src = CovariantVec::<CloneBomb, 8>::new();
    for i in 0..5 {
        src.push(CloneBomb {
            val: i,
            clone_ctr: Arc::clone(&clone_ctr),
            drop_ctr: Arc::clone(&drop_ctr),
            panic_at: 2,
        });
    }
    assert!(!src.is_heap());

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = src.clone();
    }));
    assert!(r.is_err());
    assert_eq!(clone_ctr.load(Ordering::SeqCst), 3); // 0,1 ok; 2 panicked
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 2); // partial clone dropped once
    assert_eq!(src.len(), 5);
    drop(src);
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 7);
}

// -------------------------------------------------------------------------- //
// retain at len == N keeping everything must not spill                     //
// -------------------------------------------------------------------------- //

#[test]
fn retain_full_inline_does_not_spill() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3, 4]);
    assert!(!v.is_heap());
    v.retain(|_| true);
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), 4);
    assert_eq!(&*v, &[1, 2, 3, 4]);

    // Same for `retain_mut`.
    v.retain_mut(|x| {
        *x += 1;
        true
    });
    assert!(!v.is_heap());
    assert_eq!(&*v, &[2, 3, 4, 5]);
}

// -------------------------------------------------------------------------- //
// Clone of a long ZST vector (regression: must not take the heap branch)    //
// -------------------------------------------------------------------------- //

#[test]
fn clone_long_zst_stays_inline() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Z;
    impl Clone for Z {
        fn clone(&self) -> Z {
            Z
        }
    }
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    let mut v = CovariantVec::<Z, 4>::new();
    for _ in 0..10 {
        v.push(Z); // len > N, still inline (ZSTs never spill)
    }
    assert!(!v.is_heap());
    assert_eq!(v.len(), 10);

    let c = v.clone();
    assert!(!c.is_heap(), "cloning a ZST vector must stay inline");
    assert_eq!(c.len(), 10);

    drop(c);
    drop(v);
    assert_eq!(DROPS.load(Ordering::SeqCst), 20); // exactly one drop per element
}

// -------------------------------------------------------------------------- //
// ZST reserve boundary                                                        //
// -------------------------------------------------------------------------- //

#[test]
fn zst_reserve_at_isize_max_ok() {
    let mut v = CovariantVec::<(), 4>::new();
    v.reserve(isize::MAX as usize); // exactly the representable bound
    assert!(!v.is_heap());
    assert_eq!(v.capacity(), isize::MAX as usize);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn zst_reserve_isize_max_plus_one_panics() {
    let mut v = CovariantVec::<(), 4>::new();
    v.reserve(isize::MAX as usize + 1);
}

// -------------------------------------------------------------------------- //
// spare_capacity_mut                                                          //
// -------------------------------------------------------------------------- //

#[test]
fn spare_capacity_mut_workflow() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.reserve(8);
    assert!(v.is_heap());
    let cap = v.capacity();

    {
        let spare = v.spare_capacity_mut();
        assert_eq!(spare.len(), cap);
        spare[0].write(1);
        spare[1].write(2);
        spare[2].write(3);
    }
    unsafe { v.set_len(3) };
    assert_eq!(&*v, &[1, 2, 3]);
    assert_eq!(v.spare_capacity_mut().len(), cap - 3);

    // ZST: a huge (but zero-sized) spare slice, still sound.
    let mut z = CovariantVec::<(), 4>::new();
    assert_eq!(z.spare_capacity_mut().len(), isize::MAX as usize);
}

// -------------------------------------------------------------------------- //
// from_elem / resize_with                                                     //
// -------------------------------------------------------------------------- //

#[test]
fn from_elem_and_resize_with() {
    let v = CovariantVec::<i32, 4>::from_elem(7, 3);
    assert!(!v.is_heap());
    assert_eq!(&*v, &[7, 7, 7]);

    let big = CovariantVec::<i32, 2>::from_elem(9, 5); // spills
    assert!(big.is_heap());
    assert_eq!(&*big, &[9, 9, 9, 9, 9]);

    assert_eq!(CovariantVec::<i32, 4>::from_elem(1, 0).len(), 0);

    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let mut k = 0;
    v.resize_with(6, || {
        k += 1;
        k
    });
    assert_eq!(&*v, &[1, 2, 3, 1, 2, 3]);
    v.resize_with(1, || unreachable!()); // shrink never calls the closure
    assert_eq!(&*v, &[1]);
}

/// Element whose `clone` panics on the `panic_on`-th call, counting drops.
struct BombClone {
    drops: Arc<AtomicUsize>,
    clones: Arc<AtomicUsize>,
    panic_on: usize,
}

impl Clone for BombClone {
    fn clone(&self) -> Self {
        let n = self.clones.fetch_add(1, Ordering::SeqCst);
        assert!(n != self.panic_on, "intentional clone panic");
        BombClone {
            drops: Arc::clone(&self.drops),
            clones: Arc::clone(&self.clones),
            panic_on: self.panic_on,
        }
    }
}

impl Drop for BombClone {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn from_elem_panicking_clone_drops_are_balanced() {
    let drops = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let elem = BombClone {
            drops: Arc::clone(&drops),
            clones: Arc::clone(&clones),
            panic_on: 2,
        };
        let _v = CovariantVec::<BombClone, 4>::from_elem(elem, 10);
    }));
    assert!(r.is_err());
    // Slot 0 holds the moved `elem`; two clones were built before the third
    // panicked, so three values exist and must each drop exactly once.
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn resize_with_panicking_closure_leaves_consistent() {
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut k = 0;
        v.resize_with(10, || {
            k += 1;
            if k == 3 {
                panic!("intentional resize_with panic");
            }
            k
        });
    }));
    assert!(r.is_err());
    // Two new elements were produced before the panic; the length was advanced
    // after each write, so the vector is valid and shorter than requested.
    assert_eq!(v.len(), 5);
    assert_eq!(&*v, &[1, 2, 3, 1, 2]);
}

#[test]
fn resize_with_panicking_closure_drop_accounting() {
    // The test above pins the *length*; `i32` cannot observe a leak or a
    // double drop, so this pins the accounting on an inline container: exactly
    // the elements produced so far are live, and `Drop` cleans up each once.
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<ResizeTracked, 4>::new();
    for i in 0..3 {
        v.push(ResizeTracked::new(i, &clones, &drops));
    }
    assert!(!v.is_heap());
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut k = 0;
        v.resize_with(6, || {
            k += 1;
            if k == 2 {
                panic!("intentional resize_with panic");
            }
            ResizeTracked::new(100 + k, &clones, &drops)
        });
    }));
    assert!(r.is_err());
    assert_eq!(v.len(), 4); // 3 originals + the 1 produced before the panic
    assert_eq!(
        v.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 1, 2, 101]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(v);
    assert_eq!(drops.load(Ordering::SeqCst), 4); // each live element exactly once
}

#[test]
fn resize_panicking_clone_drops_are_balanced() {
    // `resize`'s grow path routes the first `additional - 1` clones through
    // `fill_with` (the `|_| value.clone()` closure in the stable
    // configuration; `CloneRepeat::tail_fill`'s element-wise loop under
    // `specialization`), so a panicking `T::clone` exercises the same
    // per-element publication: the clones that landed stay counted and the
    // in-flight `value` is dropped exactly once during unwinding.
    let drops = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let mut v = CovariantVec::<BombClone, 4>::new();
    for _ in 0..2 {
        v.push(BombClone {
            drops: Arc::clone(&drops),
            clones: Arc::clone(&clones),
            panic_on: usize::MAX,
        });
    }
    assert!(!v.is_heap());
    let r = catch_unwind(AssertUnwindSafe(|| {
        v.resize(
            6,
            BombClone {
                drops: Arc::clone(&drops),
                clones: Arc::clone(&clones),
                panic_on: 2, // the third `clone` call panics
            },
        );
    }));
    assert!(r.is_err());
    assert_eq!(clones.load(Ordering::SeqCst), 3); // two landed, the third panicked
    assert_eq!(drops.load(Ordering::SeqCst), 1); // the in-flight `value`
    assert_eq!(v.len(), 4); // 2 originals + the 2 clones that landed
    drop(v);
    assert_eq!(drops.load(Ordering::SeqCst), 5); // 2 originals + 2 clones + `value`
}

// -------------------------------------------------------------------------- //
// Panicking iterator during extend                                            //
// -------------------------------------------------------------------------- //

struct PanicAfter {
    yielded: usize,
    panic_at: usize,
}

impl Iterator for PanicAfter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        assert!(self.yielded != self.panic_at, "intentional iterator panic");
        self.yielded += 1;
        Some(self.yielded as i32)
    }
}

#[test]
fn extend_with_panicking_iterator_leaves_consistent() {
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([0, 1]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        v.extend(PanicAfter {
            yielded: 0,
            panic_at: 3,
        });
    }));
    assert!(r.is_err());
    // Three items were appended before the iterator panicked.
    assert_eq!(&*v, &[0, 1, 1, 2, 3]);
    assert_eq!(v.len(), 5);
}

#[test]
fn extend_heap_fast_path_panicking_iterator_leaves_prefix() {
    // Panic *inside* the heap fast path (spare capacity covers the yielded
    // items): the length is published after each write, so the panic leaves
    // exactly the appended prefix counted — the fast loop's version of
    // `extend_with_panicking_iterator_leaves_consistent` above.
    let mut v = CovariantVec::<i32, 2>::new();
    v.extend([0, 1, 2]); // spills: len 3, cap 4
    v.reserve(5); // grows to cap 8: five spare slots
    assert!(v.is_heap());
    assert_eq!(v.capacity(), 8);
    let r = catch_unwind(AssertUnwindSafe(|| {
        v.extend(PanicAfter {
            yielded: 0,
            panic_at: 2, // yields 1, 2, then panics
        });
    }));
    assert!(r.is_err());
    assert_eq!(&*v, &[0, 1, 2, 1, 2]);
    assert_eq!(v.len(), 5);
    assert_eq!(v.capacity(), 8); // the panic happened before any realloc
}

#[test]
fn from_iter_with_panicking_iterator_leaves_nothing_observable() {
    // `from_iter` panics out of the constructor; the partially built vector is
    // dropped during unwinding (no leak, no double drop — verified by the
    // absence of an abort under Miri).
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _v: CovariantVec<i32, 2> = PanicAfter {
            yielded: 0,
            panic_at: 4,
        }
        .collect();
    }));
    assert!(r.is_err());
}

// -------------------------------------------------------------------------- //
// clone_from                                                                  //
// -------------------------------------------------------------------------- //

#[test]
fn clone_from_reuses_destination_capacity() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut dest =
        CovariantVec::<ResizeTracked, 2>::from_elem(ResizeTracked::new(1, &clones, &drops), 8);
    assert!(dest.is_heap());
    let big_cap = dest.capacity();
    let src =
        CovariantVec::<ResizeTracked, 2>::from_elem(ResizeTracked::new(9, &clones, &drops), 3);
    assert!(src.is_heap());

    clones.store(0, Ordering::SeqCst);
    drops.store(0, Ordering::SeqCst);
    dest.clone_from(&src);
    // The destination's block is reused, not reallocated.
    assert_eq!(dest.capacity(), big_cap);
    assert_eq!(dest.len(), src.len());
    assert_eq!(dest[0].val, 1009); // clone: `val` shifted by 1000
    assert_eq!(dest[1].val, 2009); // clone of a clone: shifted twice
    assert_eq!(dest[2].val, 2009);
    // The destination's old 8 elements were dropped; the source's were cloned.
    assert_eq!(drops.load(Ordering::SeqCst), 8);
    assert_eq!(clones.load(Ordering::SeqCst), 3);
    drop(dest);
    drop(src);
}

#[test]
fn clone_from_zst_len_gt_n_reruns_clones() {
    // A ZST never spills, so `len > N` is still the inline state. `clone_from`
    // takes the `size_of::<T>() == 0` reuse branch (destination storage is
    // irrelevant), clears the old count, and — deliberately, unlike `to_mut`'s
    // count adoption — runs `T::clone` once per source element.
    static CLONES: AtomicUsize = AtomicUsize::new(0);
    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct Z;
    impl Clone for Z {
        fn clone(&self) -> Self {
            CLONES.fetch_add(1, Ordering::SeqCst);
            Z
        }
    }
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    let mut dest = CovariantVec::<Z, 4>::new();
    for _ in 0..10 {
        dest.push(Z); // `push` adopts: no clones, no drops
    }
    let mut src = CovariantVec::<Z, 4>::new();
    for _ in 0..7 {
        src.push(Z);
    }
    assert!(!dest.is_heap() && !src.is_heap());
    assert_eq!(dest.len(), 10);

    CLONES.store(0, Ordering::SeqCst);
    DROPS.store(0, Ordering::SeqCst);
    dest.clone_from(&src);
    assert_eq!(dest.len(), 7);
    assert!(!dest.is_heap());
    assert_eq!(CLONES.load(Ordering::SeqCst), 7); // one clone per source element
    assert_eq!(DROPS.load(Ordering::SeqCst), 10); // the old 10 were dropped

    drop(dest);
    drop(src);
    assert_eq!(DROPS.load(Ordering::SeqCst), 10 + 7 + 7); // both counts exactly once
}

#[test]
fn clone_from_panic_mid_fill_leaves_valid_partial_prefix() {
    // Element that clones fine twice, then panics on the third clone.
    struct Boom {
        val: i32,
        seen: Arc<AtomicUsize>,
    }
    impl Clone for Boom {
        fn clone(&self) -> Self {
            let n = self.seen.fetch_add(1, Ordering::SeqCst);
            if n == 2 {
                panic!("clone boom");
            }
            Boom {
                val: self.val,
                seen: Arc::clone(&self.seen),
            }
        }
    }
    let seen = Arc::new(AtomicUsize::new(0));
    let mut dest: CovariantVec<Boom, 8> = CovariantVec::new();
    let src: CovariantVec<Boom, 8> = (0..5)
        .map(|i| Boom {
            val: i,
            seen: Arc::clone(&seen),
        })
        .collect();

    let r = catch_unwind(AssertUnwindSafe(|| dest.clone_from(&src)));
    assert!(r.is_err());
    // The published prefix is exactly the clones completed before the panic;
    // it stays valid (droppable) — `Drop` reclaims it without double drops.
    assert_eq!(dest.len(), 2);
    assert_eq!(dest[0].val, 0);
    assert_eq!(dest[1].val, 1);
}

/// An iterator whose lower-bound `size_hint` is a lie of `usize::MAX`.
struct MaxHintIter {
    items: std::vec::IntoIter<i32>,
}

impl Iterator for MaxHintIter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        self.items.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

#[test]
fn extend_with_usize_max_hint_panics_before_allocating() {
    // A hint beyond `isize::MAX` elements is rejected by the element-count
    // bound ("capacity overflow") *before* any allocation is attempted; only a
    // huge-but-layout-valid hint would reach the allocator (and abort there).
    let mut v = CovariantVec::<i32, 4>::new();
    v.extend([1, 2, 3]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        v.extend(MaxHintIter {
            items: vec![4, 5, 6].into_iter(),
        });
    }));
    assert!(r.is_err());
    // Untouched: the panic happens during the up-front reservation.
    assert_eq!(&*v, &[1, 2, 3]);
}
