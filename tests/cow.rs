//! Tests for the optional three-state copy-on-write `CovariantCow`.

#![cfg(feature = "cow")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use covariant_vec::CovariantCow;

#[test]
fn borrowed_is_zero_copy() {
    let data = vec![1, 2, 3, 4, 5];
    let c = CovariantCow::<i32, 2>::from_borrowed(&data);
    assert!(c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(c.len(), 5);
    assert_eq!(&*c, &[1, 2, 3, 4, 5]);

    // `as_borrowed` hands out the original long-lived borrow.
    let b = c.as_borrowed().unwrap();
    assert_eq!(b.as_ptr(), data.as_ptr());
}

#[test]
fn to_mut_materializes_inline() {
    let data = [10, 20, 30];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    assert!(c.is_borrowed());

    c.to_mut()[0] = 99;
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(&*c, &[99, 20, 30]);
    // The borrowed source is untouched (copy-on-write).
    assert_eq!(data, [10, 20, 30]);
}

#[test]
fn to_mut_materializes_heap() {
    let data: Vec<i32> = (0..10).collect();
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    assert!(c.is_borrowed());

    c.to_mut()[0] = 99;
    assert!(c.is_heap());
    assert_eq!(c[0], 99);
    assert_eq!(c[9], 9);
    assert_eq!(data[0], 0);
}

#[test]
fn push_after_borrow() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.push(4);
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[1, 2, 3, 4]);
    assert_eq!(c.pop(), Some(4));
    assert_eq!(&*c, &[1, 2, 3]);
}

#[test]
fn deref_mut_copies_on_write() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c[1] = 20; // DerefMut -> to_mut
    assert_eq!(&*c, &[1, 20, 3]);
    assert_eq!(data, [1, 2, 3]);
}

#[test]
fn clear_and_clone() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let d = c.clone();
    assert!(d.is_borrowed()); // cloning a borrow stays zero-copy
    assert_eq!(c, d);

    c.clear();
    // `clear` is `truncate(0)`: a borrowed cow stays borrowed, narrowed to
    // empty (consistent with `truncate`/`resize`-shrink).
    assert!(c.is_borrowed());
    assert!(c.is_empty());
    assert_eq!(&*c, &[]);
    assert_eq!(d.len(), 3);
}

#[test]
fn clear_owned_preserves_state_and_capacity() {
    // Heap: keeps the block and capacity.
    let mut h: CovariantCow<i32, 2> = (0..8).collect();
    assert!(h.is_heap());
    let cap = h.capacity();
    h.clear();
    assert!(h.is_empty());
    assert!(h.is_heap());
    assert_eq!(h.capacity(), cap);

    // Inline: stays inline.
    let mut i: CovariantCow<i32, 8> = (0..3).collect();
    assert!(!i.is_heap());
    i.clear();
    assert!(i.is_empty());
    assert!(!i.is_heap());
    assert_eq!(i.capacity(), 8);
}

#[test]
fn cow_zst() {
    let mut c = CovariantCow::<(), 4>::new();
    for _ in 0..1000 {
        c.push(());
    }
    assert_eq!(c.len(), 1000);
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
}

// --------------------------------------------------------------------------
// `as_mut_slice` on a borrowed cow must not hand out `&mut` (aliasing UB)
// --------------------------------------------------------------------------

#[test]
#[should_panic(expected = "borrowed Cow")]
fn as_mut_slice_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.as_mut_slice(); // must panic in release too, not just debug
}

// --------------------------------------------------------------------------
// `to_mut` must be panic-safe
// --------------------------------------------------------------------------

#[derive(Debug)]
struct PanicClone {
    val: i32,
    ctr: Arc<AtomicUsize>,
    panic_at: usize,
}

impl Clone for PanicClone {
    fn clone(&self) -> Self {
        let n = self.ctr.fetch_add(1, Ordering::SeqCst);
        assert!(n != self.panic_at, "intentional clone panic");
        PanicClone {
            val: self.val,
            ctr: Arc::clone(&self.ctr),
            panic_at: self.panic_at,
        }
    }
}

fn make_data(n: i32, ctr: &Arc<AtomicUsize>, panic_at: usize) -> Vec<PanicClone> {
    (0..n)
        .map(|i| PanicClone {
            val: i,
            ctr: Arc::clone(ctr),
            panic_at,
        })
        .collect()
}

#[test]
fn to_mut_is_panic_safe_inline() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let data = make_data(3, &ctr, 1);
    let mut c = CovariantCow::<PanicClone, 4>::from_borrowed(&data);

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = c.to_mut();
    }));
    assert!(r.is_err());
    // The first element cloned; the second panicked. State must be valid.
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(c.len(), 1);
    drop(c); // must not be UB
}

#[test]
fn to_mut_is_panic_safe_heap() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let data = make_data(10, &ctr, 3);
    let mut c = CovariantCow::<PanicClone, 2>::from_borrowed(&data);

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = c.to_mut();
    }));
    assert!(r.is_err());
    assert!(!c.is_borrowed());
    assert!(c.is_heap());
    assert_eq!(c.len(), 3);
    drop(c); // frees the partially-filled heap block, no leak/UB
}

// --------------------------------------------------------------------------
// Ordering / hashing forwarding
// --------------------------------------------------------------------------

#[test]
fn ord_and_hash() {
    let a_src = [1, 2, 3];
    let b_src = [1, 2, 4];
    let a = CovariantCow::<i32, 4>::from_borrowed(&a_src);
    let b = CovariantCow::<i32, 4>::from_borrowed(&b_src);
    assert!(a < b);

    let mut h1 = DefaultHasher::new();
    a.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    CovariantCow::<i32, 4>::from_borrowed(&a_src).hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

// --------------------------------------------------------------------------
// The owned-path API does not require `T: Clone`
// --------------------------------------------------------------------------

struct NoClone(i32);

#[test]
fn push_owned_pop_owned_need_no_clone() {
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.push_owned(NoClone(1));
    c.push_owned(NoClone(2));
    assert_eq!(c.len(), 2);
    assert_eq!(c.pop_owned().map(|x| x.0), Some(2));
    assert_eq!(c.len(), 1);
}

#[test]
#[should_panic(expected = "borrowed CovariantCow")]
fn push_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.push_owned(4);
}

#[test]
#[should_panic(expected = "borrowed CovariantCow")]
fn pop_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.pop_owned();
}

// --------------------------------------------------------------------------
// as_ptr / resize / Into<Vec>
// --------------------------------------------------------------------------

#[test]
fn cow_as_ptr_and_resize() {
    let mut c = CovariantCow::<i32, 2>::new();
    c.push_owned(1);
    assert_eq!(unsafe { *c.as_ptr() }, 1);
    unsafe { *c.as_mut_ptr() = 5 };
    assert_eq!(c[0], 5);

    c.resize(4, 9);
    assert_eq!(&*c, &[5, 9, 9, 9]);
    c.resize(1, 0);
    assert_eq!(&*c, &[5]);
}

#[test]
fn into_vec_from_cow() {
    // heap state
    let mut c = CovariantCow::<i32, 2>::new();
    for i in 0..6 {
        c.push(i);
    }
    let v: Vec<i32> = c.into();
    assert_eq!(v, (0..6).collect::<Vec<_>>());

    // borrowed state (copied out)
    let data = [7, 8, 9];
    let c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let v: Vec<i32> = c.into();
    assert_eq!(v, vec![7, 8, 9]);

    // inline state
    let mut c = CovariantCow::<i32, 4>::new();
    c.push_owned(1);
    c.push_owned(2);
    let v: Vec<i32> = c.into();
    assert_eq!(v, vec![1, 2]);
}

// --------------------------------------------------------------------------
// API-parity additions
// --------------------------------------------------------------------------

#[test]
fn from_vec_owned() {
    let c = CovariantCow::<i32, 4>::from(vec![1, 2, 3]);
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(&*c, &[1, 2, 3]);

    let mut big: Vec<i32> = Vec::with_capacity(100);
    big.extend([1, 2, 3]);
    let c = CovariantCow::<i32, 4>::from(big);
    assert!(c.is_heap());
    assert_eq!(&*c, &[1, 2, 3]);

    // `From<&[T]>` must not require `T: Clone` — exercised by construction.
    let data = [1, 2, 3];
    let c: CovariantCow<i32, 4> = CovariantCow::from(&data[..]);
    assert!(c.is_borrowed());
}

#[test]
fn extend_and_from_iter() {
    let mut c: CovariantCow<i32, 4> = (0..3).collect();
    c.extend([3, 4, 5]);
    assert_eq!(&*c, &[0, 1, 2, 3, 4, 5]);

    // Extending a borrowed cow materializes it.
    let data = [1, 2, 3];
    let mut b = CovariantCow::<i32, 4>::from_borrowed(&data);
    b.extend([4, 5]);
    assert!(!b.is_borrowed());
    assert_eq!(&*b, &[1, 2, 3, 4, 5]);
}

#[test]
fn into_iter_owned_and_by_ref() {
    let c: CovariantCow<i32, 4> = (0..5).collect();
    let v: Vec<i32> = c.into_iter().collect();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);

    let c: CovariantCow<i32, 4> = (0..3).collect();
    let s: i32 = (&c).into_iter().sum();
    assert_eq!(s, 3);

    let mut c: CovariantCow<i32, 4> = (0..3).collect();
    for x in &mut c {
        *x *= 10;
    }
    assert_eq!(&*c, &[0, 10, 20]);
}

#[test]
fn retain_truncate_insert_remove_swap_remove() {
    let mut c: CovariantCow<i32, 4> = (0..10).collect();
    c.retain(|x| x % 2 == 0);
    assert_eq!(&*c, &[0, 2, 4, 6, 8]);

    c.truncate(3);
    assert_eq!(&*c, &[0, 2, 4]);

    c.insert(0, 9);
    assert_eq!(&*c, &[9, 0, 2, 4]);
    assert_eq!(c.remove(0), 9);
    assert_eq!(&*c, &[0, 2, 4]);
    assert_eq!(c.swap_remove(0), 0); // last (4) takes its place
    assert_eq!(&*c, &[4, 2]);
}

#[test]
fn truncate_borrowed_stays_borrowed() {
    let data = [1, 2, 3, 4];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    c.truncate(2);
    assert!(c.is_borrowed());
    assert_eq!(&*c, &[1, 2]);
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), data.as_ptr());
}

// A clone-counting element, to prove the shrink-borrowed path copies nothing.
struct CountClone {
    val: i32,
    ctr: Arc<AtomicUsize>,
}
impl Clone for CountClone {
    fn clone(&self) -> Self {
        self.ctr.fetch_add(1, Ordering::SeqCst);
        CountClone {
            val: self.val,
            ctr: Arc::clone(&self.ctr),
        }
    }
}

#[test]
fn resize_shrink_borrowed_does_not_clone() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let data: Vec<CountClone> = (0..1000)
        .map(|i| CountClone {
            val: i,
            ctr: Arc::clone(&ctr),
        })
        .collect();
    let mut c = CovariantCow::<CountClone, 4>::from_borrowed(&data);
    c.resize(
        3,
        CountClone {
            val: -1,
            ctr: Arc::clone(&ctr),
        },
    );
    assert!(c.is_borrowed());
    assert_eq!(c.len(), 3);
    assert_eq!(ctr.load(Ordering::SeqCst), 0); // 1000 clones avoided
    assert_eq!(c[0].val, 0);
    assert_eq!(c[2].val, 2);
}

#[test]
fn resize_grow_bulk_reserves() {
    // Growing a borrowed cow materializes it, then reserves once and fills.
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.resize(100, 9);
    assert_eq!(c.len(), 100);
    assert!(c.capacity() >= 100);
    assert_eq!(&c[..5], &[1, 2, 3, 9, 9]);
    assert!(c[99] == 9);
}

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

static ZC_CLONE: AtomicUsize = AtomicUsize::new(0);
static ZC_DROP: AtomicUsize = AtomicUsize::new(0);

struct ZCow;
impl Clone for ZCow {
    fn clone(&self) -> Self {
        ZC_CLONE.fetch_add(1, Ordering::SeqCst);
        ZCow
    }
}
impl Drop for ZCow {
    fn drop(&mut self) {
        ZC_DROP.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn resize_grow_owned_inline_moves_original_into_last_slot() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut c: CovariantCow<ResizeTracked, 4> = (0..3)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    // 3 -> 6: two clones, and the original argument lands in slot 5.
    c.resize(6, ResizeTracked::new(7, &clones, &drops));
    assert_eq!(c.len(), 6);
    assert_eq!(clones.load(Ordering::SeqCst), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 0); // the moved original is not dropped
    for i in 0..3 {
        assert_eq!(c[i].val, i as i32);
    }
    for i in 3..5 {
        assert_eq!(c[i].val, 1007);
    }
    assert_eq!(c[5].val, 7);
    drop(c);
    assert_eq!(drops.load(Ordering::SeqCst), 6); // exactly `new_len` destructors
}

#[test]
fn resize_grow_owned_heap_moves_original_into_last_slot() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut c: CovariantCow<ResizeTracked, 2> = (0..5)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();
    assert!(c.is_heap());
    // 5 -> 8: two clones, and the original argument lands in slot 7.
    c.resize(8, ResizeTracked::new(7, &clones, &drops));
    assert_eq!(c.len(), 8);
    assert_eq!(clones.load(Ordering::SeqCst), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    for i in 5..7 {
        assert_eq!(c[i].val, 1007);
    }
    assert_eq!(c[7].val, 7);
    drop(c);
    assert_eq!(drops.load(Ordering::SeqCst), 8);
}

#[test]
fn resize_grow_borrowed_moves_original_into_last_slot() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let data: Vec<ResizeTracked> = (0..5)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();
    let mut c = CovariantCow::<ResizeTracked, 4>::from_borrowed(&data);
    c.resize(9, ResizeTracked::new(7, &clones, &drops));
    assert!(!c.is_borrowed());
    assert_eq!(c.len(), 9);
    // Materialization clones the 5 borrowed elements, then the fill clones
    // `additional - 1 = 3` more; the original moves into the last slot.
    assert_eq!(clones.load(Ordering::SeqCst), 5 + 3);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    // Slots 0..5 are the materialized copies; 5..8 are the fill clones; slot 8
    // is the moved original (unshifted `val`).
    for i in 0..5 {
        assert_eq!(c[i].val, (i as i32) + 1000);
    }
    for i in 5..8 {
        assert_eq!(c[i].val, 1007);
    }
    assert_eq!(c[8].val, 7);
    drop(c);
    assert_eq!(drops.load(Ordering::SeqCst), 9);
    // The borrower's originals are untouched and still drop later.
    assert_eq!(data[4].val, 4);
}

#[test]
fn resize_grow_cow_zst_clone_drop_counts() {
    // A borrowed ZST cow 5 -> 9: materialization adopts the count (no clones),
    // then the fill runs exactly `additional - 1 = 3` clones, and the cow ends
    // up owning `new_len` values.
    ZC_CLONE.store(0, Ordering::SeqCst);
    ZC_DROP.store(0, Ordering::SeqCst);
    let data = [ZCow, ZCow, ZCow, ZCow, ZCow];
    let mut c = CovariantCow::<ZCow, 4>::from_borrowed(&data);
    c.resize(9, ZCow);
    assert_eq!(c.len(), 9);
    assert_eq!(ZC_CLONE.load(Ordering::SeqCst), 3);
    assert_eq!(ZC_DROP.load(Ordering::SeqCst), 0);
    drop(c);
    assert_eq!(ZC_DROP.load(Ordering::SeqCst), 9);
}

/// An iterator that lies about its lower-bound `size_hint`, reporting `0`.
struct CowLyingIter {
    items: std::vec::IntoIter<i32>,
}

impl Iterator for CowLyingIter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        self.items.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Deliberately understate the lower bound.
        (0, self.items.size_hint().1)
    }
}

/// Lie-about-`size_hint` variant yielding references (for the `T: Copy` impl).
struct CowLyingRefIter<'x> {
    items: std::slice::Iter<'x, i32>,
}

impl<'x> Iterator for CowLyingRefIter<'x> {
    type Item = &'x i32;
    fn next(&mut self) -> Option<&'x i32> {
        self.items.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.items.size_hint().1)
    }
}

#[test]
fn extend_lying_size_hint_on_borrowed_cow() {
    // `extend` trusts `size_hint().0` only as a *reservation*; an understated
    // bound must not lose elements or corrupt state, even though the cow starts
    // borrowed (peek then materialize in one allocation).
    let data = [1i32, 2, 3];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    c.extend(CowLyingIter {
        items: (10..18).collect::<Vec<i32>>().into_iter(),
    });
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17]);

    // Same for the `T: Copy` by-reference variant.
    let more: Vec<i32> = (20..26).collect();
    c.extend(CowLyingRefIter { items: more.iter() });
    assert_eq!(
        &*c,
        &[1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
    );
}

#[test]
fn extend_owned_lying_size_hint_via_extend_trait() {
    // The owned arm of `Extend` delegates straight to `extend_owned` (single
    // reserve, no peek); an understated hint must still append everything,
    // growing through `push_owned` as needed.
    let mut c = CovariantCow::<i32, 2>::new();
    c.extend_owned([1, 2]);
    assert!(!c.is_heap());
    c.extend(CowLyingIter {
        items: (10..16).collect::<Vec<i32>>().into_iter(), // lo == 0
    });
    assert_eq!(&*c, &[1, 2, 10, 11, 12, 13, 14, 15]);

    // Same for the `T: Copy` by-reference impl.
    let more: Vec<i32> = (20..24).collect();
    c.extend(CowLyingRefIter { items: more.iter() });
    assert_eq!(&*c, &[1, 2, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]);
}

#[test]
fn try_reserve_err_leaves_borrowed_cow_untouched() {
    let data = [1u8, 2, 3];
    let mut c = CovariantCow::<u8, 4>::from_borrowed(&data);
    let src = c.as_borrowed().unwrap().as_ptr();
    // `len + additional` overflows -> `Err` before any state change.
    assert!(c.try_reserve(usize::MAX).is_err());
    // A non-overflowing but impossible request (beyond `MAX_LEN`).
    assert!(c.try_reserve(usize::MAX - 4).is_err());
    assert!(c.is_borrowed());
    assert_eq!(c.len(), 3);
    assert_eq!(&*c, &data);
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), src);
}

#[test]
fn pop_borrowed_clones_only_one() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let data: Vec<CountClone> = (0..5)
        .map(|i| CountClone {
            val: i,
            ctr: Arc::clone(&ctr),
        })
        .collect();
    let mut c = CovariantCow::<CountClone, 2>::from_borrowed(&data);
    assert!(c.is_borrowed());
    let last = c.pop().unwrap();
    assert_eq!(last.val, 4);
    assert_eq!(ctr.load(Ordering::SeqCst), 1); // one clone, not five
    assert!(c.is_borrowed()); // stays borrowed
    assert_eq!(c.len(), 4);
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), data.as_ptr());
}

#[test]
fn extend_empty_on_borrowed_does_not_materialize() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let data: Vec<CountClone> = (0..5)
        .map(|i| CountClone {
            val: i,
            ctr: Arc::clone(&ctr),
        })
        .collect();
    let mut c = CovariantCow::<CountClone, 2>::from_borrowed(&data);
    c.extend(core::iter::empty());
    assert!(c.is_borrowed()); // not materialized
    assert_eq!(ctr.load(Ordering::SeqCst), 0); // no clones
    assert_eq!(c.len(), 5);
}

// --------------------------------------------------------------------------
// `with_capacity` / const `from_borrowed`
// --------------------------------------------------------------------------

#[test]
fn with_capacity_and_const_borrowed() {
    let c = CovariantCow::<i32, 4>::with_capacity(10);
    assert!(c.is_heap());
    assert!(c.capacity() >= 10);
    assert!(c.is_empty());

    // `capacity <= N` stays inline.
    let c = CovariantCow::<i32, 4>::with_capacity(3);
    assert!(!c.is_heap());
    assert!(!c.is_borrowed());

    // `from_borrowed` is usable in a `const` context.
    const B: CovariantCow<'static, i32, 4> = CovariantCow::from_borrowed(&[1, 2, 3]);
    assert_eq!(&*B, &[1, 2, 3]);
    assert!(B.is_borrowed());
}

// --------------------------------------------------------------------------
// try_reserve / try_reserve_exact (owned and borrowed)
// --------------------------------------------------------------------------

#[test]
fn cow_try_reserve_owned_grows_and_errs() {
    use covariant_vec::TryReserveErrorKind;
    let mut c: CovariantCow<i32, 4> = CovariantCow::with_capacity(2);
    c.extend([1, 2, 3]);
    assert!(c.try_reserve_owned(50).is_ok());
    assert!(c.capacity() >= 3 + 50);
    assert_eq!(&*c, &[1, 2, 3]);
    assert_eq!(
        c.try_reserve_owned(usize::MAX).unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
}

#[test]
fn cow_reserve_exact_is_exact() {
    let mut c: CovariantCow<u32, 4> = CovariantCow::new();
    c.reserve_exact(50);
    assert_eq!(c.capacity(), 50);
    let mut d: CovariantCow<u32, 4> = CovariantCow::new();
    d.reserve_exact_owned(50);
    assert_eq!(d.capacity(), 50);
    d.reserve_exact_owned(0); // no-op
    assert_eq!(d.capacity(), 50);
}

#[test]
fn cow_try_reserve_borrowed_materializes_heap() {
    // A borrowed cow is materialized (copy-on-write) as part of the fallible
    // reservation, fused into one allocation; data and capacity are correct.
    let data = [1, 2, 3, 4, 5];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    assert!(c.is_borrowed());
    assert!(c.try_reserve(20).is_ok());
    assert!(!c.is_borrowed());
    assert!(c.is_heap());
    assert!(c.capacity() >= 5 + 20);
    assert_eq!(&*c, &[1, 2, 3, 4, 5]);
}

#[test]
fn cow_try_reserve_borrowed_inline_target() {
    // `N = 8`, borrowed len 3; a small exact request keeps the result inline.
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 8>::from_borrowed(&data);
    assert!(c.try_reserve_exact(2).is_ok()); // needs 5 <= N
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert!(c.capacity() >= 5);
    assert_eq!(&*c, &[1, 2, 3]);
}

#[test]
fn cow_try_reserve_borrowed_fits_inline_stays_inline() {
    // The non-exact path must also avoid the amortized first-spill when the
    // request already fits inline (parity with `reserve`, which materializes
    // inline and then finds the request satisfied).
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 8>::from_borrowed(&data);
    assert!(c.try_reserve(1).is_ok()); // needs 4 <= N = 8
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(&*c, &[1, 2, 3]);
}

#[test]
fn cow_try_reserve_zero_on_borrowed_stays_borrowed() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    assert!(c.try_reserve(0).is_ok());
    assert!(c.is_borrowed()); // no materialization for a zero request
}

#[test]
#[should_panic(expected = "borrowed CovariantCow")]
fn cow_try_reserve_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    let _ = c.try_reserve_owned(1);
}

#[test]
fn cow_try_reserve_zst_bounds() {
    use covariant_vec::TryReserveErrorKind;
    // The cow spends bit 62 on the borrowed tag, so its ZST length bound is
    // `2^62 - 1` rather than `isize::MAX`.
    let bound = (1usize << 62) - 1;
    let mut c: CovariantCow<(), 4> = CovariantCow::with_capacity(0);
    assert!(c.try_reserve(bound).is_ok());
    assert_eq!(
        c.try_reserve(bound + 1).unwrap_err().kind(),
        TryReserveErrorKind::CapacityOverflow
    );
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn cow_zst_with_capacity_too_large_panics() {
    // Symmetric with `reserve_owned`: a ZST request above the 62-bit length
    // bound is rejected rather than silently accepted.
    let _ = CovariantCow::<(), 4>::with_capacity(1usize << 62);
}

#[test]
fn cow_n_zero_first_spill_is_floored() {
    // `N = 0`: the first heap block is floored to `min_non_zero_cap`.
    let mut c = CovariantCow::<u8, 0>::new();
    c.push_owned(1);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), 8);

    let mut d = CovariantCow::<u32, 0>::new();
    d.push_owned(1);
    assert!(d.is_heap());
    assert_eq!(d.capacity(), 4);
}

#[test]
fn cow_spare_capacity_mut_owned_workflow() {
    let mut c: CovariantCow<i32, 4> = CovariantCow::with_capacity(8);
    assert!(c.is_heap());
    let cap = c.capacity();
    {
        let spare = c.spare_capacity_mut_owned();
        assert_eq!(spare.len(), cap);
        spare[0].write(1);
        spare[1].write(2);
    }
    unsafe { c.set_len(2) };
    assert_eq!(&*c, &[1, 2]);
    assert_eq!(c.spare_capacity_mut_owned().len(), cap - 2);
}

#[test]
#[should_panic(expected = "borrowed CovariantCow")]
fn cow_spare_capacity_mut_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    let _ = c.spare_capacity_mut_owned();
}

#[test]
fn cow_from_array_reference_borrows_zero_copy() {
    let arr = [10, 20, 30];
    let c = CovariantCow::<i32, 2>::from(&arr);
    assert!(c.is_borrowed());
    assert_eq!(c.as_ptr(), arr.as_ptr());
    assert_eq!(&*c, &[10, 20, 30]);
}

/// An iterator that yields `1, 2, ...` and panics on the `panic_at`-th `next`.
struct CowPanicAfter {
    yielded: usize,
    panic_at: usize,
}

impl Iterator for CowPanicAfter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        assert!(self.yielded != self.panic_at, "intentional iterator panic");
        self.yielded += 1;
        Some(self.yielded as i32)
    }
}

#[test]
fn cow_extend_owned_panicking_iterator_leaves_prefix() {
    let mut c: CovariantCow<i32, 2> = CovariantCow::new();
    c.extend_owned([0, 1]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.extend_owned(CowPanicAfter {
            yielded: 0,
            panic_at: 3,
        });
    }));
    assert!(r.is_err());
    // Three items were appended before the iterator panicked; the prefix stands.
    assert_eq!(&*c, &[0, 1, 1, 2, 3]);
    assert_eq!(c.len(), 5);
}

#[test]
fn cow_extend_owned_heap_fast_path_panicking_iterator_leaves_prefix() {
    // The cow twin of the `CovariantVec` test: panic inside the heap fast
    // path, with spare capacity covering the yielded items.
    let mut c: CovariantCow<i32, 2> = CovariantCow::new();
    c.extend_owned([0, 1, 2]); // spills: len 3, cap 4
    c.reserve_owned(5); // grows to cap 8
    assert!(c.is_heap());
    assert_eq!(c.capacity(), 8);
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.extend_owned(CowPanicAfter {
            yielded: 0,
            panic_at: 2, // yields 1, 2, then panics
        });
    }));
    assert!(r.is_err());
    assert_eq!(&*c, &[0, 1, 2, 1, 2]);
    assert_eq!(c.len(), 5);
    assert_eq!(c.capacity(), 8); // the panic happened before any realloc
}

#[test]
fn cow_extend_borrowed_panicking_iterator_leaves_prefix() {
    // The borrowed-aware `Extend` materializes first (copy-on-write), then
    // appends; a mid-iteration panic leaves the materialized prefix + appended
    // items, and the cow is owned.
    let data = [9, 9];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    assert!(c.is_borrowed());
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.extend(CowPanicAfter {
            yielded: 0,
            panic_at: 3,
        });
    }));
    assert!(r.is_err());
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[9, 9, 1, 2, 3]);
    assert_eq!(c.len(), 5);
}

// --------------------------------------------------------------------------
// Clone panic safety (inline and heap owned states)
// --------------------------------------------------------------------------

struct CowCloneBomb {
    val: i32,
    clone_ctr: Arc<AtomicUsize>,
    drop_ctr: Arc<AtomicUsize>,
    panic_at: usize,
}

impl Clone for CowCloneBomb {
    fn clone(&self) -> Self {
        let n = self.clone_ctr.fetch_add(1, Ordering::SeqCst);
        assert!(n != self.panic_at, "intentional clone panic");
        CowCloneBomb {
            val: self.val,
            clone_ctr: Arc::clone(&self.clone_ctr),
            drop_ctr: Arc::clone(&self.drop_ctr),
            panic_at: self.panic_at,
        }
    }
}

impl Drop for CowCloneBomb {
    fn drop(&mut self) {
        self.drop_ctr.fetch_add(1, Ordering::SeqCst);
    }
}

fn bomb(
    i: i32,
    clone_ctr: &Arc<AtomicUsize>,
    drop_ctr: &Arc<AtomicUsize>,
    at: usize,
) -> CowCloneBomb {
    CowCloneBomb {
        val: i,
        clone_ctr: Arc::clone(clone_ctr),
        drop_ctr: Arc::clone(drop_ctr),
        panic_at: at,
    }
}

#[test]
fn cow_clone_panic_safety_inline() {
    let clone_ctr = Arc::new(AtomicUsize::new(0));
    let drop_ctr = Arc::new(AtomicUsize::new(0));
    let mut src = CovariantCow::<CowCloneBomb, 8>::new();
    for i in 0..4 {
        src.push_owned(bomb(i, &clone_ctr, &drop_ctr, 2));
    }
    assert!(!src.is_heap() && !src.is_borrowed());

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = src.clone();
    }));
    assert!(r.is_err());
    assert_eq!(clone_ctr.load(Ordering::SeqCst), 3); // 0,1 ok; 2 panicked
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 2); // partial clone dropped once
    assert_eq!(src.len(), 4);
    drop(src);
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 6);
}

#[test]
fn cow_clone_panic_safety_heap() {
    let clone_ctr = Arc::new(AtomicUsize::new(0));
    let drop_ctr = Arc::new(AtomicUsize::new(0));
    let mut src = CovariantCow::<CowCloneBomb, 2>::new();
    for i in 0..5 {
        src.push_owned(bomb(i, &clone_ctr, &drop_ctr, 3));
    }
    assert!(src.is_heap());

    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = src.clone();
    }));
    assert!(r.is_err());
    assert_eq!(clone_ctr.load(Ordering::SeqCst), 4); // 0,1,2 ok; 3 panicked
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 3);
    assert_eq!(src.len(), 5);
    drop(src);
    assert_eq!(drop_ctr.load(Ordering::SeqCst), 8); // frees the partial heap block
}

// --------------------------------------------------------------------------
// retain / retain_mut on a borrowed cow (materialization path)
// --------------------------------------------------------------------------

#[test]
fn retain_on_borrowed_materializes_heap() {
    let data = [1, 2, 3, 4, 5, 6];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    assert!(c.is_borrowed());
    c.retain(|x| x % 2 == 0); // len 6 > N: materializes to the heap state
    assert!(!c.is_borrowed());
    assert!(c.is_heap());
    assert_eq!(&*c, &[2, 4, 6]);
    assert_eq!(data, [1, 2, 3, 4, 5, 6]); // borrower untouched
}

#[test]
fn retain_mut_on_borrowed_materializes_inline() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.retain_mut(|x| {
        *x *= 10;
        *x != 20
    }); // len 3 <= N: materializes inline
    assert!(!c.is_borrowed());
    assert!(!c.is_heap());
    assert_eq!(&*c, &[10, 30]);
    assert_eq!(data, [1, 2, 3]);
}

// --------------------------------------------------------------------------
// No-`Clone` owned API (retain_*_owned / into_vec_owned / extend_owned /
// into_iter_owned)
// --------------------------------------------------------------------------

#[test]
fn owned_api_needs_no_clone() {
    // extend_owned + retain_owned
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.extend_owned([NoClone(1), NoClone(2), NoClone(3), NoClone(4)]);
    c.retain_owned(|x| x.0 % 2 == 1);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3]);

    // retain_mut_owned
    c.retain_mut_owned(|x| {
        x.0 += 10;
        x.0 != 12
    });
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![11, 13]);

    // into_iter_owned (no Clone, zero-copy out of the inline state)
    let vals: Vec<i32> = c.into_iter_owned().map(|x| x.0).collect();
    assert_eq!(vals, vec![11, 13]);

    // into_vec_owned
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.extend_owned([NoClone(7), NoClone(8)]);
    let v = c.into_vec_owned();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0].0, 7);
    assert_eq!(v[1].0, 8);
}

#[test]
#[should_panic(expected = "into_iter_owned on a borrowed CovariantCow")]
fn into_iter_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.into_iter_owned();
}

#[test]
#[should_panic(expected = "retain_mut_owned on a borrowed CovariantCow")]
fn retain_mut_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.retain_mut_owned(|_| true);
}

// --------------------------------------------------------------------------
// Dedicated CowIntoIter
// --------------------------------------------------------------------------

#[test]
fn cow_into_iter_owned_zero_copy() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2, 3, 4, 5]); // len 5 > N -> heap
    assert!(c.is_heap());
    let mut it = c.into_iter_owned();
    assert_eq!(it.as_slice(), &[1, 2, 3, 4, 5]);
    it.as_mut_slice()[0] = 10;
    assert_eq!(it.next(), Some(10));
    assert_eq!(it.next_back(), Some(5));
    assert_eq!(it.as_slice(), &[2, 3, 4]);
    assert_eq!(it.collect::<Vec<_>>(), vec![2, 3, 4]);
}

#[test]
fn into_iter_borrowed_materializes() {
    let data = [1, 2, 3];
    let c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let v: Vec<i32> = c.into_iter().collect();
    assert_eq!(v, vec![1, 2, 3]);
    assert_eq!(data, [1, 2, 3]); // borrower untouched
}

/// Inline twin of the heap `CowIntoIter` coverage, with drop glue: with
/// `len <= N` the elements live *inside* the iterator, so `next`/`next_back`
/// derive a slot pointer into the iterator's own allocation and only then
/// write the cursor field. Sound — borrow stacks are per-location and the
/// cursor is byte-disjoint from the container — but inline-only, so this pins
/// the inline `next`/`next_back`/`drop_remaining` paths under Miri.
#[test]
fn cow_into_iter_inline_non_zst_both_ends_and_drop_remaining() {
    let ctr = Arc::new(AtomicUsize::new(0));
    {
        let mut c = CovariantCow::<CowTracked, 4>::new();
        for i in 0..3 {
            c.push_owned(CowTracked {
                ctr: Arc::clone(&ctr),
                val: i,
            });
        }
        assert!(!c.is_heap());
        let mut it = c.into_iter_owned();
        assert_eq!(it.next().map(|t| t.val), Some(0));
        assert_eq!(it.next_back().map(|t| t.val), Some(2));
        // The yielded pair was dropped by the caller.
        assert_eq!(ctr.load(Ordering::SeqCst), 2);
        // One element is left: dropping the iterator runs `drop_remaining`
        // over the inline storage.
    }
    assert_eq!(ctr.load(Ordering::SeqCst), 3);
}

#[test]
fn cow_into_iter_zst() {
    let mut c = CovariantCow::<(), 4>::new();
    c.extend_owned(core::iter::repeat(()).take(100));
    let it = c.into_iter_owned();
    assert_eq!(it.len(), 100);
    assert_eq!(it.count(), 100);
}

// --------------------------------------------------------------------------
// truncate / clear / resize-shrink destructor-panic safety
// --------------------------------------------------------------------------

/// Element that increments its *own* slot in a shared table on drop (before an
/// optional panic), so a test can detect any element being dropped twice.
struct PerElemDropCow {
    idx: usize,
    counts: Arc<Vec<AtomicUsize>>,
    boom: bool,
}

impl Drop for PerElemDropCow {
    fn drop(&mut self) {
        self.counts[self.idx].fetch_add(1, Ordering::SeqCst);
        assert!(!self.boom, "intentional drop panic");
    }
}

// Only needed to satisfy `CovariantCow::resize`'s `T: Clone` bound; the shrink
// path under test never actually clones an element.
impl Clone for PerElemDropCow {
    fn clone(&self) -> Self {
        PerElemDropCow {
            idx: self.idx,
            counts: Arc::clone(&self.counts),
            boom: self.boom,
        }
    }
}

fn cow_per_elem(
    len: usize,
    boom_at: usize,
) -> (
    CovariantCow<'static, PerElemDropCow, 2>,
    Arc<Vec<AtomicUsize>>,
) {
    let counts = Arc::new((0..len).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let mut c = CovariantCow::new();
    for i in 0..len {
        c.push_owned(PerElemDropCow {
            idx: i,
            counts: Arc::clone(&counts),
            boom: i == boom_at,
        });
    }
    (c, counts)
}

fn no_double_drop(seen: &[usize]) {
    assert!(seen.iter().all(|&n| n <= 1), "double drop: {seen:?}");
}

#[test]
fn cow_truncate_destructor_panic_leaks_not_double_drops() {
    let (mut cow, counts) = cow_per_elem(8, 5);
    assert!(cow.is_heap());
    let r = catch_unwind(AssertUnwindSafe(|| cow.truncate(2)));
    assert!(r.is_err());
    assert_eq!(cow.len(), 2); // published before the destructors ran
    drop(cow);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    no_double_drop(&seen);
    assert!(seen[..2].iter().all(|&n| n == 1), "{seen:?}"); // kept prefix dropped once
}

#[test]
fn cow_clear_destructor_panic_leaks_not_double_drops() {
    let (mut cow, counts) = cow_per_elem(8, 5);
    assert!(cow.is_heap());
    let r = catch_unwind(AssertUnwindSafe(|| cow.clear()));
    assert!(r.is_err());
    assert_eq!(cow.len(), 0);
    drop(cow);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    no_double_drop(&seen);
}

#[test]
fn cow_resize_shrink_destructor_panic_leaks_not_double_drops() {
    let (mut cow, counts) = cow_per_elem(8, 5);
    assert!(cow.is_heap());
    // A separate table for the `value` argument, which `resize` drops during
    // unwinding (it is unused on the shrink path).
    let vcounts = Arc::new(vec![AtomicUsize::new(0)]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        cow.resize(
            2,
            PerElemDropCow {
                idx: 0,
                counts: Arc::clone(&vcounts),
                boom: false,
            },
        )
    }));
    assert!(r.is_err());
    assert_eq!(vcounts[0].load(Ordering::SeqCst), 1); // unused `value` dropped once
    assert_eq!(cow.len(), 2);
    drop(cow);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    no_double_drop(&seen);
    assert!(seen[..2].iter().all(|&n| n == 1), "{seen:?}");
}

// --------------------------------------------------------------------------
// drain / shrink_to_inline / set_len / forwarding impls
// --------------------------------------------------------------------------

#[test]
fn cow_drain_yields_and_rolls_back() {
    let mut c: CovariantCow<i32, 2> = (0..6).collect();
    assert!(c.is_heap());
    let drained: Vec<i32> = c.drain(1..4).collect();
    assert_eq!(drained, vec![1, 2, 3]);
    assert_eq!(&*c, &[0, 4, 5]);
    assert_eq!(c.len(), 3);

    // Partially consuming a drain still drops the un-yielded drained elements
    // and shifts the tail back.
    let mut it = c.drain(1..2);
    assert_eq!(it.next(), Some(4));
    drop(it);
    assert_eq!(&*c, &[0, 5]);
}

#[test]
fn cow_drain_borrowed_materializes() {
    let data = [10, 20, 30, 40];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    assert!(c.is_borrowed());
    let drained: Vec<i32> = c.drain(0..2).collect();
    assert_eq!(drained, vec![10, 20]);
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[30, 40]);
    assert_eq!(data, [10, 20, 30, 40]); // source untouched
}

#[test]
fn cow_drain_drop_accounting() {
    let (mut cow, counts) = cow_per_elem(8, usize::MAX);
    assert!(cow.is_heap());
    // Yield one element (dropped when the local goes out of scope), then drop
    // the drain; every drained element must be dropped exactly once, and the
    // surviving tail must remain.
    {
        let mut it = cow.drain(2..5);
        let first = it.next().unwrap();
        assert_eq!(first.idx, 2);
    }
    assert_eq!(cow.len(), 5);
    drop(cow);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    no_double_drop(&seen);
    // Every element of the original 8 is dropped exactly once: 3 drained
    // (2,3,4) plus the 5 survivors.
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
}

/// A panic from a not-yet-yielded element's destructor *inside* `CowDrain`'s own
/// drop loop must still roll the cow back during unwinding, leaking the
/// elements whose destructors have not run rather than double-dropping them.
///
/// The element owns no allocation (its counter lives in a function-local
/// `static`), so the deliberate leak-on-panic does not trip Miri's leak checker.
#[test]
fn cow_drain_drop_loop_destructor_panic_rolls_back() {
    static COUNTS: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];

    struct El {
        idx: usize,
        boom: bool,
    }
    impl Clone for El {
        fn clone(&self) -> Self {
            El {
                idx: self.idx,
                boom: self.boom,
            }
        }
    }
    impl Drop for El {
        fn drop(&mut self) {
            COUNTS[self.idx].fetch_add(1, Ordering::SeqCst);
            assert!(!self.boom, "intentional drop panic");
        }
    }

    let mut cow: CovariantCow<El, 2> = CovariantCow::new();
    for i in 0..8 {
        cow.push_owned(El {
            idx: i,
            boom: i == 3,
        }); // element 3 panics on drop
    }
    assert!(cow.is_heap());
    let r = catch_unwind(AssertUnwindSafe(|| {
        // Consume nothing: dropping `d` runs the loop over 2, 3 (panic!), 4, 5.
        let _d = cow.drain(2..6);
    }));
    assert!(r.is_err());
    // The rollback ran during unwinding: tail [6, 7] shifted behind [0, 1].
    assert_eq!(cow.len(), 4);
    assert!(cow.iter().map(|t| t.idx).eq([0, 1, 6, 7]));
    drop(cow);
    let seen: Vec<usize> = COUNTS[..8]
        .iter()
        .map(|c| c.load(Ordering::SeqCst))
        .collect();
    no_double_drop(&seen);
    assert_eq!(&seen[..2], &[1, 1]); // kept prefix dropped once
    assert_eq!(seen[2], 1); // dropped before the panic
    assert_eq!(seen[3], 1); // the panicking element counts itself first
    assert_eq!(seen[4], 0); // never reached -> leaked (no double drop)
    assert_eq!(seen[5], 0); // never reached -> leaked (no double drop)
    assert_eq!(seen[6], 1); // shifted tail dropped once
    assert_eq!(seen[7], 1); // shifted tail dropped once
}

#[test]
fn cow_drain_empty_range_does_not_take() {
    let (mut cow, counts) = cow_per_elem(8, usize::MAX);
    {
        let it = cow.drain(3..3);
        assert_eq!(it.len(), 0);
    }
    // Container was never moved out; still heap-backed with the full length.
    assert!(cow.is_heap());
    assert_eq!(cow.len(), 8);
    drop(cow);
    let seen: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
}

#[test]
fn cow_shrink_to_inline() {
    let mut c: CovariantCow<i32, 4> = (0..10).collect();
    assert!(c.is_heap());
    c.truncate(3); // still heap-backed
    assert!(c.is_heap());
    c.shrink_to_inline();
    assert!(!c.is_heap());
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[0, 1, 2]);

    // No-op when the length does not fit inline.
    let mut big: CovariantCow<i32, 2> = (0..10).collect();
    big.shrink_to_inline();
    assert!(big.is_heap());
    assert_eq!(big.len(), 10);
}

#[test]
fn cow_shrink_to_inline_noop_for_borrowed_and_inline() {
    let data = [1, 2, 3];
    let mut b = CovariantCow::<i32, 4>::from_borrowed(&data);
    b.shrink_to_inline();
    assert!(b.is_borrowed()); // untouched, still zero-copy

    let mut inl = CovariantCow::<i32, 4>::new();
    inl.extend_owned([1, 2, 3]);
    assert!(!inl.is_heap());
    inl.shrink_to_inline();
    assert!(!inl.is_heap());
    assert_eq!(&*inl, &[1, 2, 3]);
}

#[test]
fn cow_set_len_grow_and_shrink() {
    let mut c: CovariantCow<i32, 4> = CovariantCow::with_capacity(8);
    c.extend_owned([1, 2, 3]);
    // Grow: the caller has initialized the new slots.
    unsafe {
        c.as_mut_ptr().add(3).write(4);
        c.set_len(4);
    }
    assert_eq!(&*c, &[1, 2, 3, 4]);
    // Shrink: the removed slots are `Copy`, so nothing needs dropping.
    unsafe { c.set_len(2) };
    assert_eq!(&*c, &[1, 2]);
    assert_eq!(c.capacity(), 8); // capacity is preserved
}

#[test]
#[should_panic(expected = "set_len on a borrowed CovariantCow")]
fn cow_set_len_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    unsafe { c.set_len(2) };
}

#[test]
fn cow_forwarding_impls() {
    use core::borrow::{Borrow, BorrowMut};

    let c: CovariantCow<i32, 4> = (0..3).collect();

    // PartialEq against slice-like RHS types.
    assert_eq!(c, vec![0, 1, 2]);
    assert_eq!(c, [0, 1, 2]);
    assert_eq!(c, &[0, 1, 2][..]);
    assert_ne!(c, vec![0, 1]);

    // AsRef / AsMut / Borrow / BorrowMut.
    let s: &[i32] = c.as_ref();
    assert_eq!(s, &[0, 1, 2]);
    let s: &[i32] = Borrow::borrow(&c);
    assert_eq!(s, &[0, 1, 2]);

    let mut m: CovariantCow<i32, 4> = (0..3).collect();
    let s: &mut [i32] = m.as_mut();
    s[0] = 9;
    assert_eq!(&*m, &[9, 1, 2]);
    let s: &mut [i32] = BorrowMut::borrow_mut(&mut m);
    s[1] = 8;
    assert_eq!(&*m, &[9, 8, 2]);

    // AsMut on a borrowed cow copy-on-writes.
    let data = [1, 2, 3];
    let mut b = CovariantCow::<i32, 4>::from_borrowed(&data);
    b.as_mut()[0] = 99;
    assert!(!b.is_borrowed());
    assert_eq!(&*b, &[99, 2, 3]);
    assert_eq!(data, [1, 2, 3]);

    // Extend / FromIterator over `&T` (T: Copy).
    let mut e: CovariantCow<i32, 4> = CovariantCow::new();
    e.extend([1, 2, 3].iter());
    assert_eq!(&*e, &[1, 2, 3]);
    let f: CovariantCow<i32, 4> = [4, 5, 6].iter().collect();
    assert_eq!(&*f, &[4, 5, 6]);
}

#[test]
fn cow_n_zero() {
    let mut c = CovariantCow::<i32, 0>::new();
    assert_eq!(c.capacity(), 0);
    c.push_owned(1); // must spill to the heap
    assert!(c.is_heap());
    assert_eq!(&*c, &[1]);
    c.push_owned(2);
    assert_eq!(&*c, &[1, 2]);
    assert_eq!(c.pop_owned(), Some(2));

    // A borrowed N = 0 cow materializes on write.
    let data = [7, 8, 9];
    let mut b = CovariantCow::<i32, 0>::from_borrowed(&data);
    assert!(b.is_borrowed());
    assert_eq!(&*b, &[7, 8, 9]);
    b.to_mut()[0] = 70;
    assert!(b.is_heap());
    assert_eq!(&*b, &[70, 8, 9]);
    assert_eq!(data, [7, 8, 9]);
}

#[test]
fn cow_borrowed_insert_remove_swap_remove() {
    let data = [1, 2, 3, 4];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.insert(1, 99);
    assert!(!c.is_borrowed()); // materialized
    assert_eq!(&*c, &[1, 99, 2, 3, 4]);
    assert_eq!(c.remove(1), 99);
    assert_eq!(&*c, &[1, 2, 3, 4]);
    assert_eq!(c.swap_remove(0), 1); // last (4) takes its place
    assert_eq!(&*c, &[4, 2, 3]);
    assert_eq!(data, [1, 2, 3, 4]); // borrower untouched
}

#[test]
fn cow_borrowed_retain_closure_panic() {
    let data = [1, 2, 3, 4, 5, 6];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.retain(|x| {
            if *x == 4 {
                panic!("intentional retain panic");
            }
            x % 2 == 0
        })
    }));
    assert!(r.is_err());
    // The cow materialized before the scan and must remain usable.
    assert!(!c.is_borrowed());
    // The guard leaves a consistent (possibly partially-shifted) slice.
    let len = c.len();
    assert!(len <= 6);
    // Dropping must not double-free.
    drop(c);
    assert_eq!(data, [1, 2, 3, 4, 5, 6]);
}

// --------------------------------------------------------------------------
// New _owned mutators / from_elem / resize_with / From<[T; M]> / drain
// --------------------------------------------------------------------------

#[test]
fn cow_owned_mutators_need_no_clone() {
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.extend_owned([NoClone(1), NoClone(2), NoClone(3)]);
    c.insert_owned(1, NoClone(9));
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 9, 2, 3]);
    assert_eq!(c.remove_owned(1).0, 9);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(c.swap_remove_owned(0).0, 1); // last (3) takes its place
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 2]);

    // drain_owned
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.extend_owned([NoClone(1), NoClone(2), NoClone(3), NoClone(4)]);
    let drained: Vec<i32> = c.drain_owned(1..3).map(|x| x.0).collect();
    assert_eq!(drained, vec![2, 3]);
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 4]);

    // resize_with_owned
    let mut c = CovariantCow::<NoClone, 4>::new();
    c.extend_owned([NoClone(1)]);
    let mut k = 0;
    c.resize_with_owned(3, || {
        k += 1;
        NoClone(k)
    });
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 1, 2]);
    c.resize_with_owned(1, || unreachable!()); // shrink never calls the closure
    assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1]);
}

#[test]
#[should_panic(expected = "insert_owned on a borrowed CovariantCow")]
fn insert_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.insert_owned(0, 9);
}

#[test]
#[should_panic(expected = "remove_owned on a borrowed CovariantCow")]
fn remove_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.remove_owned(0);
}

#[test]
#[should_panic(expected = "swap_remove_owned on a borrowed CovariantCow")]
fn swap_remove_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.swap_remove_owned(0);
}

#[test]
#[should_panic(expected = "drain_owned on a borrowed CovariantCow")]
fn drain_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let _ = c.drain_owned(0..1);
}

#[test]
#[should_panic(expected = "resize_with_owned on a borrowed CovariantCow")]
fn resize_with_owned_on_borrowed_panics() {
    let data = [1, 2, 3];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    c.resize_with_owned(5, || 0);
}

#[test]
fn cow_drain_empty_range_on_borrowed_stays_borrowed() {
    let data = [1, 2, 3, 4];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    {
        let it = c.drain(0..0);
        assert_eq!(it.len(), 0);
    }
    assert!(c.is_borrowed(), "empty-range drain must not materialize");
    assert_eq!(&*c, &[1, 2, 3, 4]);
    assert_eq!(data, [1, 2, 3, 4]);

    // A fully-unbounded empty range on an empty borrowed cow.
    let empty: [i32; 0] = [];
    let mut e = CovariantCow::<i32, 4>::from_borrowed(&empty);
    assert_eq!(e.drain(..).len(), 0);
    assert!(e.is_borrowed());
}

#[test]
fn cow_drain_as_mut_slice() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2, 3, 4, 5]);
    let mut d = c.drain(1..4);
    {
        let s = d.as_mut_slice();
        assert_eq!(s, &[2, 3, 4]);
        s[0] = 20;
        s[1] = 30;
    }
    assert_eq!(d.as_slice(), &[20, 30, 4]);
    let collected: Vec<i32> = d.collect();
    assert_eq!(collected, vec![20, 30, 4]);
    assert_eq!(&*c, &[1, 5]);
}

#[test]
fn cow_from_elem_resize_with_from_array() {
    let c = CovariantCow::<i32, 4>::from_elem(5, 3);
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[5, 5, 5]);

    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2]);
    let mut k = 0;
    c.resize_with(5, || {
        k += 1;
        k
    });
    assert_eq!(&*c, &[1, 2, 1, 2, 3]);
    c.resize_with(1, || unreachable!());
    assert_eq!(&*c, &[1]);

    let c: CovariantCow<i32, 4> = [10, 20, 30].into();
    assert_eq!(&*c, &[10, 20, 30]);
    let c: CovariantCow<i32, 2> = [1, 2, 3, 4, 5].into(); // spills
    assert!(c.is_heap());
    assert_eq!(&*c, &[1, 2, 3, 4, 5]);
}

#[test]
#[allow(clippy::uninit_vec)] // the whole point is a bogus ZST length
fn cow_from_vec_zst_overflow_panics() {
    // The cow's length bound is `MAX_LEN = 2^62 - 1` (tighter than the
    // two-state type's `isize::MAX`), because it spends bit 62 on the borrowed
    // tag. A ZST `Vec` length past it must be rejected.
    let mut v: Vec<()> = Vec::new();
    let bogus_len = isize::MAX as usize + 1; // 2^63 on 64-bit, 2^31 on 32-bit: exceeds MAX_LEN either way
                                             // Safety: `()` is a ZST, so the length is representable and no bytes exist.
    unsafe { v.set_len(bogus_len) };
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = CovariantCow::<(), 4>::from(v);
    }));
    assert!(r.is_err());
}

// -------------------------------------------------------------------------- //
// drain: heap-state fast path                                                 //
// -------------------------------------------------------------------------- //
// `make_drain` swaps only `(ptr, cap)` + the tagged length for a heap-backed
// cow (and `CowDrainRollback` swaps it back), instead of `mem::take`-ing the
// whole struct. These pin the observable consequences.

/// Element with a drop counter, for the cow heap-drain accounting test.
struct CowTracked {
    ctr: Arc<AtomicUsize>,
    val: i32,
}

impl Clone for CowTracked {
    fn clone(&self) -> Self {
        CowTracked {
            ctr: Arc::clone(&self.ctr),
            val: self.val,
        }
    }
}

impl Drop for CowTracked {
    fn drop(&mut self) {
        self.ctr.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cow_drain_heap_state_swap_preserves_capacity_and_drop_accounting() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut c = CovariantCow::<CowTracked, 4>::new();
    for i in 0..10 {
        c.push(CowTracked {
            ctr: Arc::clone(&ctr),
            val: i,
        });
    }
    assert!(c.is_heap());
    let cap = c.capacity();

    let mut d = c.drain(2..8);
    let first = d.next().unwrap();
    let last = d.next_back().unwrap();
    assert_eq!((first.val, last.val), (2, 7));
    assert_eq!(d.as_slice().len(), 4);
    drop(d);
    // Elements 3..7 were still held by the drain.
    assert_eq!(ctr.load(Ordering::SeqCst), 4);
    drop(first);
    drop(last);
    assert_eq!(ctr.load(Ordering::SeqCst), 6);

    assert!(c.is_heap());
    assert_eq!(c.capacity(), cap);
    let vals: Vec<i32> = c.iter().map(|t| t.val).collect();
    assert_eq!(vals, vec![0, 1, 8, 9]);

    drop(c);
    assert_eq!(ctr.load(Ordering::SeqCst), 10);
}

// As in `tests/retain_drain.rs`: the placeholder is only observable after
// `mem::forget`, which intentionally leaks the heap block, so Miri's leak
// checker skips this test by design.
#[cfg(not(miri))]
#[test]
fn cow_drain_heap_forget_leaves_a_valid_placeholder() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend(0..10);
    assert!(c.is_heap());

    let d = c.drain(2..5);
    std::mem::forget(d); // leaks the taken cow (documented behaviour)

    assert!(c.is_empty());
    assert!(!c.is_heap());
    assert!(!c.is_borrowed());
    assert_eq!(c.capacity(), 4);
    c.push(9);
    assert_eq!(&*c, &[9]);
}

#[test]
fn cow_drain_heap_prefix_and_suffix_ranges() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend(0..8);
    assert!(c.is_heap());

    assert_eq!(c.drain(..3).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!(c.is_heap());
    assert_eq!(&*c, &[3, 4, 5, 6, 7]);

    assert_eq!(c.drain(3..5).collect::<Vec<_>>(), vec![6, 7]);
    assert!(c.is_heap());
    assert_eq!(&*c, &[3, 4, 5]);

    assert_eq!(c.drain(1..1).count(), 0);
    assert!(c.is_heap());
    assert_eq!(&*c, &[3, 4, 5]);

    assert_eq!(c.drain(..).collect::<Vec<_>>(), vec![3, 4, 5]);
    assert!(c.is_heap());
    assert!(c.is_empty());
}

#[test]
fn cow_drain_owned_heap_state_swap() {
    // `drain_owned` takes the same path without a `T: Clone` bound.
    #[derive(Debug, PartialEq)]
    struct NoClone(i32);

    let mut c = CovariantCow::<NoClone, 2>::new();
    for i in 0..6 {
        c.push_owned(NoClone(i));
    }
    assert!(c.is_heap());
    let cap = c.capacity();

    let got: Vec<NoClone> = c.drain_owned(1..4).collect();
    assert_eq!(got, vec![NoClone(1), NoClone(2), NoClone(3)]);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), cap);
    let rest: Vec<i32> = c.iter().map(|x| x.0).collect();
    assert_eq!(rest, vec![0, 4, 5]);
}

// -------------------------------------------------------------------------- //
// `CowDrain::keep_rest`                                                       //
// -------------------------------------------------------------------------- //

#[test]
fn cow_drain_keep_rest_keeps_unyielded_inline() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2, 3]); // 3 <= N: stays inline and owned
    assert!(!c.is_heap());
    let mut d = c.drain_owned(1..3); // [2, 3]
    assert_eq!(d.next(), Some(2)); // caller-owned
    d.keep_rest();
    drop(d);
    assert_eq!(&*c, &[1, 3]);
    assert!(!c.is_borrowed());
}

/// Inline-state twin of `cow_drain_drop_accounting`, with drop glue: the
/// elements live inside the `CowDrain` struct, so `next` derives a slot
/// pointer into the drain's own allocation and only then bumps `pos`. Sound —
/// borrow stacks are per-location, and the cursor is byte-disjoint from the
/// container — but inline-only, so this pins the inline `next` read and
/// `CowDrain::drop`'s per-element drop loop under Miri, with exact drop
/// accounting.
#[test]
fn cow_drain_inline_non_zst_next_and_drop_loop_accounting() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut c = CovariantCow::<CowTracked, 4>::new();
    for i in 0..3 {
        c.push_owned(CowTracked {
            ctr: Arc::clone(&ctr),
            val: i,
        });
    }
    assert!(!c.is_heap());
    {
        let mut d = c.drain_owned(1..3); // [1, 2]
        assert_eq!(d.next().map(|t| t.val), Some(1)); // caller-owned from here
                                                      // [2] is still unconsumed: `CowDrain::drop`'s loop drops it inline.
    }
    assert_eq!(ctr.load(Ordering::SeqCst), 2); // 1 (caller) + 2 (drop loop)
    assert!(!c.is_heap());
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].val, 0);
    drop(c);
    assert_eq!(ctr.load(Ordering::SeqCst), 3); // every element exactly once
}

/// `drain_bounds` is shared with `CovariantVec::drain`, so the std-exact
/// out-of-range messages apply on the cow side too.
#[test]
#[should_panic(expected = "range start index 5 out of range for slice of length 3")]
fn cow_drain_start_out_of_bounds_panics() {
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2, 3]);
    let _ = c.drain_owned(5..);
}

#[test]
fn cow_drain_keep_rest_heap() {
    let mut c = CovariantCow::<i32, 2>::new();
    for i in 0..6 {
        c.push_owned(i);
    }
    assert!(c.is_heap());
    let cap = c.capacity();
    let mut d = c.drain_owned(1..5); // [1, 2, 3, 4]
    assert_eq!(d.next(), Some(1));
    assert_eq!(d.next_back(), Some(4));
    d.keep_rest(); // unyielded: 2, 3
    drop(d);
    assert!(c.is_heap());
    assert_eq!(&*c, &[0, 2, 3, 5]);
    assert_eq!(c.capacity(), cap);
}

#[test]
fn cow_drain_keep_rest_borrowed() {
    // `drain` on a borrowed cow materializes it; `keep_rest` then behaves
    // exactly like the owned case and leaves the source untouched.
    let data = [1, 2, 3, 4, 5, 6];
    let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
    let mut d = c.drain(1..4);
    assert_eq!(d.next(), Some(2));
    d.keep_rest();
    drop(d);
    assert!(!c.is_borrowed());
    assert_eq!(&*c, &[1, 3, 4, 5, 6]);
    assert_eq!(data, [1, 2, 3, 4, 5, 6]);
}

#[test]
fn cow_drain_keep_rest_nothing_yielded_is_a_full_keep() {
    // `keep_rest` before any yield: `pos == begin` skips the shift and
    // `begin + kept == end` skips the rollback's tail move — both no-op paths
    // must leave the cow exactly as `drain` found it.
    let mut c = CovariantCow::<i32, 4>::new();
    c.extend_owned([1, 2, 3, 4, 5, 6]);
    assert!(c.is_heap());
    let cap = c.capacity();
    let mut d = c.drain_owned(1..4);
    d.keep_rest();
    assert_eq!(d.len(), 0);
    drop(d);
    assert!(c.is_heap());
    assert_eq!(c.capacity(), cap);
    assert_eq!(&*c, &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn cow_drain_keep_rest_drop_accounting() {
    let ctr = Arc::new(AtomicUsize::new(0));
    let mut c: CovariantCow<CowTracked, 4> = (0..6)
        .map(|i| CowTracked {
            ctr: Arc::clone(&ctr),
            val: i,
        })
        .collect();
    let mut d = c.drain_owned(1..4);
    let yielded = d.next().unwrap();
    d.keep_rest();
    drop(d);
    drop(yielded);
    assert_eq!(ctr.load(Ordering::SeqCst), 1);
    assert_eq!(
        c.iter().map(|t| t.val).collect::<Vec<_>>(),
        vec![0, 2, 3, 4, 5]
    );
    drop(c);
    assert_eq!(ctr.load(Ordering::SeqCst), 6);
}

// -------------------------------------------------------------------------- //
// clone_from                                                                  //
// -------------------------------------------------------------------------- //

#[test]
fn cow_clone_from_reuses_destination_capacity() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut dest: CovariantCow<ResizeTracked, 2> = (0..8)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();
    assert!(dest.is_heap());
    let big_cap = dest.capacity();
    let src: CovariantCow<ResizeTracked, 2> = (0..3)
        .map(|i| ResizeTracked::new(9 + i, &clones, &drops))
        .collect();

    clones.store(0, Ordering::SeqCst);
    drops.store(0, Ordering::SeqCst);
    dest.clone_from(&src);
    assert_eq!(dest.capacity(), big_cap);
    assert_eq!(dest.len(), src.len());
    let vals: Vec<i32> = dest.iter().map(|t| t.val).collect();
    assert_eq!(vals, vec![1009, 1010, 1011]); // clones: `val` shifted by 1000
    assert_eq!(drops.load(Ordering::SeqCst), 8);
    assert_eq!(clones.load(Ordering::SeqCst), 3);
}

#[test]
fn cow_clone_from_zst_len_gt_n_reruns_clones() {
    // The cow twin of the `CovariantVec` test: `len > N` is still the inline
    // state for a ZST, so the `size_of::<T>() == 0` reuse branch runs, and the
    // fill deliberately runs `T::clone` once per element (contrast `to_mut`'s
    // count adoption, which skips the clone side effects).
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

    let mut dest = CovariantCow::<Z, 4>::new();
    for _ in 0..10 {
        dest.push(Z); // adopts: no clones, no drops
    }
    let mut src = CovariantCow::<Z, 4>::new();
    for _ in 0..7 {
        src.push(Z);
    }
    assert_eq!(dest.len(), 10);

    CLONES.store(0, Ordering::SeqCst);
    DROPS.store(0, Ordering::SeqCst);
    dest.clone_from(&src);
    assert_eq!(dest.len(), 7);
    assert_eq!(CLONES.load(Ordering::SeqCst), 7);
    assert_eq!(DROPS.load(Ordering::SeqCst), 10);

    drop(dest);
    drop(src);
    assert_eq!(DROPS.load(Ordering::SeqCst), 10 + 7 + 7);
}

#[test]
fn cow_clone_from_borrowed_source_is_zero_copy() {
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let data: Vec<ResizeTracked> = (0..3)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();
    let src = CovariantCow::<ResizeTracked, 4>::from_borrowed(&data);
    let mut dest: CovariantCow<ResizeTracked, 4> = (0..8)
        .map(|i| ResizeTracked::new(i, &clones, &drops))
        .collect();

    dest.clone_from(&src);
    // A borrowed source is adopted by the `clone` fallback (no allocation, no
    // element clones).
    assert!(dest.is_borrowed());
    assert_eq!(dest.as_borrowed().unwrap().as_ptr(), data.as_ptr());
}

#[test]
fn cow_clone_from_panic_mid_fill_leaves_valid_partial_prefix() {
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
    let mut dest: CovariantCow<Boom, 8> = CovariantCow::new();
    let src: CovariantCow<Boom, 8> = (0..5)
        .map(|i| Boom {
            val: i,
            seen: Arc::clone(&seen),
        })
        .collect();

    let r = catch_unwind(AssertUnwindSafe(|| dest.clone_from(&src)));
    assert!(r.is_err());
    assert_eq!(dest.len(), 2);
    assert_eq!(dest[0].val, 0);
    assert_eq!(dest[1].val, 1);
}

/// Iterator whose lower-bound `size_hint` lies with `usize::MAX`.
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
fn cow_extend_with_usize_max_hint_panics_before_allocating() {
    let data = [1i32, 2, 3];
    let mut c = CovariantCow::<i32, 2>::from_borrowed(&data);
    let src = c.as_borrowed().unwrap().as_ptr();
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.extend(MaxHintIter {
            items: vec![4, 5, 6].into_iter(),
        });
    }));
    assert!(r.is_err());
    // Untouched: still the original borrow, same pointer, same contents.
    assert!(c.is_borrowed());
    assert_eq!(c.as_borrowed().unwrap().as_ptr(), src);
    assert_eq!(&*c, &data);
}
