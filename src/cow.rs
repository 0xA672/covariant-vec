//! Optional three-state copy-on-write variant.
//!
//! Enabled with the `cow` feature. Adds a third, zero-copy `Borrowed` state on
//! top of the two-state layout by spending the now-free **bit 62** of the `len`
//! field:
//!
//! ```text
//! len = [ bit63 | bit62 | 62-bit length ]
//!   bit63 bit62  state
//!     0     0    Inline   raw_len ∈ [0, N]
//!     1     0    Heap     raw_len ≤ cap
//!     0     1    Borrowed raw_len = slice length
//!     1     1    illegal (never produced)
//! ```
//!
//! The length upper bound therefore tightens to `2^62 - 1`.
//!
//! ## Auto traits
//!
//! Because the borrowed state stores a `&'a [T]`, `Send` requires
//! `T: Send + Sync` and `Sync` requires `T: Sync` (std-`Cow` parity). This is a
//! **stricter `Send`** than [`CovariantVec`](crate::CovariantVec), whose
//! `Send ⇔ T: Send` (it has no shared-borrow state). The negative direction is
//! pinned with `compile_fail` doctests:
//!
//! ```compile_fail
//! use std::cell::Cell;
//! use covariant_vec::CovariantCow;
//! fn assert_send<T: Send>() {}
//! // `Cell<i32>` is `!Sync`, so the cow is `!Send`.
//! assert_send::<CovariantCow<'static, Cell<i32>, 4>>();
//! ```
//!
//! ```compile_fail
//! use std::cell::Cell;
//! use covariant_vec::CovariantCow;
//! fn assert_sync<T: Sync>() {}
//! assert_sync::<CovariantCow<'static, Cell<i32>, 4>>();
//! ```

use core::alloc::Layout;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut, RangeBounds};
use core::ptr::{self, addr_of, addr_of_mut, NonNull};
use core::slice;

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use alloc::vec::Vec;

#[cfg(feature = "specialization")]
use crate::clone_fill::CloneRepeat;
use crate::clone_fill::{CloneFill, CloneSink};
use crate::{try_heap_layout, BackshiftTarget, TryReserveError, HEAP_TAG};

/// Borrowed-state tag: bit 62.
const BORROWED_TAG: usize = 1 << (usize::BITS - 2);
/// Mask of the length bits.
const LEN_MASK: usize = !(HEAP_TAG | BORROWED_TAG);
/// Maximum representable length (`2^62 - 1` on 64-bit).
const MAX_LEN: usize = LEN_MASK;

/// Overlapping storage body. The `repr(C)` is load-bearing specification, not
/// interop: `repr(Rust)` unions carry no layout guarantees, while `repr(C)`
/// pins every field to offset 0 with the size/alignment of the largest field —
/// the layout the three-state encoding and `tests/layout.rs` assume.
#[repr(C)]
union Data3<'a, T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    heap: (NonNull<u8>, usize),
    /// A shared slice is `Copy`, so it is a legal union field, and it is
    /// covariant in both `'a` and `T`.
    borrowed: &'a [T],
}

/// A three-state (inline / heap / borrowed) copy-on-write vector.
///
/// # Borrowed-aware vs. owned-only methods
///
/// Methods that may encounter the borrowed state carry a `T: Clone` bound,
/// because materializing a borrow requires cloning. Each such method has a
/// `_owned` counterpart that instead *panics* on a borrowed cow, so `T: Clone`
/// is not required when the cow is known to be owned:
///
/// | borrowed-aware (`T: Clone`) | owned-only (no `Clone`) |
/// |---|---|
/// | [`push`](Self::push) | [`push_owned`](Self::push_owned) |
/// | [`pop`](Self::pop) | [`pop_owned`](Self::pop_owned) |
/// | [`reserve`](Self::reserve) / [`reserve_exact`](Self::reserve_exact) | [`reserve_owned`](Self::reserve_owned) / [`reserve_exact_owned`](Self::reserve_exact_owned) |
/// | [`retain`](Self::retain) / [`retain_mut`](Self::retain_mut) | [`retain_owned`](Self::retain_owned) / [`retain_mut_owned`](Self::retain_mut_owned) |
/// | [`insert`](Self::insert) | [`insert_owned`](Self::insert_owned) |
/// | [`remove`](Self::remove) | [`remove_owned`](Self::remove_owned) |
/// | [`swap_remove`](Self::swap_remove) | [`swap_remove_owned`](Self::swap_remove_owned) |
/// | [`drain`](Self::drain) | [`drain_owned`](Self::drain_owned) |
/// | [`append`](Self::append) | [`append_owned`](Self::append_owned) |
/// | [`split_off`](Self::split_off) | — (no `Clone` needed) |
/// | [`splice`](Self::splice) | [`splice_owned`](Self::splice_owned) |
/// | [`extract_if`](Self::extract_if) | [`extract_if_owned`](Self::extract_if_owned) |
/// | [`resize_with`](Self::resize_with) | [`resize_with_owned`](Self::resize_with_owned) |
/// | [`into_vec`](Self::into_vec) | [`into_vec_owned`](Self::into_vec_owned) |
/// | [`Extend`] | [`extend_owned`](Self::extend_owned) |
/// | `IntoIterator` | [`into_iter_owned`](Self::into_iter_owned) |
///
/// [`resize`](Self::resize) mutates in place and has **no** owned-only form:
/// its grow path fills by cloning `value`, so it genuinely requires `T: Clone`.
/// Every other borrowed-aware method above has an owned-only twin, so an owned
/// cow of a `!Clone` type can still mutate in every way that does not clone.
///
/// [`split_off`](Self::split_off) is the exception in the other direction: it
/// needs no `Clone` even on a borrowed cow, because it splits the borrow into
/// two disjoint sub-slices instead of materializing it. [`append`](Self::append)
/// with a borrowed `other` likewise clones that borrow directly into `self`
/// (fused into a single allocation when `self` is borrowed too) instead of
/// materializing `other` first.
///
/// # Panicking vs. copy-on-write mutable access
///
/// Raw or unchecked mutable access cannot copy-on-write: it carries no
/// `T: Clone` bound, and the caller would keep a pointer into storage that a
/// later materialization replaces. Those methods **panic** on a borrowed cow.
/// The safe mutable views and iteration APIs instead copy-on-write through
/// [`to_mut`](Self::to_mut):
///
/// | panics if borrowed | copy-on-write |
/// |---|---|
/// | [`as_mut_slice`](Self::as_mut_slice) | [`DerefMut`] (`&mut cow[..]`, `iter_mut`, indexing) |
/// | [`as_mut_ptr`](Self::as_mut_ptr) | [`AsMut`] |
/// | [`set_len`](Self::set_len) | [`BorrowMut`] |
/// | [`spare_capacity_mut_owned`](Self::spare_capacity_mut_owned) | the borrowed-aware mutation methods above |
///
/// The rule is mechanical: no `T: Clone` bound, no materialization.
///
/// # Which operations materialize a borrow?
///
/// A borrowed cow is narrowed **zero-copy** (stays borrowed; no clone, no
/// drop) by the shrinking operations: [`pop`](Self::pop),
/// [`truncate`](Self::truncate), [`clear`](Self::clear), the shrinking arm
/// of [`resize`](Self::resize), and [`split_off`](Self::split_off) (which
/// splits the borrow into two disjoint sub-slices). It must be
/// **materialized** by operations that need contiguous mutable storage:
/// [`push`](Self::push), [`insert`](Self::insert),
/// [`remove`](Self::remove), [`swap_remove`](Self::swap_remove), a
/// non-empty [`drain`](Self::drain), [`retain`](Self::retain),
/// [`extract_if`](Self::extract_if), a [`splice`](Self::splice) over a
/// non-empty range or with a replacement that may yield, and
/// [`append`](Self::append) when `self` is borrowed.
/// [`append`](Self::append) with a borrowed `other` clones that borrow
/// directly and never materializes `other`.
///
/// The fallible [`try_reserve`](Self::try_reserve) /
/// [`try_reserve_exact`](Self::try_reserve_exact) (and their `_owned` twins)
/// and [`try_to_mut`](Self::try_to_mut) (the fallible twin of `to_mut`)
/// report capacity overflow and allocation failure as
/// [`TryReserveError`](crate::TryReserveError) instead of panicking/aborting.
/// The borrowed-aware `reserve*` and `try_reserve*` alike fuse the
/// copy-on-write into the reservation, so a borrowed cow is materialized and
/// grown with a **single** allocation (the infallible pair aborts on
/// allocation failure, like the rest of the infallible capacity methods).
///
/// # The implied `T: 'a` bound
///
/// The `borrowed` union field stores `&'a [T]`, which makes well-formedness
/// require `T: 'a` on every `CovariantCow<'a, T, N>` — exactly as `&'a [T]`
/// does. [`CovariantVec`](crate::CovariantVec) has no such bound: nothing in
/// its layout ties `T` to `'a` (the lifetime is only a covariance anchor), so
/// `CovariantVec<'static, U, N>` is well-formed for a non-`'static` `U` while
/// `CovariantCow<'static, U, N>` is not. The motivating shape
/// `CovariantCow<'src, Label<'src>, N>` satisfies the bound, but code ported
/// from the two-state type can hit it.
#[repr(C)]
pub struct CovariantCow<'a, T, const N: usize> {
    data: Data3<'a, T, N>,
    len: usize,
}

/// Next capacity given the current one, for the cow's 62-bit length field.
///
/// `from <= N` selects the first-spill capacity `max(N + 1, 2N)` floored to
/// `floor` (the caller passes [`min_non_zero_cap`](crate::min_non_zero_cap), so
/// `N = 0` does not crawl `1 -> 2 -> 4 -> 8`); above that we double. Always
/// clamped to `MAX_LEN`.
#[inline]
const fn grow_to<const N: usize>(from: usize, floor: usize) -> usize {
    let candidate = if from <= N {
        let a = N.saturating_add(1);
        let b = N.saturating_mul(2);
        let base = if a > b { a } else { b };
        if base > floor {
            base
        } else {
            floor
        }
    } else {
        from.saturating_mul(2)
    };
    // Unlike the two-state type, the length must stay within the 62-bit field.
    if candidate > MAX_LEN {
        MAX_LEN
    } else {
        candidate
    }
}

/// The capacity a reservation should grow to: `needed` exactly when `exact`,
/// otherwise the amortized `grow_to` curve raised to at least `needed`.
#[inline]
fn planned_cap<T, const N: usize>(from: usize, needed: usize, exact: bool) -> usize {
    if exact {
        needed
    } else {
        let grown = grow_to::<N>(from, crate::min_non_zero_cap::<T>());
        if grown < needed {
            needed
        } else {
            grown
        }
    }
}

impl<'a, T, const N: usize> CovariantCow<'a, T, N> {
    /// Creates an empty inline cow.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        const {
            assert!(
                N <= MAX_LEN,
                "N too large for CovariantCow (length field is 62 bits)"
            )
        };
        CovariantCow {
            data: Data3 {
                inline: ManuallyDrop::new(MaybeUninit::uninit()),
            },
            len: 0,
        }
    }

    /// Creates a cow that borrows `slice` without copying.
    ///
    /// This is a `const fn`, so a borrowed cow can be built in a `const`/`static`
    /// context (e.g. `const C: CovariantCow<i32, 4> = CovariantCow::from_borrowed(&[1, 2, 3]);`).
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() > 2^62 - 1` (not representable in the length
    /// field); as a `const fn` this is a compile-time error in a `const`
    /// context.
    ///
    /// # Materialization
    ///
    /// Only the element count is checked here. Materializing the borrow —
    /// [`to_mut`](Self::to_mut), [`reserve`](Self::reserve), or any mutating
    /// operation — additionally requires the byte size to fit an allocation:
    /// `slice.len() * size_of::<T>()`, rounded up to `align_of::<T>()`, at
    /// most `isize::MAX`. A larger slice can still be borrowed, read, and
    /// dropped, but the infallible materializers panic with
    /// `"capacity overflow"` (or abort on allocation failure); the fallible
    /// [`try_to_mut`](Self::try_to_mut) and
    /// [`try_reserve`](Self::try_reserve) report it as
    /// [`TryReserveError`](crate::TryReserveError) instead.
    #[must_use]
    pub const fn from_borrowed(slice: &'a [T]) -> Self {
        // `from_borrowed` does not go through `new()`, so repeat the
        // precondition on `N`.
        const {
            assert!(
                N <= MAX_LEN,
                "N too large for CovariantCow (length field is 62 bits)"
            )
        };
        assert!(slice.len() <= MAX_LEN, "slice too long");
        CovariantCow {
            data: Data3 { borrowed: slice },
            len: BORROWED_TAG | slice.len(),
        }
    }

    /// Creates an empty cow with at least `capacity` capacity.
    ///
    /// `capacity <= N` stays inline; a larger request goes straight to the heap
    /// state. For a zero-sized element nothing is allocated (ZSTs never spill):
    /// [`capacity`](Self::capacity) reports `MAX_LEN` regardless of the
    /// requested value. The request is still bounds-checked, though — a capacity
    /// above `2^62 - 1` panics with `"capacity overflow"` rather than being
    /// silently accepted, mirroring [`reserve_owned`](Self::reserve_owned).
    #[must_use]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut out = Self::new();
        if size_of::<T>() == 0 {
            // No allocation, but the 62-bit length field still bounds it.
            assert!(capacity <= MAX_LEN, "capacity overflow");
        } else if capacity > N {
            out.alloc_heap(capacity);
        }
        out
    }

    #[inline]
    #[must_use]
    pub const fn is_heap(&self) -> bool {
        self.len & HEAP_TAG != 0
    }

    #[inline]
    #[must_use]
    pub const fn is_borrowed(&self) -> bool {
        self.len & BORROWED_TAG != 0
    }

    #[inline]
    const fn raw_len(&self) -> usize {
        self.len & LEN_MASK
    }

    #[inline]
    const fn heap_cap(&self) -> usize {
        debug_assert!(self.is_heap());
        // Safety: heap state.
        unsafe { self.data.heap.1 }
    }

    /// Raw `*mut T` to the first element slot, for the **owned** states only.
    ///
    /// Panics in the borrowed state: we deliberately never derive a `*mut T`
    /// from the shared `&'a [T]` (that would be aliasing UB). Callers must
    /// copy-on-write via [`Self::to_mut`] first. There is intentionally no
    /// borrowed branch here, so a future caller cannot accidentally reach one.
    #[track_caller]
    fn base_ptr(&mut self) -> *mut T {
        assert!(
            !self.is_borrowed(),
            "mutable element access on a borrowed CovariantCow"
        );
        // Safety: just checked.
        unsafe { self.base_ptr_owned() }
    }

    /// Unchecked twin of [`Self::base_ptr`] for loops whose container is
    /// statically known to be owned.
    ///
    /// The `CowIntoIter`/`CowDrain` constructors eagerly materialize a
    /// borrowed cow, so their per-element loops would otherwise re-run the
    /// same state test (and panic branch) for every element. The check is
    /// kept as a `debug_assert!`.
    ///
    /// # Safety
    ///
    /// The cow must not be in the borrowed state. Deriving a `*mut T` from
    /// the shared `&'a [T]` would be aliasing UB.
    #[inline]
    unsafe fn base_ptr_owned(&mut self) -> *mut T {
        debug_assert!(
            !self.is_borrowed(),
            "mutable element access on a borrowed CovariantCow"
        );
        if self.is_heap() {
            // Safety: heap state.
            unsafe { self.data.heap.0.cast::<T>().as_ptr() }
        } else {
            addr_of_mut!(self.data.inline).cast::<T>()
        }
    }

    fn base_ptr_const(&self) -> *const T {
        if self.is_borrowed() {
            // Safety: reading the borrowed variant.
            unsafe { self.data.borrowed.as_ptr() }
        } else if self.is_heap() {
            // Safety: heap state.
            unsafe { self.data.heap.0.cast::<T>().as_ptr() as *const T }
        } else {
            addr_of!(self.data.inline).cast::<T>()
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw_len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw_len() == 0
    }

    /// The current capacity.
    ///
    /// In the **borrowed** state this returns the borrowed slice's length: the
    /// borrowed data *is* the storage, and it is already exactly full. In
    /// the inline state it is `N`; in the heap state the heap capacity; for a
    /// ZST, `MAX_LEN` (`2^62 - 1`).
    ///
    /// Note the cross-type asymmetry: `CovariantVec<_, ZST, _>` reports
    /// `isize::MAX` for a ZST, because the two-state type has one more usable
    /// length bit (no borrowed tag). Its `From<Vec<ZST>>` overflow threshold is
    /// correspondingly higher.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        if self.is_borrowed() {
            self.raw_len()
        } else if size_of::<T>() == 0 {
            MAX_LEN
        } else if self.is_heap() {
            self.heap_cap()
        } else {
            N
        }
    }

    /// The elements as a slice. Zero-copy in the borrowed state.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_borrowed() {
            // Safety: reconstruct the `&'a [T]` that is still stored there.
            unsafe { self.data.borrowed }
        } else {
            let len = self.raw_len();
            // Safety: non-null, aligned, `len` valid elements.
            unsafe { slice::from_raw_parts(self.base_ptr_const(), len) }
        }
    }

    /// If borrowed, hands out the original long-lived borrow.
    ///
    /// This is sound because [`Self::to_mut`] copies rather than mutating in
    /// place, so the borrower's slice is never altered.
    #[must_use]
    pub fn as_borrowed(&self) -> Option<&'a [T]> {
        if self.is_borrowed() {
            // Safety: reading the borrowed variant.
            Some(unsafe { self.data.borrowed })
        } else {
            None
        }
    }

    /// The elements as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state: handing out `&mut [T]`
    /// over the still-live shared `&'a [T]` would be aliasing UB. Use
    /// [`Self::to_mut`] (or `DerefMut`) to copy-on-write first.
    #[inline]
    #[track_caller]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            !self.is_borrowed(),
            "as_mut_slice on a borrowed Cow; call to_mut first"
        );
        let len = self.raw_len();
        // Safety: `&mut self` guarantees no aliases.
        unsafe { slice::from_raw_parts_mut(self.base_ptr(), len) }
    }

    /// A raw pointer to the first element slot (zero-copy in the borrowed
    /// state).
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.base_ptr_const()
    }

    /// A mutable raw pointer to the first element slot.
    ///
    /// # Panics
    ///
    /// Panics in the borrowed state (same reason as [`Self::as_mut_slice`]).
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.base_ptr()
    }

    /// Resizes to `new_len`, cloning `value` to fill new slots. Requires
    /// `T: Clone` because a borrowed cow must be materialized first.
    ///
    /// Growing follows `Vec::resize` parity: `value` is cloned into the first
    /// `new_len - len - 1` new slots and the original is **moved** into the last
    /// new slot, so `T::clone` runs `new_len - len - 1` times and the total
    /// number of destructors is exactly `new_len`. A borrowed cow is
    /// materialized with capacity for the whole fill in a single allocation.
    ///
    /// Shrinking a **borrowed** cow does not materialize it: the cow stays
    /// borrowed and simply narrows the slice it points at, so no element is
    /// cloned or dropped (the discarded tail is still owned by the borrower).
    /// This mirrors [`Self::clear`], which likewise avoids cloning a borrowed
    /// cow.
    #[track_caller]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        let len = self.raw_len();
        if new_len > len {
            let additional = new_len - len;
            if self.is_borrowed() {
                // Fuse materialization and the capacity request: one allocation.
                self.materialize_with_min_capacity(additional);
            } else {
                self.reserve_owned(additional);
            }
            if size_of::<T>() == 0 {
                // ZSTs: the count *is* the ownership (ZST protocol site 1, as
                // in `push_owned`). Adopt the `additional - 1` clones and then
                // the original into the count: exactly `additional - 1` clones
                // run, none of them drops, and the cow ends up owning
                // `new_len` values. The same tag-hygiene guard as `push_owned`:
                // an untagged length write on a (buggy) heap-backed ZST cow
                // would silently clear `HEAP_TAG` and leak the block.
                assert!(!self.is_heap(), "ZST cows are never heap-backed");
                #[cfg(feature = "specialization")]
                {
                    // Feature parity with `CovariantVec::resize`: go through
                    // `T::tail_fill`, which uses the `T: Copy` bulk path, so a
                    // `Copy` ZST does not run the element-wise `Clone` loop.
                    // The default (non-`Copy`) body has the same
                    // clone-before-publish panic-safety order as the loop below.
                    let l = self.raw_len();
                    // Safety: owned ZST state, never heap-backed; `l + additional
                    // <= MAX_LEN` was enforced by the reserve above, and slots
                    // `[l, l + additional)` are conceptually uninitialized.
                    unsafe { T::tail_fill(value, self, l, additional) };
                }
                #[cfg(not(feature = "specialization"))]
                {
                    for _ in 0..(additional - 1) {
                        assert!(self.raw_len() < MAX_LEN, "capacity overflow");
                        // Clone *before* publishing: a panic inside `T::clone`
                        // must not leave the count claiming a value that was
                        // never created. The in-flight `value` is dropped by the
                        // unwind, so the created values stay exactly balanced.
                        let cloned = value.clone();
                        self.len = self.raw_len() + 1;
                        mem::forget(cloned);
                    }
                    assert!(self.raw_len() < MAX_LEN, "capacity overflow");
                    self.len = self.raw_len() + 1;
                    mem::forget(value);
                }
            } else {
                #[cfg(feature = "specialization")]
                {
                    // Bulk path (see `CovariantVec::resize`): clones into the
                    // first `additional - 1` slots and moves the original into
                    // the last; for `T: Copy` it is one length publication plus
                    // the O(log additional) doubling copies of `spec::bulk` —
                    // sound because no user code runs in between and, for a
                    // type with no destructor, a moved original is
                    // unobservable.
                    let l = self.raw_len();
                    // Safety: owned state with capacity for `additional` more
                    // elements, so slots `[l, l + additional)` are in bounds
                    // and uninitialized; `additional >= 1` because
                    // `new_len > len`.
                    unsafe { T::tail_fill(value, self, l, additional) };
                }
                #[cfg(not(feature = "specialization"))]
                {
                    // Clone into the first `additional - 1` slots and move
                    // `value` into the last one, so `T::clone` runs
                    // `additional - 1` times and no extra destructor is run for
                    // the original.
                    if self.is_heap() {
                        // Hoisted heap loop: the block is independent of
                        // `self`, so `base` survives the length publications;
                        // `reserve_owned` already sized it for all
                        // `additional` slots.
                        let base = self.base_ptr();
                        let mut l = self.raw_len();
                        for _ in 0..(additional - 1) {
                            let v = value.clone();
                            // Safety: slot `l` is in bounds and uninitialized
                            // (capacity reserved above); `base` stays valid
                            // across the publication below.
                            unsafe { base.add(l).write(v) };
                            l += 1;
                            // Published before the next `clone` (user code,
                            // may panic), so a panic leaves the written prefix
                            // counted.
                            self.len = HEAP_TAG | l;
                        }
                        // Safety: as above, for the final uninitialized slot;
                        // moving `value` in consumes it without running its
                        // destructor here.
                        unsafe { base.add(l).write(value) };
                        self.len = HEAP_TAG | (l + 1);
                    } else {
                        for _ in 0..(additional - 1) {
                            let v = value.clone();
                            let l = self.raw_len();
                            // Safety: owned state with capacity for
                            // `additional` more elements, so slot `[l]` is in
                            // bounds and uninitialized. The pointer is
                            // re-derived each iteration because a write to
                            // `self.len` is a use of `self` that invalidates
                            // hoisted raw derivations under Stacked Borrows.
                            unsafe { self.base_ptr().add(l).write(v) };
                            self.len = (self.len & HEAP_TAG) | (l + 1);
                        }
                        // Safety: as above, for the final uninitialized slot;
                        // moving `value` in consumes it without running its
                        // destructor here.
                        let l = self.raw_len();
                        unsafe { self.base_ptr().add(l).write(value) };
                        self.len = (self.len & HEAP_TAG) | (l + 1);
                    }
                }
            }
        } else if new_len < len {
            if self.is_borrowed() {
                // Stay borrowed: hand out the truncated prefix of the original
                // slice (still a valid `&'a [T]`). No element is cloned or
                // dropped — the borrower keeps owning the discarded tail.
                let b: &'a [T] = unsafe { self.data.borrowed };
                self.data.borrowed = &b[..new_len];
                self.len = BORROWED_TAG | new_len;
            } else {
                // Publish the new length *before* running destructors (std
                // parity), so a panicking destructor leaks the un-dropped tail
                // rather than leaving already-dropped slots counted.
                self.len = (self.len & HEAP_TAG) | new_len;
                if mem::needs_drop::<T>() {
                    // Derive the pointer *after* the `len` write (Stacked
                    // Borrows). Safety: `[new_len, len)` were initialized and
                    // are no longer claimed by the container.
                    let p = unsafe { self.base_ptr().add(new_len) };
                    unsafe {
                        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len - new_len));
                    }
                }
            }
        }
    }

    /// Reserves room for at least `additional` more elements, materializing a
    /// borrowed cow first (hence the `T: Clone` bound; use
    /// [`Self::reserve_owned`] when the cow is known to be owned).
    ///
    /// The copy-on-write is **fused** with the reservation: a borrowed cow is
    /// cloned into a buffer sized for the request directly, so a borrowed
    /// `reserve` performs one allocation, not a materialize-at-`len` followed
    /// by a realloc (same capacity trajectory as
    /// [`Self::try_reserve`](Self::try_reserve)).
    ///
    /// Asking for zero additional capacity is a no-op and leaves a borrowed cow
    /// borrowed (parity with [`Self::try_reserve`]).
    #[track_caller]
    pub fn reserve(&mut self, additional: usize)
    where
        T: Clone,
    {
        if additional == 0 {
            return;
        }
        if self.is_borrowed() {
            self.materialize_with_min_capacity(additional);
            return;
        }
        self.reserve_owned(additional);
    }

    /// Reserves room for exactly `additional` more elements, materializing a
    /// borrowed cow first (hence the `T: Clone` bound).
    ///
    /// Like [`Self::reserve`], the copy-on-write is fused with the reservation
    /// (one allocation for a borrowed cow); the requested capacity is `len +
    /// additional` verbatim, matching [`Self::reserve_exact_owned`] and
    /// [`Self::try_reserve_exact`].
    ///
    /// Asking for zero additional capacity is a no-op and leaves a borrowed cow
    /// borrowed.
    #[track_caller]
    pub fn reserve_exact(&mut self, additional: usize)
    where
        T: Clone,
    {
        if additional == 0 {
            return;
        }
        if self.is_borrowed() {
            self.materialize_with_exact_capacity(additional);
            return;
        }
        self.reserve_exact_owned(additional);
    }

    /// Tries to reserve room for at least `additional` more elements, reporting
    /// failure as an error rather than panicking or aborting.
    ///
    /// A **borrowed** cow is materialized (copy-on-write) as part of the
    /// reservation, fused into a single fallible allocation (the same fused
    /// shape as [`Self::reserve`], which aborts instead of returning `Err` on
    /// allocation failure). Asking for zero additional capacity on a borrowed
    /// cow is a no-op and leaves it borrowed.
    ///
    /// # Errors
    ///
    /// Same as [`CovariantVec::try_reserve`](crate::CovariantVec::try_reserve).
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>
    where
        T: Clone,
    {
        if !self.is_borrowed() {
            return self.try_reserve_owned(additional);
        }
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if additional == 0 {
            // Nothing to reserve; the borrowed slice is already exactly full.
            return Ok(());
        }
        let new_cap = if needed <= N {
            // The borrowed prefix already fits inline; materialize there rather
            // than paying for the amortized first-spill. `reserve`'s fused path
            // (`materialize_with_min_capacity`) makes the same choice.
            N
        } else {
            planned_cap::<T, N>(len, needed, false)
        };
        if new_cap > MAX_LEN {
            return Err(TryReserveError::capacity_overflow());
        }
        self.try_materialize(new_cap)
    }

    /// Tries to reserve room for exactly `additional` more elements, reporting
    /// failure as an error rather than panicking or aborting.
    ///
    /// Like [`Self::try_reserve`], but requests the exact capacity.
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_reserve`].
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError>
    where
        T: Clone,
    {
        if !self.is_borrowed() {
            return self.try_reserve_exact_owned(additional);
        }
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if additional == 0 {
            return Ok(());
        }
        let new_cap = planned_cap::<T, N>(len, needed, true);
        if new_cap > MAX_LEN {
            return Err(TryReserveError::capacity_overflow());
        }
        self.try_materialize(new_cap)
    }

    /// Resizes to `new_len`, using `f` to build any new elements.
    ///
    /// A **borrowed** cow is materialized first when *growing* (hence the
    /// `T: Clone` bound); shrinking a borrowed cow narrows the borrow without
    /// materializing (see [`Self::truncate`]). Use
    /// [`Self::resize_with_owned`] when the cow is known to be owned and
    /// `T: Clone` is unavailable.
    #[track_caller]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        T: Clone,
        F: FnMut() -> T,
    {
        let len = self.raw_len();
        if new_len > len {
            if self.is_borrowed() {
                self.materialize_with_min_capacity(new_len - len);
            }
            // `extend_owned` reserves once and writes directly into spare
            // capacity; see `Self::resize`.
            self.extend_owned(core::iter::repeat_with(f).take(new_len - len));
        } else {
            self.truncate(new_len);
        }
    }

    /// [`Self::resize_with`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::resize_with`].
    #[track_caller]
    pub fn resize_with_owned<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        assert!(
            !self.is_borrowed(),
            "resize_with_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        if new_len > len {
            self.extend_owned(core::iter::repeat_with(f).take(new_len - len));
        } else {
            self.truncate(new_len);
        }
    }

    /// Creates an owned cow with `n` copies of `elem` (like `vec![elem; n]`).
    ///
    /// Clone/drop accounting matches `Vec`'s (`extend_with`): for `n >= 1`,
    /// `elem` is cloned into the first `n - 1` slots and the **original** is
    /// moved into the last one, so `T::clone` runs `n - 1` times and the
    /// original is not dropped during the call. This is also the clone order:
    /// a side-effectful or identity-carrying `Clone` sees the same sequence
    /// and leaves the original in the same slot as `vec![elem; n]`.
    #[must_use]
    #[track_caller]
    pub fn from_elem(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        let mut out = Self::new();
        if n == 0 {
            return out;
        }
        out.reserve_owned(n);
        #[cfg(feature = "specialization")]
        {
            // Clone into slots `[0, n - 1)` and move the original into the
            // last one — exactly `Vec`'s `extend_with` order (exactly `n`
            // values, hence `n` destructors). Under `T: Copy` this is one
            // length publication plus the O(log n) doubling copies of
            // `spec::bulk`; the stable path below is the same accounting,
            // element-wise.
            // Safety: capacity `n` reserved, the cow is owned and empty, and
            // slots `[0, n)` are uninitialized. For a ZST the write consumes
            // `elem` (ZST protocol site 1).
            unsafe { T::tail_fill(elem, &mut out, 0, n) };
        }
        #[cfg(not(feature = "specialization"))]
        {
            // Clone into slots `[0, n - 1)` and move the original into the
            // last one — exactly `Vec`'s `extend_with` order. `extend_owned`
            // carries the hoisted heap fast path and publishes after each
            // clone, so a panic leaves the produced prefix counted and the
            // in-flight original is dropped by the unwind.
            out.extend_owned(core::iter::repeat_with(|| elem.clone()).take(n - 1));
            let l = out.raw_len();
            // Safety: capacity `n` was reserved, slots `[0, n - 1)` are
            // initialized, and slot `l == n - 1` is uninitialized. For a ZST
            // the write consumes `elem` (ZST protocol site 1).
            unsafe { out.base_ptr().add(l).write(elem) };
            out.len = (out.len & HEAP_TAG) | (l + 1);
        }
        out
    }

    /// Reserves room for at least `additional` more elements on an
    /// **already-owned** cow, without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::reserve`].
    #[track_caller]
    pub fn reserve_owned(&mut self, additional: usize) {
        assert!(
            !self.is_borrowed(),
            "reserve_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let needed = match len.checked_add(additional) {
            Some(n) => n,
            None => panic!("capacity overflow"),
        };
        if needed <= self.capacity() {
            return;
        }
        self.grow_to_at_least(needed);
    }

    /// Reserves room for exactly `additional` more elements on an
    /// **already-owned** cow, without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::reserve_exact`].
    #[track_caller]
    pub fn reserve_exact_owned(&mut self, additional: usize) {
        assert!(
            !self.is_borrowed(),
            "reserve_exact_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let needed = match len.checked_add(additional) {
            Some(n) => n,
            None => panic!("capacity overflow"),
        };
        if needed <= self.capacity() {
            return;
        }
        if let Err(e) = self.try_grow(needed, true) {
            e.abort();
        }
    }

    /// Tries to reserve room for at least `additional` more elements on an
    /// **already-owned** cow, without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::try_reserve`] (which copy-on-writes automatically).
    ///
    /// # Errors
    ///
    /// Same as [`CovariantVec::try_reserve`](crate::CovariantVec::try_reserve).
    pub fn try_reserve_owned(&mut self, additional: usize) -> Result<(), TryReserveError> {
        assert!(
            !self.is_borrowed(),
            "try_reserve_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if needed <= self.capacity() {
            return Ok(());
        }
        self.try_grow(needed, false)
    }

    /// Tries to reserve room for exactly `additional` more elements on an
    /// **already-owned** cow, without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state.
    ///
    /// # Errors
    ///
    /// Same as [`CovariantVec::try_reserve`](crate::CovariantVec::try_reserve).
    pub fn try_reserve_exact_owned(&mut self, additional: usize) -> Result<(), TryReserveError> {
        assert!(
            !self.is_borrowed(),
            "try_reserve_exact_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if needed <= self.capacity() {
            return Ok(());
        }
        self.try_grow(needed, true)
    }

    /// Allocates a heap block of `cap` slots, migrating the current inline
    /// elements into it, and switches to the heap state.
    ///
    /// Preconditions: owned (not borrowed), `size_of::<T>() != 0`, `cap > 0`,
    /// `cap <= MAX_LEN`.
    #[track_caller]
    fn alloc_heap(&mut self, cap: usize) {
        // Safety contract of `try_alloc_heap`: `layout` must be
        // `try_heap_layout::<T>(cap)`, which it is here.
        let layout = match try_heap_layout::<T>(cap) {
            Ok(layout) => layout,
            Err(e) => e.abort(),
        };
        if let Err(e) = self.try_alloc_heap(cap, layout) {
            e.abort();
        }
    }

    /// Fallible twin of [`Self::alloc_heap`]: capacity overflow and allocation
    /// failure become `Err`, leaving the cow unchanged.
    ///
    /// `layout` must equal `try_heap_layout::<T>(cap)`; callers that already
    /// computed it (e.g. the inline-migration path of `try_grow`) pass it in to
    /// avoid recomputation.
    fn try_alloc_heap(&mut self, cap: usize, layout: Layout) -> Result<(), TryReserveError> {
        debug_assert!(!self.is_borrowed());
        debug_assert!(size_of::<T>() != 0);
        debug_assert!(cap > 0);
        if cap > MAX_LEN {
            return Err(TryReserveError::capacity_overflow());
        }
        // Safety: valid non-zero layout.
        let new_ptr = unsafe { alloc(layout) };
        let nn = NonNull::new(new_ptr).ok_or(TryReserveError::alloc_error(layout))?;
        let len = self.raw_len();
        // Safety: migrate the initialized inline elements into the fresh block
        // (distinct allocations, non-overlapping bitwise move), then switch.
        unsafe {
            let src = addr_of_mut!(self.data.inline).cast::<T>();
            ptr::copy_nonoverlapping(src, nn.cast::<T>().as_ptr(), len);
            self.data.heap = (nn, cap);
        }
        self.len = HEAP_TAG | len;
        Ok(())
    }

    /// Grows capacity so that at least `needed` elements fit. The cow must be
    /// owned (inline or heap).
    #[track_caller]
    fn grow_to_at_least(&mut self, needed: usize) {
        if let Err(e) = self.try_grow(needed, false) {
            e.abort();
        }
    }

    /// Fallible capacity-growth core shared by the `reserve*` and `try_reserve*`
    /// methods on the **owned** states. `exact` requests `needed` verbatim
    /// instead of the amortized `grow_to` curve.
    fn try_grow(&mut self, needed: usize, exact: bool) -> Result<(), TryReserveError> {
        debug_assert!(!self.is_borrowed());
        if size_of::<T>() == 0 {
            return if needed > MAX_LEN {
                Err(TryReserveError::capacity_overflow())
            } else {
                Ok(())
            };
        }
        let cur = self.capacity();
        let new_cap = planned_cap::<T, N>(cur, needed, exact);
        if new_cap > MAX_LEN {
            return Err(TryReserveError::capacity_overflow());
        }
        let new_layout = try_heap_layout::<T>(new_cap)?;

        if self.is_heap() {
            let old_layout = crate::heap_layout::<T>(cur);
            // Safety: old layout matches the allocation.
            let new_ptr =
                unsafe { realloc(self.data.heap.0.as_ptr(), old_layout, new_layout.size()) };
            let nn = NonNull::new(new_ptr).ok_or(TryReserveError::alloc_error(new_layout))?;
            // Update the tag-consistent heap metadata (a union field write is
            // safe; the heap field has no drop glue).
            self.data.heap = (nn, new_cap);
        } else {
            // Safety: preconditions of `try_alloc_heap` hold (owned, non-ZST,
            // `new_cap > N`); `new_layout == try_heap_layout::<T>(new_cap)`.
            self.try_alloc_heap(new_cap, new_layout)?;
        }
        Ok(())
    }

    /// Ensures uniquely-owned storage, cloning the borrowed data if needed,
    /// and returns a mutable slice.
    ///
    /// This is panic-safe: the union is first switched to a *valid*
    /// owned state with length 0, and the length is advanced after each
    /// successful clone, so a panic inside `T::clone` leaves a consistent
    /// value (the already-cloned prefix is valid, and `Drop` handles it).
    ///
    /// Under the nightly-only `specialization` feature a `T: Copy` element is
    /// filled with one bulk copy instead of element by element (see the crate
    /// docs on `T: Copy` fast paths); the result is identical, since `Copy`
    /// requires `clone` to be a bitwise copy.
    ///
    /// # ZST semantics
    ///
    /// When `T` is zero-sized, materialization adopts the element count
    /// directly and does **not** call `T::clone`. The destructor bookkeeping
    /// stays balanced. Writing the arithmetic out: true copy-on-write would run
    /// `len` clones, for `2·len` total destructor runs (`len` for the borrower's
    /// originals plus `len` for the cow's clones); adoption instead transfers
    /// the count, for the same `2·len` total (`len` borrower originals plus
    /// `len` adopted elements). Only the `Clone::clone` side effects are
    /// skipped. For an exotic ZST whose `Clone`/`Drop` have observable side
    /// effects, this therefore diverges from strict copy-on-write — and, more
    /// visibly, from [`Clone`]: `CovariantCow`'s `clone`/`clone_from` run
    /// `T::clone` once per element for a ZST too (see the `clone_from` note in
    /// this impl), so cloning and then mutating can run more `clone` calls
    /// than `to_mut` on the same borrowed cow. Every *value-consuming* count
    /// update (`push`/`insert`/`resize`/`extend`, and `append` with a
    /// borrowed `other`) adopts, like `to_mut`; `Clone` is the one deliberate
    /// exception, because calling `T::clone` is its entire contract.
    ///
    /// The fallible twin is [`try_to_mut`](Self::try_to_mut): it reports an
    /// allocation failure or a capacity overflow instead of aborting/panicking,
    /// and leaves a borrowed cow borrowed.
    #[track_caller]
    pub fn to_mut(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        if self.is_borrowed() {
            let len = self.raw_len();
            // Copy the borrow out first: it points at *external* memory, so it
            // stays valid even after the union storage is overwritten below.
            let src: &'a [T] = unsafe { self.data.borrowed };

            if size_of::<T>() == 0 {
                // ZSTs: no storage needed. Activate the inline variant
                // and keep the count. (Assigning a union field is safe.)
                self.data.inline = ManuallyDrop::new(MaybeUninit::uninit());
                self.len = len;
            } else if len <= N {
                // Activate a valid empty inline state *before* cloning.
                self.data.inline = ManuallyDrop::new(MaybeUninit::uninit());
                self.len = 0;
                // Safety: `src` is valid for `len` reads and does not overlap
                // `self` (the borrow outlives `self`, and an `&mut self` exists
                // here, so it cannot point into `self`); `self` is owned, empty
                // and has capacity `N >= len`.
                unsafe { T::clone_fill(src.as_ptr(), self, len) };
            } else {
                let cap = len;
                let layout = crate::heap_layout::<T>(cap);
                // Safety: valid non-zero layout.
                let ptr = unsafe { alloc(layout) };
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                // Publish a valid heap state with length 0 *before* cloning, so
                // a panic is recovered by the normal `Drop` path.
                unsafe {
                    self.data.heap = (NonNull::new_unchecked(ptr), cap);
                }
                self.len = HEAP_TAG;
                // Safety: as in the inline branch, with `self` freshly
                // heap-allocated with capacity exactly `len`.
                unsafe { T::clone_fill(src.as_ptr(), self, len) };
            }
        }
        self.as_mut_slice()
    }

    /// Fallible version of [`to_mut`](Self::to_mut).
    ///
    /// Ensures uniquely-owned storage, cloning the borrowed data if needed, and
    /// returns a mutable slice. The clone semantics are exactly `to_mut`'s,
    /// including the ZST adoption rule (see its documentation).
    ///
    /// Unlike `to_mut`, a materialization that cannot be represented as an
    /// allocation (capacity overflow) or whose `alloc` fails returns
    /// [`TryReserveError`] instead of panicking/aborting, and the cow is left
    /// **borrowed and unmodified**. An already-owned cow performs no allocation
    /// and always succeeds.
    ///
    /// A panic inside `T::clone` is recovered exactly as in `to_mut`: the
    /// already-cloned prefix is owned and the cow stays valid (the state is
    /// published before cloning starts).
    ///
    /// The block is sized exactly `len`, like `to_mut` — not on the amortized
    /// growth curve — because the caller asked for access, not for room to
    /// grow. Use [`try_reserve`](Self::try_reserve) to materialize and grow in
    /// one allocation.
    pub fn try_to_mut(&mut self) -> Result<&mut [T], TryReserveError>
    where
        T: Clone,
    {
        if self.is_borrowed() {
            self.try_materialize(self.raw_len())?;
        }
        Ok(self.as_mut_slice())
    }

    /// Fallible copy-on-write materialization into a buffer of capacity
    /// `new_cap`, used by the borrowed-aware `try_reserve*`.
    ///
    /// Unlike [`Self::to_mut`] this fuses the materialization with the capacity
    /// request, so a borrowed `try_reserve` performs a **single** fallible
    /// allocation rather than materializing and then growing. On `Err` the cow
    /// is left borrowed and unmodified.
    ///
    /// Preconditions: borrowed, `raw_len <= new_cap <= MAX_LEN`.
    fn try_materialize(&mut self, new_cap: usize) -> Result<(), TryReserveError>
    where
        T: Clone,
    {
        debug_assert!(self.is_borrowed());
        let len = self.raw_len();
        if size_of::<T>() == 0 {
            // No storage: adopt the count into the inline state (same ZST
            // semantics as `to_mut`).
            self.data.inline = ManuallyDrop::new(MaybeUninit::uninit());
            self.len = len;
            return Ok(());
        }
        // Copy the borrow out first: it points at *external* memory, so it stays
        // valid after the union storage is overwritten below.
        let src: &'a [T] = unsafe { self.data.borrowed };

        if new_cap <= N {
            // Fits inline: no allocation, so this cannot fail. Publish a valid
            // empty inline state *before* cloning.
            self.data.inline = ManuallyDrop::new(MaybeUninit::uninit());
            self.len = 0;
            // Safety: `src` is valid for `len <= new_cap <= N` reads and does
            // not overlap `self`; `self` is owned, empty, capacity `N`.
            unsafe { T::clone_fill(src.as_ptr(), self, len) };
            return Ok(());
        }

        let layout = try_heap_layout::<T>(new_cap)?;
        // Safety: valid non-zero layout.
        let ptr = unsafe { alloc(layout) };
        let nn = NonNull::new(ptr).ok_or(TryReserveError::alloc_error(layout))?;
        // Publish a valid empty heap state *before* cloning, so a panic inside
        // `T::clone` is recovered by the normal `Drop` path. (Union field
        // assignment is safe; the heap field has no drop glue.)
        self.data.heap = (nn, new_cap);
        self.len = HEAP_TAG;
        // Safety: as in the inline branch, with `self` freshly heap-allocated
        // with capacity `new_cap >= len`.
        unsafe { T::clone_fill(src.as_ptr(), self, len) };
        Ok(())
    }

    /// Infallible twin of [`Self::try_materialize`]: materializes a **borrowed**
    /// cow with capacity for at least `len + additional` elements, fusing the
    /// copy-on-write with the reservation into a single allocation.
    ///
    /// This is what makes borrowed `push`/`insert`/`resize`/`reserve`/`extend`
    /// allocate once instead of twice (`to_mut` sized the clone exactly, then
    /// the mutation grew again). Allocation failure aborts (via
    /// [`TryReserveError::abort`]), matching the other infallible capacity
    /// methods; capacity overflow panics with `"capacity overflow"`.
    ///
    /// The block is sized on the amortized `planned_cap` curve: when
    /// `2 * len` exceeds `len + additional`, the reservation is up to ~2×
    /// the required length. That is the same curve `Vec::reserve` uses (and
    /// `Vec::append`/`extend_from_slice` reserve through it), so a one-shot
    /// bulk operation over a large borrowed slice may hold ~2× peak memory.
    /// The `_exact` variants ([`Self::reserve_exact`],
    /// [`Self::try_reserve_exact`]) size the block exactly.
    ///
    /// Preconditions: borrowed, `additional <= MAX_LEN - raw_len`.
    #[track_caller]
    #[inline]
    fn materialize_with_min_capacity(&mut self, additional: usize)
    where
        T: Clone,
    {
        self.materialize_with_capacity(additional, false);
    }

    /// [`Self::materialize_with_min_capacity`] with `exact` capacity: the
    /// buffer is sized `len + additional` verbatim (the `reserve_exact`
    /// contract) instead of the amortized growth curve. Used by
    /// [`Self::reserve_exact`]; [`Self::try_reserve_exact`] is the fallible
    /// twin, sharing [`planned_cap`]'s exact branch.
    ///
    /// Preconditions: borrowed, `additional <= MAX_LEN - raw_len`.
    #[track_caller]
    #[inline]
    fn materialize_with_exact_capacity(&mut self, additional: usize)
    where
        T: Clone,
    {
        self.materialize_with_capacity(additional, true);
    }

    /// Shared core of the two infallible fused materializations. `exact`
    /// selects `len + additional` verbatim as the new capacity, otherwise the
    /// amortized `planned_cap` curve (with the fits-inline shortcut: when the
    /// request already fits inline, materialize there rather than paying for
    /// the first spill).
    #[track_caller]
    fn materialize_with_capacity(&mut self, additional: usize, exact: bool)
    where
        T: Clone,
    {
        debug_assert!(self.is_borrowed());
        let len = self.raw_len();
        let needed = match len.checked_add(additional) {
            Some(n) => n,
            None => panic!("capacity overflow"),
        };
        let new_cap = if !exact && needed <= N {
            N
        } else {
            planned_cap::<T, N>(len, needed, exact)
        };
        if new_cap > MAX_LEN {
            panic!("capacity overflow");
        }
        if let Err(e) = self.try_materialize(new_cap) {
            e.abort();
        }
    }

    #[inline]
    #[track_caller]
    pub fn push(&mut self, value: T)
    where
        T: Clone,
    {
        if self.is_borrowed() {
            self.materialize_with_min_capacity(1);
        }
        self.push_owned(value);
    }

    /// Appends an element to an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::push`] (which copy-on-writes automatically, at the
    /// cost of a `T: Clone` bound).
    #[inline]
    #[track_caller]
    pub fn push_owned(&mut self, value: T) {
        assert!(!self.is_borrowed(), "push_owned on a borrowed CovariantCow");
        if size_of::<T>() == 0 {
            // A ZST is never heap-backed, so `raw_len + 1` must not carry the
            // heap tag (nor the borrowed tag, which `BORROWED_TAG` would set if
            // `raw_len` reached it). A real (not debug) assert, since in release
            // a violated invariant would silently clear a tag bit.
            assert!(!self.is_heap(), "ZST cows are never heap-backed");
            assert!(self.raw_len() < MAX_LEN, "capacity overflow");
            self.len = self.raw_len() + 1;
            // ZST protocol (site 1): the count *is* the ownership; see the
            // crate-level "ZST ownership protocol". The one conceptual
            // destructor is deferred to `drop_contents`/`clear`.
            mem::forget(value);
            return;
        }
        let len = self.raw_len();
        if !self.is_heap() {
            if len < N {
                // Safety: inline slot [len] is uninitialized.
                unsafe {
                    addr_of_mut!(self.data.inline)
                        .cast::<T>()
                        .add(len)
                        .write(value);
                }
                self.len = len + 1;
            } else {
                // The spill branch hard-codes `N` as the migration count and
                // the new slot index, so the inline container must hold exactly
                // `N` elements (`len < N` is the branch above).
                debug_assert_eq!(self.raw_len(), N);
                let new_cap = grow_to::<N>(N, crate::min_non_zero_cap::<T>());
                // Defense in depth: `grow_to` saturates at `MAX_LEN`.
                assert!(new_cap > N, "capacity overflow");
                let layout = crate::heap_layout::<T>(new_cap);
                // Safety: valid non-zero layout.
                let new_ptr = unsafe { alloc(layout) };
                if new_ptr.is_null() {
                    handle_alloc_error(layout);
                }
                // Safety: migrate the initialized inline elements into the fresh
                // block. The fresh block and the inline array are distinct
                // allocations, so the copy does not overlap; it is a bitwise
                // move (no drop glue), valid for `!Copy`.
                unsafe {
                    let src = addr_of_mut!(self.data.inline).cast::<T>();
                    let dst = new_ptr as *mut T;
                    ptr::copy_nonoverlapping(src, dst, N);
                    self.data.heap = (NonNull::new_unchecked(new_ptr), new_cap);
                    dst.add(N).write(value);
                }
                self.len = HEAP_TAG | (N + 1);
            }
        } else {
            let cap = self.heap_cap();
            if len < cap {
                // Safety: heap slot [len] is uninitialized.
                unsafe {
                    self.data.heap.0.cast::<T>().as_ptr().add(len).write(value);
                }
                self.len = HEAP_TAG | (len + 1);
            } else {
                let new_cap = grow_to::<N>(cap, crate::min_non_zero_cap::<T>());
                // Defense in depth: a saturated `new_cap == cap` would
                // write out of bounds *and* set `BORROWED_TAG` via `len + 1`.
                assert!(new_cap > cap, "capacity overflow");
                let new_layout = crate::heap_layout::<T>(new_cap);
                let old_layout = crate::heap_layout::<T>(cap);
                // Safety: old layout matches the allocation.
                let new_ptr =
                    unsafe { realloc(self.data.heap.0.as_ptr(), old_layout, new_layout.size()) };
                if new_ptr.is_null() {
                    handle_alloc_error(new_layout);
                }
                // Safety: update the tag-consistent heap metadata, then write the fresh slot.
                unsafe {
                    self.data.heap = (NonNull::new_unchecked(new_ptr), new_cap);
                    self.data.heap.0.cast::<T>().as_ptr().add(len).write(value);
                }
                self.len = HEAP_TAG | (len + 1);
            }
        }
    }

    /// Removes and returns the last element of an **already-owned** cow,
    /// without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::pop`].
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn pop_owned(&mut self) -> Option<T> {
        assert!(!self.is_borrowed(), "pop_owned on a borrowed CovariantCow");
        let len = self.raw_len();
        if len == 0 {
            return None;
        }
        // Safety: slot [len - 1] is initialized. ZST protocol (site 3): a ZST
        // slot was never written, but `read` touches no bytes and the count is
        // decremented below, so no conceptual destructor is run twice.
        let value = unsafe { self.base_ptr().add(len - 1).read() };
        self.len = (self.len & HEAP_TAG) | (len - 1);
        Some(value)
    }

    /// Removes and returns the last element.
    ///
    /// A **borrowed** cow stays borrowed and is merely narrowed by one element,
    /// so exactly one element is cloned — not the whole slice — consistent with
    /// the borrowed-shrinking family ([`Self::truncate`], [`Self::resize`]). Use
    /// [`Self::pop_owned`] when the cow is known to be owned.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
        if self.is_borrowed() {
            let len = self.raw_len();
            if len == 0 {
                return None;
            }
            // Clone exactly the last element; the borrower keeps the rest.
            let last = self.as_slice()[len - 1].clone();
            // Safety: reading the borrowed variant to narrow it.
            let b: &'a [T] = unsafe { self.data.borrowed };
            self.data.borrowed = &b[..len - 1];
            self.len = BORROWED_TAG | (len - 1);
            return Some(last);
        }
        self.pop_owned()
    }

    /// Drops all elements.
    ///
    /// An **owned** cow keeps its storage state (inline stays inline, heap keeps
    /// its block and capacity). A **borrowed** cow stays borrowed, narrowed to
    /// an empty slice — consistent with [`Self::truncate`] and the
    /// [`Self::resize`] shrink path, neither of which materializes a borrowed
    /// cow. (It is exactly `truncate(0)`.)
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Shortens the cow to `new_len`, dropping the removed elements.
    ///
    /// Truncating a **borrowed** cow does not materialize it: the cow stays
    /// borrowed and merely narrows the slice it points at, so no element is
    /// cloned or dropped (the discarded tail is still owned by the borrower).
    /// An owned cow keeps its state (inline stays inline, heap keeps its block
    /// and capacity) — it never reallocates.
    pub fn truncate(&mut self, new_len: usize) {
        let len = self.raw_len();
        if new_len >= len {
            return;
        }
        if self.is_borrowed() {
            // Stay borrowed: keep the truncated prefix of the original slice.
            let b: &'a [T] = unsafe { self.data.borrowed };
            self.data.borrowed = &b[..new_len];
            self.len = BORROWED_TAG | new_len;
            return;
        }
        // Publish the new length *before* running destructors (std parity), so
        // a panicking destructor leaks the un-dropped tail instead of leaving
        // already-dropped slots counted (which `Drop` would re-drop).
        self.len = (self.len & HEAP_TAG) | new_len;
        if mem::needs_drop::<T>() {
            // Derive the pointer *after* the `len` write (Stacked Borrows).
            // Safety: `[new_len, len)` were initialized and are no longer
            // claimed by the container.
            let p = unsafe { self.base_ptr().add(new_len) };
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len - new_len)) };
        }
    }

    /// Moves the elements back into the inline buffer and frees the heap block,
    /// if the length currently fits in `N`. A no-op for a borrowed or inline
    /// cow.
    pub fn shrink_to_inline(&mut self) {
        if !self.is_heap() {
            return;
        }
        let len = self.raw_len();
        if len > N {
            return;
        }
        // Safety: heap state.
        let (ptr, cap) = unsafe { self.data.heap };
        // Safety: migrate `len <= N` elements back into the union. The heap
        // block and the inline array are distinct allocations, so the copy does
        // not overlap; it is a bitwise move (no drop glue), valid for `!Copy`.
        unsafe {
            let src = ptr.cast::<T>().as_ptr();
            let dst = addr_of_mut!(self.data.inline).cast::<T>();
            ptr::copy_nonoverlapping(src, dst, len);
            // ZST can never be in the heap state, so this is a real allocation.
            debug_assert!(size_of::<T>() != 0);
            dealloc(ptr.as_ptr(), crate::heap_layout::<T>(cap));
        }
        self.len = len;
    }

    /// The spare capacity of an **already-owned** cow as a slice of
    /// `MaybeUninit<T>`, starting at `len()` and running for
    /// `capacity() - len()` elements (std parity, and the same low-level append
    /// workflow as [`CovariantVec::spare_capacity_mut`](crate::CovariantVec::spare_capacity_mut):
    /// write into the slots, then publish with [`Self::set_len`]).
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state: handing out a `*mut T` into
    /// the still-live shared `&'a [T]` would alias. Call [`Self::to_mut`] first.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn spare_capacity_mut_owned(&mut self) -> &mut [MaybeUninit<T>] {
        assert!(
            !self.is_borrowed(),
            "spare_capacity_mut_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let cap = self.capacity();
        // Safety: the owned allocation has `capacity()` slots; slots
        // `[len, cap)` are uninitialized but valid for writes viewed as
        // `MaybeUninit<T>`. For a ZST `cap - len` is a huge count of zero-sized
        // slots, which is sound (any aligned pointer is a valid ZST slice base).
        unsafe {
            slice::from_raw_parts_mut(self.base_ptr().cast::<MaybeUninit<T>>().add(len), cap - len)
        }
    }

    /// Directly sets the length (same contract as `Vec::set_len`).
    ///
    /// # Safety
    ///
    /// `new_len <= capacity()`; if growing, the newly exposed slots must be
    /// initialized; if shrinking, the removed elements must already have been
    /// dropped. Must not be called in the borrowed state (there is no owned
    /// storage to reinterpret).
    #[track_caller]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(!self.is_borrowed(), "set_len on a borrowed CovariantCow");
        debug_assert!(new_len <= self.capacity());
        self.len = (self.len & HEAP_TAG) | new_len;
    }

    /// Tag-preserving length publication for the clone loops, which are
    /// statically known to run on an **owned** cow.
    ///
    /// `set_len` carries a real `assert!(!is_borrowed())`, which would re-run
    /// once per element in `CloneSink::publish_len` (the hot element-wise
    /// clone loop). The `CloneSink` safety contract already guarantees the
    /// sink is owned, so only a `debug_assert!` remains here.
    ///
    /// # Safety
    ///
    /// Slots `[0, new_len)` must be initialized, the cow must be owned, and
    /// `new_len <= capacity()`.
    #[inline]
    unsafe fn publish_len_owned(&mut self, new_len: usize) {
        debug_assert!(!self.is_borrowed());
        debug_assert!(new_len <= self.capacity());
        // Preserve both tag bits (`BORROWED_TAG` is zero for owned cows, but
        // preserving it matches `backshift_set_len`'s discipline).
        self.len = (self.len & !LEN_MASK) | new_len;
    }

    /// Builds the [`CowDrain`] for already-validated `(begin, end)` over a
    /// length-`len` container: inert for an empty range, otherwise taking the
    /// whole container out of `self` (the caller must have materialized any
    /// borrowed state first).
    fn make_drain(&mut self, begin: usize, end: usize, len: usize) -> CowDrain<'_, 'a, T, N> {
        make_drain_impl(self, begin, end, len)
    }

    /// Removes the specified range and returns an iterator over the removed
    /// elements, in order, from either end.
    ///
    /// A **borrowed** cow is drained lazily: the kept prefix and suffix are cloned
    /// into a fresh owned cow at construction and the removed elements are
    /// cloned only as they are yielded (see [`CowDrainLazy`]), so a partial
    /// drain never clones the elements it does not yield. An empty range clones
    /// nothing and leaves a borrowed cow borrowed.
    ///
    /// An **owned** cow behaves exactly like [`CovariantVec::drain`](crate::CovariantVec::drain):
    /// the whole container is moved into the returned value and restored on
    /// drop, so the tail is shifted back and the length restored — even while
    /// unwinding. See that method for the aliasing rationale and the
    /// `mem::forget` semantics (forgetting a non-empty drain leaks the kept
    /// elements; an empty range never takes the cow).
    ///
    /// Use [`Self::drain_owned`] when the cow is known to be owned and `T: Clone`
    /// is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if the range's start is greater than its end, or its end is
    /// greater than the length.
    #[track_caller]
    pub fn drain<R>(&mut self, range: R) -> CowDrainLazy<'_, 'a, T, N>
    where
        R: RangeBounds<usize>,
        T: Clone,
    {
        // Validate the range *before* materializing: `raw_len` already masks
        // off the tag bits, so it is correct for the borrowed state too.
        let len = self.raw_len();
        let (begin, end) = crate::drain_bounds(&range, len);
        if !self.is_borrowed() {
            return CowDrainLazy::from_owned(self.make_drain(begin, end, len));
        }
        if begin == end {
            // Empty range: stay borrowed, clone nothing.
            // Safety: reading the borrowed variant.
            let src: &'a [T] = unsafe { self.data.borrowed };
            return CowDrainLazy::from_borrowed_empty(self, src, begin);
        }
        if size_of::<T>() == 0 {
            // ZSTs have no storage or clone cost; the owned drain's documented
            // "adopt then drop removed" accounting is preserved.
            self.to_mut();
            return CowDrainLazy::from_owned(self.make_drain(begin, end, len));
        }
        // Safety: reading the borrowed variant.
        let src: &'a [T] = unsafe { self.data.borrowed };
        // Clone only the kept prefix and suffix now; the removed elements are
        // cloned lazily as they are yielded (and never if they are not).
        let mut kept = CovariantCow::new();
        kept.reserve_owned(len);
        if begin > 0 {
            // Safety: `kept` is owned, empty and has capacity for `len`;
            // `src[..begin]` is a distinct initialized slice.
            unsafe { T::clone_fill(src.as_ptr(), &mut kept, begin) };
        }
        if end < len {
            // Safety: as above, filling `[begin, begin + len - end)`.
            unsafe { T::clone_fill_range(src.as_ptr().add(end), &mut kept, begin, len - end) };
        }
        CowDrainLazy::from_borrowed_kept(self, src, begin, end, kept)
    }

    /// [`Self::drain`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::drain`].
    #[track_caller]
    pub fn drain_owned<R>(&mut self, range: R) -> CowDrain<'_, 'a, T, N>
    where
        R: RangeBounds<usize>,
    {
        assert!(
            !self.is_borrowed(),
            "drain_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let (begin, end) = crate::drain_bounds(&range, len);
        self.make_drain(begin, end, len)
    }

    // ----------------------------------------------------------------------
    // Bulk operations
    // ----------------------------------------------------------------------

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// A **borrowed** `other` is not materialized: its `&'a [T]` is cloned
    /// directly into `self`, then narrowed to an empty borrow (the
    /// `truncate(0)` semantics) — costing exactly the clone `self` would
    /// have paid anyway. When `self` is also borrowed, the copy-on-write and
    /// the room for `other` are fused into a single allocation by the
    /// crate's fused materialization helper.
    ///
    /// The one exception is a **zero-sized** `T`: the count is the
    /// ownership, so a borrowed `other` is *adopted* (`self`'s count grows
    /// by `other_len`) without calling `T::clone`, exactly like
    /// [`Self::to_mut`]'s ZST materialization. Destructor bookkeeping stays
    /// balanced; only the `Clone` side effects are skipped.
    ///
    /// An **owned** `other` is moved element-wise with no clone (the same
    /// native path as [`Self::append_owned`]); `self` is materialized first
    /// if it is borrowed. For a zero-sized element only the counts move.
    ///
    /// `other` keeps its storage state and capacity when it is owned (std
    /// parity); a borrowed `other` stays borrowed but empty.
    ///
    /// # Panics
    ///
    /// Panics with `"capacity overflow"` if the combined length is not
    /// representable, and aborts through the allocation-error hook if a
    /// reservation fails. Both happen before any element moves.
    #[track_caller]
    pub fn append(&mut self, other: &mut Self)
    where
        T: Clone,
    {
        let other_len = other.raw_len();
        if other_len == 0 {
            return;
        }
        if size_of::<T>() == 0 {
            // ZST: the count *is* the ownership. Materializing a borrowed
            // `self` adopts its count without cloning; a borrowed `other` is
            // narrowed (no drops) and an owned one transfers its count.
            //
            // Validate the combined count *before* any state change: a
            // capacity-overflow panic must leave a borrowed `self` borrowed
            // (the non-ZST path below likewise checks capacity before
            // materializing). `MAX_LEN` is the ZST representable-count bound.
            let needed = match self.raw_len().checked_add(other_len) {
                Some(n) if n <= MAX_LEN => n,
                _ => panic!("capacity overflow"),
            };
            if self.is_borrowed() {
                self.to_mut();
            }
            if other.is_borrowed() {
                // Same tag-hygiene guard as `push_owned`: the untagged length
                // write below must never run on a heap-backed ZST cow.
                assert!(!self.is_heap(), "ZST cows are never heap-backed");
                self.len = needed;
                other.truncate(0);
            } else {
                self.append_owned(other);
            }
            return;
        }
        if self.is_borrowed() {
            // Fuse the copy-on-write with the room for `other`: one
            // allocation even when both sides are borrowed.
            self.materialize_with_min_capacity(other_len);
        }
        if other.is_borrowed() {
            // Clone the borrowed slice straight into `self` through the
            // range-fill primitive: one bulk copy for `T: Copy` (the old
            // `extend_owned(iter().cloned())` could never reach
            // `memcpy_fill`, which only fills from length 0), the same
            // per-element clone loop otherwise. The borrow is narrowed only
            // after every clone succeeded, so a panic inside `T::clone`
            // leaves `other` intact (with the originals still owned by the
            // borrower).
            self.reserve_owned(other_len);
            let src: &'a [T] = other.as_borrowed().unwrap();
            let self_len = self.raw_len();
            // Safety: `self` is owned (materialized above if it was borrowed)
            // with capacity for `self_len + other_len`; the borrowed slice is
            // a distinct allocation, valid for `other_len` reads, and its
            // slots do not overlap `self`'s. Slots
            // `[self_len, self_len + other_len)` are uninitialized.
            unsafe { self.fill_range(src.as_ptr(), self_len, other_len) };
            other.truncate(0);
        } else {
            self.append_owned(other);
        }
    }

    /// [`Self::append`] on **already-owned** cows, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if either cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::append`].
    #[track_caller]
    pub fn append_owned(&mut self, other: &mut Self) {
        assert!(
            !self.is_borrowed(),
            "append_owned on a borrowed CovariantCow"
        );
        assert!(
            !other.is_borrowed(),
            "append_owned with a borrowed CovariantCow"
        );
        let other_len = other.raw_len();
        if other_len == 0 {
            return;
        }
        self.reserve_owned(other_len);
        if size_of::<T>() == 0 {
            assert!(!self.is_heap(), "ZST cows are never heap-backed");
            assert!(!other.is_heap(), "ZST cows are never heap-backed");
            self.len = self.raw_len() + other_len;
            other.len = 0;
            return;
        }
        let self_len = self.raw_len();
        // Derived after `reserve_owned` (which may have reallocated).
        // Safety: capacity for `self_len + other_len` slots; slots
        // `[self_len, self_len + other_len)` are uninitialized.
        let dst = unsafe { self.base_ptr().add(self_len) };
        // Safety: `[0, other_len)` are initialized in `other`'s owned storage.
        let src = other.base_ptr_const();
        // Safety: distinct containers, hence distinct storage; bitwise move —
        // the source slots are retired by the length write below without
        // running destructors.
        unsafe { ptr::copy_nonoverlapping(src, dst, other_len) };
        self.len = (self.len & HEAP_TAG) | (self_len + other_len);
        other.len &= HEAP_TAG;
    }

    /// Appends a copy of every element of `other`, like
    /// [`Vec::extend_from_slice`](alloc::vec::Vec::extend_from_slice), but
    /// bounded on `T: Copy` so the copy is a single bulk `memcpy` **on
    /// stable** (no `specialization` feature required).
    ///
    /// A borrowed cow is materialized first, with the room for `other` fused
    /// into the same allocation; an empty `other` is a no-op that leaves a
    /// borrowed cow borrowed.
    ///
    /// # Panics
    ///
    /// Panics with `"capacity overflow"` if the combined length is not
    /// representable, and aborts through the allocation-error hook if the
    /// reservation fails. Both happen before any element is copied.
    #[track_caller]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Copy,
    {
        let other_len = other.len();
        if other_len == 0 {
            return;
        }
        if self.is_borrowed() {
            // `T: Copy` implies `Clone`, so fuse the copy-on-write and the
            // room for `other` into one allocation.
            self.materialize_with_min_capacity(other_len);
        } else {
            self.reserve_owned(other_len);
        }
        let self_len = self.raw_len();
        if size_of::<T>() == 0 {
            // No bytes exist; the count is the ownership. The reserve paths
            // enforced `MAX_LEN`, and a `Copy` ZST has no destructor to
            // balance.
            assert!(!self.is_heap(), "ZST cows are never heap-backed");
            self.len = self.raw_len() + other_len;
            return;
        }
        // Safety: capacity for `self_len + other_len` slots, so
        // `[self_len, self_len + other_len)` is in bounds and uninitialized.
        // `other` cannot overlap this cow's owned storage: a borrowed source
        // was materialized into a fresh block above, and safe code cannot
        // hold a `&[T]` into an owned cow's block alongside `&mut self`.
        unsafe {
            let dst = self.base_ptr().add(self_len);
            ptr::copy_nonoverlapping(other.as_ptr(), dst, other_len);
            self.len = (self.len & HEAP_TAG) | (self_len + other_len);
        }
    }

    /// Splits the cow in two at `at`: `self` keeps `[0, at)` and the new
    /// return value owns `[at, len)`.
    ///
    /// For an **owned** cow the result is state-normalized exactly like
    /// [`Clone`]: a tail that fits the inline buffer is inline (no
    /// allocation), a longer tail gets a heap block with capacity exactly its
    /// length. `self` keeps its state and capacity.
    ///
    /// For a **borrowed** cow the split is **zero-copy** and needs no
    /// `T: Clone`: the tail is a borrowed cow over the same `&'a [T]`, and
    /// `self` is narrowed to the prefix — two disjoint sub-slices of the
    /// original borrow.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[track_caller]
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self {
        let len = self.raw_len();
        assert!(
            at <= len,
            "`at` split index (is {at}) should be <= len (is {len})"
        );
        let tail = len - at;
        // The borrowed branch comes first so `at == len` still yields a
        // *borrowed* (empty) tail: `split_at(len)` is a valid split, and a
        // borrowed cow stays borrowed on both sides for every `at`.
        if self.is_borrowed() {
            // Zero-copy: `split_at` yields two disjoint sub-slices of the
            // same borrow, both with the original `'a`.
            let b: &'a [T] = unsafe { self.data.borrowed };
            let (head, tail_slice) = b.split_at(at);
            self.data.borrowed = head;
            self.len = BORROWED_TAG | at;
            return Self::from_borrowed(tail_slice);
        }
        let mut out = Self::new();
        if tail == 0 {
            return out;
        }
        if size_of::<T>() == 0 {
            // ZST: the count is the ownership; no bytes move and neither
            // side can be heap-backed.
            out.len = tail;
            self.len = at;
            return out;
        }
        if tail > N {
            // Allocates exactly `tail` before any element moves.
            out.alloc_heap(tail);
        }
        // Safety: `[at, len)` are initialized in `self`'s owned storage.
        let src = unsafe { self.base_ptr().add(at) };
        // Safety: `out` owns storage for `tail` slots and is empty.
        let dst = out.base_ptr();
        // Safety: distinct containers, hence distinct storage; bitwise move —
        // the source slots are retired by the length writes below.
        unsafe { ptr::copy_nonoverlapping(src, dst, tail) };
        out.len = (out.len & HEAP_TAG) | tail;
        self.len = (self.len & HEAP_TAG) | at;
        out
    }

    /// Replaces the elements of `range` with those produced by
    /// `replace_with`, returning the removed elements as a [`CowSplice`].
    ///
    /// A **borrowed** cow is materialized first (hence the `T: Clone` bound)
    /// unless the range is empty *and* the replacement **reports** an empty
    /// upper bound (`size_hint().1 == Some(0)`), in which case the cow stays
    /// borrowed and the returned `CowSplice` is inert. That shortcut trusts
    /// the iterator's upper bound. **`size_hint` is not a safety contract,
    /// and this is the only place the crate relies on it:** a replacement
    /// that reports zero and then yields an element is a logic error and
    /// panics when the `CowSplice` is dropped (the `Drop` path deliberately
    /// carries no `T: Clone` bound, so it cannot materialize lazily). An
    /// empty range with a replacement that might yield is materialized up
    /// front: it has to be written into owned storage.
    ///
    /// A replacement longer than the range is written in geometrically
    /// growing chunks (see [`Splice`](crate::Splice)); each chunk reserves
    /// room, so an inline cow can spill to the heap.
    ///
    /// Range validation and leak semantics mirror [`Self::drain`].
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds, and — only for a borrowed cow
    /// with an empty range — if `replace_with` reports an empty upper size
    /// hint and then yields an element (see above). The latter is a logic
    /// error in the iterator; no element is moved or dropped by the cow. A
    /// replacement whose lower size hint is too large to reserve may also
    /// panic with `"capacity overflow"` while a borrowed cow is materialized,
    /// before any element is moved.
    #[track_caller]
    pub fn splice<R, I>(
        &mut self,
        range: R,
        replace_with: I,
    ) -> CowSplice<'_, 'a, T, N, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
        T: Clone,
    {
        let len = self.raw_len();
        let (begin, end) = crate::drain_bounds(&range, len);
        let replace_with = replace_with.into_iter();
        if self.is_borrowed() && (begin != end || replace_with.size_hint().1 != Some(0)) {
            // Fold the replacement's lower size hint into the materialization:
            // if it may be longer than the hole, reserve the difference now so
            // phase 2 does not reallocate. This mirrors std's `Splice::drop`,
            // which sizes its tail move from the same lower bound. A lower
            // bound that overstates the length only over-reserves; one that
            // understates it — or a non-fused iterator that ends phase 1
            // early — may cost one extra reservation in phase 2, never
            // correctness.
            let extra = replace_with.size_hint().0.saturating_sub(end - begin);
            if extra == 0 {
                if begin == end {
                    // Empty range on a borrowed cow whose replacement reports
                    // a zero lower bound but might still yield (`filter`,
                    // `flat_map`): materialize with a small amount of headroom
                    // so the `CowSplice` chunk loop does not reallocate for its
                    // first chunk. `upper != Some(0)` is guaranteed by the
                    // branch condition above, so `headroom >= 1`.
                    let headroom = replace_with
                        .size_hint()
                        .1
                        .map_or(crate::SPLICE_CHUNK, |u| u.min(crate::SPLICE_CHUNK));
                    self.materialize_with_exact_capacity(headroom);
                } else {
                    // The replacement fits the hole: `to_mut`'s exact-`len`
                    // allocation is optimal.
                    self.to_mut();
                }
            } else {
                self.materialize_with_min_capacity(extra);
            }
        }
        CowSplice {
            drain: self.make_drain(begin, end, len),
            replace_with,
        }
    }

    /// [`Self::splice`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::splice`].
    #[track_caller]
    pub fn splice_owned<R, I>(
        &mut self,
        range: R,
        replace_with: I,
    ) -> CowSplice<'_, 'a, T, N, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        assert!(
            !self.is_borrowed(),
            "splice_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        let (begin, end) = crate::drain_bounds(&range, len);
        CowSplice {
            drain: self.make_drain(begin, end, len),
            replace_with: replace_with.into_iter(),
        }
    }

    /// Moves out and yields every element `e` for which `f(&mut e)` returns
    /// `true`, keeping the rest, in order.
    ///
    /// Materializes a borrowed cow first (hence the `T: Clone` bound), then
    /// runs the same in-place backshift scan as
    /// [`Self::retain_mut`], with matched elements yielded instead of
    /// dropped. See [`CowExtractIf`] for the scan-state and leak semantics.
    pub fn extract_if<F>(&mut self, pred: F) -> CowExtractIf<'_, 'a, T, N, F>
    where
        T: Clone,
        F: FnMut(&mut T) -> bool,
    {
        if self.is_borrowed() {
            self.to_mut();
        }
        self.extract_if_owned(pred)
    }

    /// [`Self::extract_if`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::extract_if`].
    pub fn extract_if_owned<F>(&mut self, pred: F) -> CowExtractIf<'_, 'a, T, N, F>
    where
        F: FnMut(&mut T) -> bool,
    {
        assert!(
            !self.is_borrowed(),
            "extract_if_owned on a borrowed CovariantCow"
        );
        let original_len = self.raw_len();
        // Owned states have `BORROWED_TAG == 0`; preserve `HEAP_TAG`.
        self.len &= HEAP_TAG;
        CowExtractIf {
            cow: self,
            pred,
            idx: 0,
            del: 0,
            original_len,
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable
    /// reference to each element.
    ///
    /// Materializes a borrowed cow first (hence the `T: Clone` bound), then runs
    /// the same in-place, never-reallocating, panic-safe scan as
    /// [`CovariantVec::retain_mut`](crate::CovariantVec::retain_mut). See that
    /// method for the panic-safety guarantees. Use [`Self::retain_mut_owned`]
    /// when the cow is known to be owned and `T: Clone` is unavailable.
    #[track_caller]
    pub fn retain_mut<F>(&mut self, f: F)
    where
        T: Clone,
        F: FnMut(&mut T) -> bool,
    {
        if self.is_borrowed() {
            self.to_mut();
        }
        self.retain_mut_owned(f);
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements `e` for which `f(&e)` returns `false`. See
    /// [`Self::retain_mut`].
    ///
    /// A **borrowed** cow is not materialized when every element is kept: the
    /// scan reads the borrowed slice directly, and only when the first removal
    /// is found does it build an owned cow from the kept prefix. Removed
    /// elements are never cloned. If `f` (or a clone) panics, the cow is left
    /// borrowed and unchanged.
    #[track_caller]
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Clone,
        F: FnMut(&T) -> bool,
    {
        if !self.is_borrowed() {
            self.retain_mut_owned(|x| f(x));
            return;
        }
        if size_of::<T>() == 0 {
            // ZSTs have no storage and no clone cost; keep the documented
            // "adopt then drop removed" accounting of `retain_mut`.
            self.to_mut();
            self.retain_mut_owned(|x| f(x));
            return;
        }
        let len = self.raw_len();
        // Safety: reading the borrowed variant. The slice points at external
        // memory, so it stays valid while we rebuild `self`.
        let src: &'a [T] = unsafe { self.data.borrowed };
        // Phase 1: find the first removal without touching `self`.
        let mut k = 0;
        while k < len {
            if !f(&src[k]) {
                break;
            }
            k += 1;
        }
        if k == len {
            // Everything kept: stay borrowed, clone nothing.
            return;
        }
        // Phase 2: build the owned result. Reserving the worst case (all
        // remaining elements kept) keeps the append loop allocation-free.
        let mut out = Self::new();
        out.reserve_owned(len);
        if k > 0 {
            // Safety: `out` is owned, empty and has capacity for all `len`
            // slots; `src[..k]` is a distinct, initialized borrowed slice.
            unsafe { T::clone_fill(src.as_ptr(), &mut out, k) };
        }
        for x in &src[k + 1..] {
            if f(x) {
                out.push_owned(x.clone());
            }
        }
        *self = out;
    }

    /// [`Self::retain_mut`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::retain_mut`].
    #[track_caller]
    pub fn retain_mut_owned<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        assert!(
            !self.is_borrowed(),
            "retain_mut_owned on a borrowed CovariantCow"
        );
        crate::retain_mut_impl(self, f);
    }

    /// [`Self::retain`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::retain`].
    #[track_caller]
    pub fn retain_owned<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut_owned(|x| f(x));
    }

    /// Inserts `value` at `index`, shifting later elements right. Materializes
    /// a borrowed cow first (hence the `T: Clone` bound).
    ///
    /// The index is validated *before* materializing, so an out-of-range index
    /// panics without paying for a clone. Use [`Self::insert_owned`] when the
    /// cow is known to be owned and `T: Clone` is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    #[track_caller]
    pub fn insert(&mut self, index: usize, value: T)
    where
        T: Clone,
    {
        let len = self.raw_len();
        assert!(
            index <= len,
            "insertion index (is {index}) should be <= len (is {len})"
        );
        if self.is_borrowed() {
            // Materialization reserves room for one more element, so the ZST
            // BORROWED_TAG-collision bound (`raw_len` must stay below
            // `MAX_LEN`, or bit 62 would be set) is enforced here by the
            // reserve path; `push` and the `extend` paths instead carry the
            // explicit `raw_len < MAX_LEN` assert themselves.
            self.materialize_with_min_capacity(1);
        }
        self.insert_owned(index, value);
    }

    /// [`Self::insert`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::insert`] — or if `index > len`.
    #[track_caller]
    pub fn insert_owned(&mut self, index: usize, value: T) {
        assert!(
            !self.is_borrowed(),
            "insert_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        assert!(
            index <= len,
            "insertion index (is {index}) should be <= len (is {len})"
        );
        self.reserve_owned(1);
        let len = self.raw_len();
        let p = self.base_ptr();
        // Safety: shift then write; `ptr::copy` handles overlap.
        unsafe {
            if index < len {
                ptr::copy(p.add(index), p.add(index + 1), len - index);
            }
            p.add(index).write(value);
        }
        self.len = (self.len & HEAP_TAG) | (len + 1);
    }

    /// Removes and returns the element at `index`, shifting later elements
    /// left. Materializes a borrowed cow first (hence the `T: Clone` bound).
    ///
    /// The index is validated *before* materializing. Use
    /// [`Self::remove_owned`] when the cow is known to be owned and `T: Clone`
    /// is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T
    where
        T: Clone,
    {
        let len = self.raw_len();
        assert!(
            index < len,
            "removal index (is {index}) should be < len (is {len})"
        );
        if self.is_borrowed() {
            self.to_mut();
        }
        self.remove_owned(index)
    }

    /// [`Self::remove`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::remove`] — or if `index >= len`.
    #[track_caller]
    pub fn remove_owned(&mut self, index: usize) -> T {
        assert!(
            !self.is_borrowed(),
            "remove_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        assert!(
            index < len,
            "removal index (is {index}) should be < len (is {len})"
        );
        let p = self.base_ptr();
        // Safety: read the hole, then close it with an overlapping copy.
        unsafe {
            let value = p.add(index).read();
            ptr::copy(p.add(index + 1), p.add(index), len - index - 1);
            self.len = (self.len & HEAP_TAG) | (len - 1);
            value
        }
    }

    /// Removes and returns the element at `index`, replacing it with the last
    /// element. Materializes a borrowed cow first (hence the `T: Clone` bound).
    ///
    /// The index is validated *before* materializing. Use
    /// [`Self::swap_remove_owned`] when the cow is known to be owned and
    /// `T: Clone` is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[track_caller]
    pub fn swap_remove(&mut self, index: usize) -> T
    where
        T: Clone,
    {
        let len = self.raw_len();
        assert!(
            index < len,
            "swap_remove index (is {index}) should be < len (is {len})"
        );
        if self.is_borrowed() {
            self.to_mut();
        }
        self.swap_remove_owned(index)
    }

    /// [`Self::swap_remove`] on an **already-owned** cow, without requiring
    /// `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::swap_remove`] — or if `index >= len`.
    #[track_caller]
    pub fn swap_remove_owned(&mut self, index: usize) -> T {
        assert!(
            !self.is_borrowed(),
            "swap_remove_owned on a borrowed CovariantCow"
        );
        let len = self.raw_len();
        assert!(
            index < len,
            "swap_remove index (is {index}) should be < len (is {len})"
        );
        let p = self.base_ptr();
        // Safety: read, then overwrite with the tail (or leave the hole).
        unsafe {
            let value = p.add(index).read();
            if index != len - 1 {
                let last = p.add(len - 1).read();
                p.add(index).write(last);
            }
            self.len = (self.len & HEAP_TAG) | (len - 1);
            value
        }
    }

    /// Converts into a `Vec<T>`, copying only if currently borrowed.
    ///
    /// Owned states (inline/heap) transfer their elements without cloning; only
    /// a borrowed cow needs `T: Clone`. This is the inherent equivalent of the
    /// `From<CovariantCow> for Vec` impl — whose signature cannot drop the
    /// bound, because one impl must cover the borrowed case. Use
    /// [`Self::into_vec_owned`] when the cow is known to be owned and `T: Clone`
    /// is unavailable.
    #[must_use]
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.into()
    }

    /// Converts an **already-owned** cow into a `Vec<T>`, without requiring
    /// `T: Clone`.
    ///
    /// The inline state is copied into a fresh `Vec` allocation; the heap state
    /// transfers its block with no copy.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Self::into_vec`].
    #[must_use]
    #[track_caller]
    pub fn into_vec_owned(self) -> Vec<T> {
        assert!(
            !self.is_borrowed(),
            "into_vec_owned on a borrowed CovariantCow"
        );
        // `ManuallyDrop` so the cow's own `Drop` never runs; ownership of the
        // elements (or the heap block) is transferred to the returned `Vec`.
        let cow = ManuallyDrop::new(self);
        if cow.is_heap() {
            let len = cow.len();
            // Safety: heap state.
            let (ptr, cap) = unsafe { cow.data.heap };
            // Safety: the block was allocated with exactly
            // `Layout::array::<T>(cap)`, which is what `Vec` expects.
            unsafe { Vec::from_raw_parts(ptr.cast::<T>().as_ptr(), len, cap) }
        } else {
            let len = cow.len();
            let mut v = Vec::with_capacity(len);
            // Safety: inline state; first `len` slots initialized, and the cow
            // is in `ManuallyDrop` so the elements are moved, not double-dropped.
            unsafe {
                ptr::copy_nonoverlapping(cow.base_ptr_const(), v.as_mut_ptr(), len);
                v.set_len(len);
            }
            v
        }
    }

    /// Consumes an **already-owned** cow into an owning iterator, without
    /// requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use `IntoIterator` (which materializes automatically, at the
    /// cost of a `T: Clone` bound).
    #[must_use]
    #[track_caller]
    pub fn into_iter_owned(self) -> CowIntoIter<'a, T, N> {
        assert!(
            !self.is_borrowed(),
            "into_iter_owned on a borrowed CovariantCow"
        );
        CowIntoIter::new(self)
    }

    /// Extends an **already-owned** cow, without requiring `T: Clone`.
    ///
    /// # Panics
    ///
    /// Panics if the cow is in the borrowed state — call [`Self::to_mut`]
    /// first, or use [`Extend`].
    #[track_caller]
    pub fn extend_owned<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        assert!(
            !self.is_borrowed(),
            "extend_owned on a borrowed CovariantCow"
        );
        let mut it = iter.into_iter();
        // `size_hint().0` is trusted as a reservation, exactly as `Vec::extend`
        // does (deliberate std parity).
        let (lo, _) = it.size_hint();
        self.reserve_owned(lo);
        if size_of::<T>() == 0 {
            // A ZST never allocates; each item only bumps the count (ZST
            // protocol site 1), exactly as in `push_owned`.
            for x in it {
                assert!(!self.is_heap(), "ZST cows are never heap-backed");
                assert!(self.raw_len() < MAX_LEN, "capacity overflow");
                self.len = self.raw_len() + 1;
                mem::forget(x);
            }
            return;
        }
        // Heap fast path, mirroring `CovariantVec::extend_from_iter`: the
        // block pointer has provenance in the independent allocation, so the
        // per-element length publication (a write to `self.len`, a different
        // location) does not invalidate it under Stacked Borrows, and both
        // pointer and capacity stay in registers. A spill inside the general
        // loop (`push_owned`) re-enters this path for the remainder.
        loop {
            if self.is_heap() && !self.extend_heap_fast_owned(&mut it) {
                return;
            }
            // General loop (inline state, or the exhausted-heap remainder):
            // per-element re-derivation — the length publication invalidates a
            // hoisted pointer when the elements live inside `self` — and
            // `push_owned` may spill.
            let mut spilled = false;
            for x in it.by_ref() {
                let len = self.raw_len();
                if len < self.capacity() {
                    // Safety: owned state, `len < capacity`, slot `[len]` is in
                    // bounds and uninitialized; `write` moves `x` in without
                    // dropping it. Owned states have `BORROWED_TAG == 0`, so
                    // preserving `HEAP_TAG` yields the right tag.
                    unsafe {
                        let p = if self.is_heap() {
                            self.data.heap.0.cast::<T>().as_ptr().add(len)
                        } else {
                            addr_of_mut!(self.data.inline).cast::<T>().add(len)
                        };
                        p.write(x);
                    }
                    self.len = (self.len & HEAP_TAG) | (len + 1);
                } else {
                    // Spare capacity exhausted; `push_owned` spills/reallocates.
                    self.push_owned(x);
                    if self.is_heap() {
                        spilled = true;
                        break;
                    }
                }
            }
            if !spilled {
                return;
            }
        }
    }

    /// The heap fast loop shared by entry and post-spill re-entry: hoists the
    /// block pointer, capacity and length so the per-element work is one
    /// store and one length publication. Returns `true` when capacity was
    /// exhausted, `false` when the iterator finished.
    #[inline]
    fn extend_heap_fast_owned<I: Iterator<Item = T>>(&mut self, it: &mut I) -> bool {
        debug_assert!(self.is_heap());
        let base = self.base_ptr();
        let cap = self.heap_cap();
        let mut len = self.raw_len();
        while len < cap {
            let Some(x) = it.next() else { return false };
            // Safety: heap state with `len < cap`, so slot `len` is in bounds
            // and uninitialized; `write` moves `x` in without dropping it.
            // `base` stays valid across the publication below (independent
            // block; see above).
            unsafe { base.add(len).write(x) };
            len += 1;
            // Published before the next `it.next()` (user code, may panic).
            // Owned states have `BORROWED_TAG == 0`, so `HEAP_TAG | len` is
            // the tag-consistent form.
            self.len = HEAP_TAG | len;
        }
        true
    }

    /// # Safety
    ///
    /// Must only be called from `Drop` (terminal position). Reads no element
    /// contents (the `may_dangle` contract).
    ///
    /// A **borrowed** cow owns no storage at all: the slice belongs to the
    /// borrower, so dropping the cow must neither drop any element nor free any
    /// block — it is a pure no-op. (This is why the borrowed state can be
    /// dropped even under `#[may_dangle]`: nothing reachable through `'a` is
    /// touched.)
    unsafe fn drop_contents(&mut self) {
        if self.is_borrowed() {
            return;
        }
        let len = self.raw_len();
        if self.is_heap() {
            // Safety: heap state.
            let (ptr, cap) = unsafe { self.data.heap };
            if mem::needs_drop::<T>() {
                // Safety: first `len` heap slots are initialized.
                let p = ptr.cast::<T>().as_ptr();
                unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len)) };
            }
            debug_assert!(size_of::<T>() != 0);
            // Safety: layout matches the allocation.
            unsafe {
                dealloc(ptr.as_ptr(), crate::heap_layout::<T>(cap));
            }
        } else if mem::needs_drop::<T>() {
            // Safety: inline state; first `len` slots are initialized.
            let p = addr_of_mut!(self.data.inline).cast::<T>();
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len)) };
        }
    }
}

// --------------------------------------------------------------------------
// Drop
// --------------------------------------------------------------------------

#[cfg(feature = "dropck_eyepatch")]
// Safety: as for the two-state type — the body reads only the tag and never
// reads any `T` nor anything reachable through `'a` (the borrowed slice is not
// even touched). `#[may_dangle]` on both `'a` and `T` is therefore sound.
unsafe impl<#[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for CovariantCow<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_contents() }
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'a, T, const N: usize> Drop for CovariantCow<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_contents() }
    }
}

// --------------------------------------------------------------------------
// Auto traits
// --------------------------------------------------------------------------

// Safety: owned states move `T` (Send needs `T: Send`); the borrowed state
// shares `&[T]` across threads (Send and Sync both need `T: Sync`).
unsafe impl<'a, T: Send + Sync, const N: usize> Send for CovariantCow<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for CovariantCow<'a, T, N> {}

// --------------------------------------------------------------------------
// Clone
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> CloneSink for CovariantCow<'a, T, N> {
    type Elem = T;

    #[inline]
    unsafe fn raw_base(&mut self) -> *mut T {
        // Safe operation: `base_ptr_owned` dispatches on the tag, uses place
        // access, and only `debug_assert!`s the non-borrowed state the caller
        // promises (the `CloneSink` contract). Using the real-assert twin here
        // would re-check the tag once per element in the hot clone loops.
        unsafe { self.base_ptr_owned() }
    }

    #[inline]
    unsafe fn publish_len(&mut self, n: usize) {
        // Safety: `n` slots were just initialized, and `n <= capacity`. The
        // fill only ever runs in an owned state, which the primitive
        // `debug_assert!`s.
        unsafe { self.publish_len_owned(n) };
    }
}

impl<'a, T: Clone, const N: usize> Clone for CovariantCow<'a, T, N> {
    fn clone(&self) -> Self {
        if self.is_borrowed() {
            // Safety: reading the borrowed variant.
            return CovariantCow::from_borrowed(unsafe { self.data.borrowed });
        }
        let len = self.raw_len();
        let mut out = Self::new();
        // Normalize the state: a clone that fits inline (or is a ZST) is always
        // inline, even from a spilled-and-popped heap source; the heap case
        // allocates exactly `len` (not the source's larger capacity). The fill
        // is `clone_fill::CloneFill`: the element-wise loop (length published
        // per element, so a panic inside `T::clone` leaves `out` holding
        // exactly the clones made so far, which its `Drop` cleans up) or — under
        // the nightly-only `specialization` feature — a single
        // `ptr::copy_nonoverlapping` for `T: Copy`. See that module for the
        // Stacked-Borrows and panic-safety rationale. For a ZST the fill runs
        // `T::clone` once per element (no bytes move, but the calls happen);
        // contrast `to_mut`, which adopts the count without cloning — the
        // asymmetry is documented there.
        //
        // The *source* pointer is hoisted: `self` is never mutated during the
        // clone, so its provenance is stable.
        let src = self.base_ptr_const();
        if size_of::<T>() == 0 || len <= N {
            // Safety: `src` is valid for `len` reads (`self` may be inline or
            // heap; `base_ptr_const` handles both; ZST offsets are no-ops);
            // `out` is owned, empty and has capacity `N >= len`.
            unsafe { T::clone_fill(src, &mut out, len) };
        } else {
            // Invariant: ZSTs never reach the heap state.
            debug_assert!(size_of::<T>() != 0);
            out.alloc_heap(len);
            // Safety: as above, with `out` freshly heap-allocated with
            // capacity exactly `len`.
            unsafe { T::clone_fill(src, &mut out, len) };
        }
        out
    }

    fn clone_from(&mut self, source: &Self) {
        let len = source.raw_len();
        // Reuse the destination's storage only when *both* cows are owned and
        // the destination already fits the source length. A borrowed source is
        // adopted cheaply (zero-copy) and a borrowed destination is dropped in
        // favor of a fresh state — both handled by the `clone` fallback, which
        // allocates exactly `len` in the owned case. The `heap_cap` probe is
        // short-circuited away for an inline target, so it never reads the
        // heap variant of an inline container's union.
        let reuse = !self.is_borrowed()
            && !source.is_borrowed()
            && (size_of::<T>() == 0 || len <= N || (self.is_heap() && self.heap_cap() >= len));
        if !reuse {
            *self = source.clone();
            return;
        }
        // Drop the old contents first: a panic inside `T::clone` then leaves
        // `self` holding exactly the clones made so far, which `Drop` cleans
        // up — no double drops. The element-wise fill publishes the length per
        // element, so the partially-filled prefix is valid (though possibly
        // not all-cloned) mid-panic; under `specialization` the `T: Copy` fill
        // publishes once, which is sound because nothing user-visible can run
        // between the copy and the publication.
        //
        // The ZST case: `clone_fill` reads and writes no bytes, but it still
        // runs `T::clone` `len` times — deliberate (callers count those
        // clones), and the one place this crate's ZST handling *does not*
        // adopt the count. Contrast `to_mut`, which adopts without cloning;
        // the asymmetry is documented there.
        self.clear();
        // Match `clone`'s state normalization: a result that fits inline must
        // be inline, not a reused heap block (see the `CovariantVec` twin).
        // The length is 0 after `clear`, so this moves no elements.
        if len <= N {
            self.shrink_to_inline();
        }
        // Safety: `src` is valid for `len` reads (`self` is owned here, inline
        // or heap; `base_ptr_const` handles both; ZST offsets are no-ops), and
        // `self` is empty after `clear` with capacity >= `len` (inline `N`,
        // the never-spilling ZST case, or a heap block checked above).
        let src = source.base_ptr_const();
        unsafe { T::clone_fill(src, self, len) };
    }
}

// --------------------------------------------------------------------------
// Deref / DerefMut
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> Deref for CovariantCow<'a, T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T: Clone, const N: usize> DerefMut for CovariantCow<'a, T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.to_mut()
    }
}

// --------------------------------------------------------------------------
// Forwarding impls
// --------------------------------------------------------------------------

impl<'a, T: fmt::Debug, const N: usize> fmt::Debug for CovariantCow<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<'a, T: PartialEq, const N: usize> PartialEq for CovariantCow<'a, T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T, B, const N: usize, const M: usize> PartialEq<[B; M]> for CovariantCow<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &[B; M]) -> bool {
        self.as_slice() == other
    }
}

impl<'a, 'b, T, B, const N: usize> PartialEq<&'b [B]> for CovariantCow<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &&'b [B]) -> bool {
        self.as_slice() == *other
    }
}

impl<'a, T, B, const N: usize> PartialEq<Vec<B>> for CovariantCow<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &Vec<B>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T, const N: usize> AsRef<[T]> for CovariantCow<'a, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, const N: usize> AsMut<[T]> for CovariantCow<'a, T, N>
where
    T: Clone,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        // Mirrors `DerefMut`: mutable access copy-on-writes a borrowed cow.
        self.to_mut()
    }
}

impl<'a, T, const N: usize> Borrow<[T]> for CovariantCow<'a, T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T: Clone, const N: usize> BorrowMut<[T]> for CovariantCow<'a, T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.to_mut()
    }
}

impl<'a, T: Eq, const N: usize> Eq for CovariantCow<'a, T, N> {}

impl<'a, T: PartialOrd, const N: usize> PartialOrd for CovariantCow<'a, T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, T: Ord, const N: usize> Ord for CovariantCow<'a, T, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, T: Hash, const N: usize> Hash for CovariantCow<'a, T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<'a, T, const N: usize> Default for CovariantCow<'a, T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, const N: usize> From<&'a [T]> for CovariantCow<'a, T, N> {
    fn from(slice: &'a [T]) -> Self {
        // No `T: Clone` bound: `from_borrowed` stores the slice zero-copy and
        // never clones.
        Self::from_borrowed(slice)
    }
}

impl<'a, T, const N: usize, const M: usize> From<&'a [T; M]> for CovariantCow<'a, T, N> {
    fn from(array: &'a [T; M]) -> Self {
        // Borrows the array zero-copy, like `From<&[T]>` (an array reference
        // unsize-coerces to a slice reference).
        Self::from_borrowed(array)
    }
}

impl<'a, T: Clone, const N: usize> From<CovariantCow<'a, T, N>> for Vec<T> {
    fn from(cow: CovariantCow<'a, T, N>) -> Self {
        if cow.is_borrowed() {
            // Not owned: copy out (this is the only reason the impl needs
            // `T: Clone`). A borrowed cow must stay borrowed here, so this is a
            // plain slice copy rather than a materialize-then-move.
            return cow.as_slice().to_vec();
        }
        // Owned states: shared with the no-`Clone` `into_vec_owned`.
        cow.into_vec_owned()
    }
}

// --------------------------------------------------------------------------
// Conversions from Vec
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> From<Vec<T>> for CovariantCow<'a, T, N> {
    #[track_caller]
    fn from(vec: Vec<T>) -> Self {
        // The heap branch below bypasses `new()`, so repeat the precondition.
        const {
            assert!(
                N <= MAX_LEN,
                "N too large for CovariantCow (length field is 62 bits)"
            )
        };
        let len = vec.len();

        if size_of::<T>() == 0 {
            // ZST protocol (site 1/2): take the count only; see the crate-level
            // "ZST ownership protocol". `MAX_LEN` (not `isize::MAX`) is the
            // bound here, since the cow spends bit 62 on the borrowed tag.
            assert!(len <= MAX_LEN, "capacity overflow");
            mem::forget(vec);
            let mut out = Self::new();
            out.len = len;
            return out;
        }

        // The length field is only 62 bits wide, and `HEAP_TAG | len` sets
        // `BORROWED_TAG` (bit 62) as soon as `len > MAX_LEN`, making an owned
        // heap cow look borrowed. This is reachable on 32-bit targets, where
        // `MAX_LEN` is `2^30 - 1`: a `Vec<u8>` of 1 GiB already exceeds it.
        // Keep this guard (and the ZST one above) in sync with
        // `CovariantVec::from(Vec)`; only the bound differs (`MAX_LEN` here,
        // `isize::MAX` there, because that encoding spends only bit 63).
        assert!(len <= MAX_LEN, "capacity overflow");

        if vec.capacity() > N {
            // Take over the allocation (cap may be <= N only when `N == 0`,
            // which cannot happen since cap > N). A capacity above `MAX_LEN`
            // would also make the cow a trap — `grow_to` clamps to `MAX_LEN`,
            // so the next `push` would panic despite spare capacity — and
            // every other entry point rejects it; keep the invariant uniform.
            assert!(vec.capacity() <= MAX_LEN, "capacity overflow");
            let mut v = ManuallyDrop::new(vec);
            let ptr = v.as_mut_ptr();
            CovariantCow {
                data: Data3 {
                    heap: (
                        unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) },
                        v.capacity(),
                    ),
                },
                len: HEAP_TAG | len,
            }
        } else {
            // Copy back into the inline buffer and free the Vec's buffer.
            let mut out = Self::new();
            // Safety: inline slots [0, len) are uninitialized.
            unsafe {
                let dst = addr_of_mut!(out.data.inline).cast::<T>();
                ptr::copy_nonoverlapping(vec.as_ptr(), dst, len);
            }
            out.len = len;
            let mut v = vec;
            // The elements now live in `out`; stop `v` from dropping them.
            unsafe { v.set_len(0) };
            drop(v);
            out
        }
    }
}

impl<'a, T, const N: usize, const M: usize> From<[T; M]> for CovariantCow<'a, T, N> {
    fn from(arr: [T; M]) -> Self {
        let mut out = Self::new();
        out.extend_owned(arr);
        out
    }
}

// --------------------------------------------------------------------------
// Iteration / collection traits
// --------------------------------------------------------------------------

/// Extending a **borrowed** cow materializes it on the amortized growth curve,
/// so peak capacity can be up to ~2× the final length. For exact sizing, call
/// [`reserve_exact`](Self::reserve_exact) with the iterator's lower bound and
/// then [`extend_owned`](Self::extend_owned).
impl<'a, T: Clone, const N: usize> Extend<T> for CovariantCow<'a, T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // Owned: no peek and no split reservation — `extend_owned` reserves the
        // hint once and uses the heap fast path. (The peek exists only so that
        // `extend([])` on a *borrowed* cow does not materialize it.)
        if !self.is_borrowed() {
            self.extend_owned(iter);
            return;
        }
        let mut it = iter.into_iter();
        let (lo, _) = it.size_hint();
        if lo > 0 {
            // Reserve before consuming anything: a capacity-overflow panic
            // must not leave an element already pulled (and dropped) from the
            // iterator. This mirrors `Vec::extend`.
            self.materialize_with_min_capacity(lo);
            self.extend_owned(it);
            return;
        }
        let Some(first) = it.next() else {
            return;
        };
        self.materialize_with_min_capacity(1);
        self.push_owned(first);
        self.extend_owned(it);
    }
}

impl<'a, T, const N: usize> core::iter::FromIterator<T> for CovariantCow<'a, T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // A fresh cow is owned, so no `T: Clone` bound is needed.
        let mut out = Self::new();
        out.extend_owned(iter);
        out
    }
}

// `T: Copy` by-reference variants, mirroring `Vec`'s own impls.
impl<'a, 'b, T: Copy, const N: usize> Extend<&'b T> for CovariantCow<'a, T, N> {
    fn extend<I: IntoIterator<Item = &'b T>>(&mut self, iter: I) {
        // As in the `Extend<T>` impl: owned cows go straight to the fused
        // reserve + heap fast path; only a borrowed cow needs the peek.
        if !self.is_borrowed() {
            self.extend_owned(iter.into_iter().copied());
            return;
        }
        let mut it = iter.into_iter().copied();
        let (lo, _) = it.size_hint();
        if lo > 0 {
            self.materialize_with_min_capacity(lo);
            self.extend_owned(it);
            return;
        }
        let Some(first) = it.next() else {
            return;
        };
        self.materialize_with_min_capacity(1);
        self.push_owned(first);
        self.extend_owned(it);
    }
}

impl<'a, 'b, T: Copy, const N: usize> core::iter::FromIterator<&'b T> for CovariantCow<'a, T, N> {
    fn from_iter<I: IntoIterator<Item = &'b T>>(iter: I) -> Self {
        let mut out = Self::new();
        out.extend(iter);
        out
    }
}

impl<'a, 'r, T, const N: usize> IntoIterator for &'r CovariantCow<'a, T, N> {
    type Item = &'r T;
    type IntoIter = slice::Iter<'r, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, 'r, T: Clone, const N: usize> IntoIterator for &'r mut CovariantCow<'a, T, N> {
    type Item = &'r mut T;
    type IntoIter = slice::IterMut<'r, T>;
    fn into_iter(self) -> Self::IntoIter {
        // Mirrors `DerefMut`: mutable iteration copy-on-writes a borrowed cow,
        // hence the `T: Clone` bound (rather than panicking like `as_mut_slice`).
        self.to_mut().iter_mut()
    }
}

impl<'a, T: Clone, const N: usize> IntoIterator for CovariantCow<'a, T, N> {
    type Item = T;
    type IntoIter = CowLazyIntoIter<'a, T, N>;
    fn into_iter(self) -> Self::IntoIter {
        // A borrowed cow is consumed **lazily**: the iterator clones elements
        // from the borrowed slice on demand, so `into_iter().next()` costs one
        // clone and no allocation. Owned states are handed straight to the
        // iterator with no allocation or copy. For an owned cow of a
        // non-`Clone` element use `into_iter_owned` directly.
        if self.is_borrowed() {
            // Safety: reading the borrowed variant; the slice points at
            // external memory and outlives the iterator.
            let slice: &'a [T] = unsafe { self.data.borrowed };
            CowLazyIntoIter::from_borrowed(slice)
        } else {
            CowLazyIntoIter::from_owned(self)
        }
    }
}

// --------------------------------------------------------------------------
// CowIntoIter
// --------------------------------------------------------------------------

/// Consuming iterator over an **owned** [`CovariantCow`].
///
/// Created by [`CovariantCow::into_iter_owned`]. [`IntoIterator`] returns the
/// lazy [`CowLazyIntoIter`] instead, so a borrowed cow is not materialized
/// just to iterate it. Unlike routing through a `Vec`, this moves elements
/// directly out of the cow's inline or heap storage, so neither owned state
/// allocates or copies on consumption.
pub struct CowIntoIter<'a, T, const N: usize> {
    cow: ManuallyDrop<CovariantCow<'a, T, N>>,
    head: usize,
    tail: usize,
}

impl<'a, T, const N: usize> CowIntoIter<'a, T, N> {
    fn new(cow: CovariantCow<'a, T, N>) -> Self {
        debug_assert!(!cow.is_borrowed());
        let tail = cow.len();
        CowIntoIter {
            cow: ManuallyDrop::new(cow),
            head: 0,
            tail,
        }
    }

    /// The not-yet-consumed elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let n = self.tail - self.head;
        // Safety: [head, tail) are initialized.
        unsafe { slice::from_raw_parts(self.cow.base_ptr_const().add(self.head), n) }
    }

    /// The not-yet-consumed elements, mutably.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let n = self.tail - self.head;
        // Safety: [head, tail) are initialized; `&mut self` excludes aliases,
        // and the iterator's constructor materialized the cow (owned state).
        unsafe { slice::from_raw_parts_mut(self.cow.base_ptr_owned().add(self.head), n) }
    }

    /// Drops the not-yet-consumed elements `[head, tail)` and frees the heap
    /// block if present.
    ///
    /// # Safety
    ///
    /// Must only be called from `Drop` (terminal position). Reads no element
    /// contents (the `may_dangle` contract).
    unsafe fn drop_remaining(&mut self) {
        let remaining = self.tail - self.head;
        if mem::needs_drop::<T>() && remaining > 0 {
            // Safety: [head, tail) are still initialized and unused. ZST
            // protocol (site 2): for a ZST these slots were never written, but
            // the drop glue touches no bytes and runs once per counted element.
            // Safety: owned state (the constructor materializes).
            let p = unsafe { self.cow.base_ptr_owned().add(self.head) };
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, remaining)) };
        }
        if self.cow.is_heap() {
            // Safety: heap state; layout matches the allocation.
            let (ptr, cap) = unsafe { self.cow.data.heap };
            debug_assert!(size_of::<T>() != 0);
            unsafe {
                dealloc(ptr.as_ptr(), crate::heap_layout::<T>(cap));
            }
        }
    }
}

impl<'a, T, const N: usize> Iterator for CowIntoIter<'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        let idx = self.head;
        self.head = idx + 1;
        // Safety: slot [idx] is initialized. The pointer is derived *after*
        // the cursor write (the two share this iterator's allocation in the
        // inline state) so the ordering never relies on Stacked Borrows'
        // per-location granularity; `next_back` already works this way.
        // Safety: owned state (the constructor materializes).
        Some(unsafe { self.cow.base_ptr_owned().add(idx).read() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.tail - self.head;
        (n, Some(n))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for CowIntoIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        self.tail -= 1;
        // Safety: slot [tail] is initialized.
        // Safety: owned state (the constructor materializes).
        let p = unsafe { self.cow.base_ptr_owned().add(self.tail) };
        Some(unsafe { p.read() })
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for CowIntoIter<'a, T, N> {}
impl<'a, T, const N: usize> FusedIterator for CowIntoIter<'a, T, N> {}

// Auto traits: both this iterator and `CowDrain` hold an **owned** cow by
// value — the constructors eagerly materialize a borrowed cow (the drain
// constructor likewise, whenever the range is non-empty; an empty-range drain
// keeps a throwaway owned cow). Nothing in the type can observe the `'a`
// borrow's pointee afterwards, so the pointee's `Sync` (which the borrowed
// *cow* needs for `Send`) is not required to move the iterator between
// threads; the `T: Send` bound below matches the owned cow's own `Send`.
//
// Safety: the only `unsafe` data the iterator touches is its owned cow's
// storage (transitively `Send`/`Sync` when `T` is), and the invariant is
// pinned by `CowIntoIter::new`/`CowDrain`'s `debug_assert!(!is_borrowed())`.
unsafe impl<'a, T: Send, const N: usize> Send for CowIntoIter<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for CowIntoIter<'a, T, N> {}

impl<'a, T: fmt::Debug, const N: usize> fmt::Debug for CowIntoIter<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CowIntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

// Same drop obligations as the container's own `IntoIter` (reads only the tag,
// drops elements, deallocates), so the same `#[may_dangle]` annotations apply.
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<#[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for CowIntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'a, T, const N: usize> Drop for CowIntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

// --------------------------------------------------------------------------
// CowLazyIntoIter
// --------------------------------------------------------------------------

/// Consuming iterator returned by [`IntoIterator`] for a [`CovariantCow`].
///
/// For an **owned** cow this is a thin wrapper around [`CowIntoIter`] with no
/// allocation or copy. For a **borrowed** cow it is lazy: elements are cloned
/// out of the borrowed slice on demand, so `into_iter().next()` costs one
/// clone (and no allocation) rather than cloning the whole slice up front.
/// Only the still-unconsumed elements are cloned; dropping the iterator drops
/// nothing (the borrower keeps its slice).
///
/// `Send` needs `T: Send + Sync` because the borrowed variant holds a shared
/// `&'a [T]`; use [`CowIntoIter`] (via [`CovariantCow::into_iter_owned`]) for
/// an owned `T: Send + !Sync` iterator.
pub struct CowLazyIntoIter<'a, T, const N: usize> {
    inner: CowLazyIntoIterInner<'a, T, N>,
    head: usize,
    tail: usize,
}

enum CowLazyIntoIterInner<'a, T, const N: usize> {
    Borrowed(&'a [T]),
    Owned(ManuallyDrop<CovariantCow<'a, T, N>>),
}

impl<'a, T, const N: usize> CowLazyIntoIter<'a, T, N> {
    fn from_owned(cow: CovariantCow<'a, T, N>) -> Self {
        debug_assert!(!cow.is_borrowed());
        let tail = cow.len();
        CowLazyIntoIter {
            inner: CowLazyIntoIterInner::Owned(ManuallyDrop::new(cow)),
            head: 0,
            tail,
        }
    }

    fn from_borrowed(slice: &'a [T]) -> Self {
        CowLazyIntoIter {
            inner: CowLazyIntoIterInner::Borrowed(slice),
            head: 0,
            tail: slice.len(),
        }
    }

    /// Switches a borrowed iterator to owned storage containing only the
    /// not-yet-consumed elements (cloning them). No-op for the owned variant.
    fn materialize(&mut self)
    where
        T: Clone,
    {
        let s: &'a [T] = match &self.inner {
            CowLazyIntoIterInner::Borrowed(s) => s,
            CowLazyIntoIterInner::Owned(_) => return,
        };
        let rem = &s[self.head..self.tail];
        let mut cow = CovariantCow::new();
        if !rem.is_empty() {
            cow.reserve_owned(rem.len());
            // Safety: `cow` is owned, empty and has capacity for `rem.len()`;
            // `rem` is a distinct borrowed slice.
            unsafe { T::clone_fill(rem.as_ptr(), &mut cow, rem.len()) };
        }
        self.inner = CowLazyIntoIterInner::Owned(ManuallyDrop::new(cow));
        self.head = 0;
        self.tail = rem.len();
    }

    /// The not-yet-consumed elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            CowLazyIntoIterInner::Borrowed(s) => &s[self.head..self.tail],
            CowLazyIntoIterInner::Owned(cow) => {
                let n = self.tail - self.head;
                // Safety: [head, tail) are initialized.
                unsafe { slice::from_raw_parts(cow.base_ptr_const().add(self.head), n) }
            }
        }
    }

    /// The not-yet-consumed elements, mutably. A borrowed iterator is
    /// materialized into owned storage first (cloning the remaining
    /// elements); an owned one needs no work.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        self.materialize();
        let n = self.tail - self.head;
        match &mut self.inner {
            CowLazyIntoIterInner::Owned(cow) => {
                // Safety: [head, tail) are initialized; `&mut self` excludes
                // aliases and the variant is owned.
                unsafe { slice::from_raw_parts_mut(cow.base_ptr_owned().add(self.head), n) }
            }
            // `materialize` always leaves the owned variant.
            CowLazyIntoIterInner::Borrowed(_) => unreachable!(),
        }
    }

    /// Drops/frees only the owned variant's remaining elements.
    ///
    /// # Safety
    ///
    /// Must only be called from `Drop` (terminal position).
    unsafe fn drop_remaining(&mut self) {
        let CowLazyIntoIterInner::Owned(cow) = &mut self.inner else {
            // Borrowed: the borrower owns the slice; nothing to drop.
            return;
        };
        let remaining = self.tail - self.head;
        if mem::needs_drop::<T>() && remaining > 0 {
            // Safety: [head, tail) are still initialized and unused; owned
            // variant (the borrowed one never reaches owned storage).
            let p = unsafe { cow.base_ptr_owned().add(self.head) };
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, remaining)) };
        }
        if cow.is_heap() {
            // Safety: heap state; layout matches the allocation.
            let (ptr, cap) = unsafe { cow.data.heap };
            debug_assert!(size_of::<T>() != 0);
            unsafe {
                dealloc(ptr.as_ptr(), crate::heap_layout::<T>(cap));
            }
        }
    }
}

impl<'a, T: Clone, const N: usize> Iterator for CowLazyIntoIter<'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        let idx = self.head;
        self.head = idx + 1;
        match &mut self.inner {
            CowLazyIntoIterInner::Borrowed(s) => Some(s[idx].clone()),
            CowLazyIntoIterInner::Owned(cow) => {
                // Safety: slot [idx] is initialized; pointer derived after the
                // cursor write (Stacked Borrows, as in `CowIntoIter::next`).
                Some(unsafe { cow.base_ptr_owned().add(idx).read() })
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.tail - self.head;
        (n, Some(n))
    }
}

impl<'a, T: Clone, const N: usize> DoubleEndedIterator for CowLazyIntoIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        self.tail -= 1;
        let idx = self.tail;
        match &mut self.inner {
            CowLazyIntoIterInner::Borrowed(s) => Some(s[idx].clone()),
            CowLazyIntoIterInner::Owned(cow) => {
                // Safety: slot [idx] is initialized.
                Some(unsafe { cow.base_ptr_owned().add(idx).read() })
            }
        }
    }
}

impl<'a, T: Clone, const N: usize> ExactSizeIterator for CowLazyIntoIter<'a, T, N> {}
impl<'a, T: Clone, const N: usize> FusedIterator for CowLazyIntoIter<'a, T, N> {}

// Auto traits: the owned variant moves `T` (Send needs `T: Send`), but the
// borrowed variant shares `&'a [T]` across threads (Send and Sync both need
// `T: Sync`). The stricter combined bound is required because the variant is
// a runtime property; `into_iter_owned` is the owned-only escape hatch.
unsafe impl<'a, T: Send + Sync, const N: usize> Send for CowLazyIntoIter<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for CowLazyIntoIter<'a, T, N> {}

impl<'a, T: fmt::Debug, const N: usize> fmt::Debug for CowLazyIntoIter<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CowLazyIntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

#[cfg(feature = "dropck_eyepatch")]
unsafe impl<#[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for CowLazyIntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'a, T, const N: usize> Drop for CowLazyIntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

// --------------------------------------------------------------------------
// CowDrain
// --------------------------------------------------------------------------

/// A draining iterator over a [`CovariantCow`].
///
/// Created by [`CovariantCow::drain_owned`] and used internally by
/// `CowSplice`. A borrowed cow is drained through the lazy [`CowDrainLazy`]
/// instead. For a **non-empty** range the whole container is moved out and
/// restored on drop (even while unwinding), so the tail is shifted back and
/// the length restored; the container is always owned while such a `CowDrain`
/// exists. For an **empty** range nothing is taken: the `CowDrain` is inert,
/// and a borrowed cow is left borrowed.
pub struct CowDrain<'r, 'a, T, const N: usize> {
    cow: ManuallyDrop<CovariantCow<'a, T, N>>,
    slot: &'r mut CovariantCow<'a, T, N>,
    /// `false` only for an empty range (the container was never taken).
    owns: bool,
    begin: usize,
    pos: usize,
    end: usize,
    back: usize,
    original_len: usize,
}

impl<'r, 'a, T, const N: usize> CowDrain<'r, 'a, T, N> {
    /// The not-yet-yielded elements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let n = self.back - self.pos;
        // Safety: `[pos, back)` is an initialized sub-range; the storage is
        // unchanged for the drain's whole lifetime.
        unsafe { slice::from_raw_parts(self.cow.base_ptr_const().add(self.pos), n) }
    }

    /// The not-yet-yielded elements as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let n = self.back - self.pos;
        // Safety: as `as_slice`, plus `&mut self` excludes aliases, and the
        // drain constructor materialized the cow (owned state).
        unsafe { slice::from_raw_parts_mut(self.cow.base_ptr_owned().add(self.pos), n) }
    }

    /// Keeps every not-yet-yielded element in the container, so only the
    /// already-yielded ones are removed (std parity with the unstable
    /// `Vec::Drain::keep_rest`).
    ///
    /// This is the safe way to abandon a `drain` early — the opposite trade
    /// from `mem::forget(drain)`, which leaks the *whole* container. The cow
    /// is rolled back into the caller with exactly `begin + kept + tail`
    /// elements, where `kept` is the number of elements the iterator has not
    /// yielded yet. Already-yielded elements stay the caller's responsibility:
    /// they were moved out by `next`/`next_back` and are never dropped here.
    pub fn keep_rest(&mut self) {
        if !self.owns {
            // Empty range: the container was never taken; nothing to keep.
            return;
        }
        // Shift the unyielded run to the start of the drained region, over the
        // slots of the elements already yielded (which the caller now owns).
        let kept = self.back - self.pos;
        // `pos == begin` (only `next_back` has been used) makes this a
        // self-move; skip the wasted O(N) memmove. The rollback below is
        // unchanged — it shifts from `end` to `begin + kept == back` either
        // way.
        if self.pos != self.begin {
            // Safety: `[pos, back)` is initialized; `[begin, pos)` holds dead
            // slots (yielded). `ptr::copy` handles the overlap. `base` is
            // derived once and consumed by this single copy before `self.cow`
            // is reborrowed, matching the rule in `CowDrainRollback::drop`.
            unsafe {
                let base = self.cow.base_ptr();
                ptr::copy(base.add(self.pos), base.add(self.begin), kept);
            }
        }
        // Roll the untouched tail back behind the kept run and hand the
        // container to the slot — precisely what `Drop` would do for the range
        // `[begin + kept, end)`.
        let rb = CowDrainRollback {
            cow: &mut self.cow,
            slot: &mut *self.slot,
            begin: self.begin + kept,
            end: self.end,
            original_len: self.original_len,
            pos: self.back,
            back: self.back,
        };
        drop(rb);
        // The rollback moved the cow back into the caller's slot, so
        // `self.cow` is a taken `ManuallyDrop`. `as_slice`/`as_mut_slice` are
        // still callable after `keep_rest`; give them a valid empty cow
        // instead of a moved-out value.
        self.cow = ManuallyDrop::new(CovariantCow::new());
        // The container is back in the caller's slot; `Drop` must do nothing,
        // and the kept elements count as consumed, so the iterator is now
        // exhausted.
        self.owns = false;
        self.pos = self.back;
    }

    /// Shared `Drop` body: drops the not-yet-yielded drained elements, then
    /// rolls the container back into the caller's slot. Panic-safe for the same
    /// reason as [`Drain`](crate::Drain) (the rollback shifts raw bytes and
    /// never runs destructors).
    fn drop_impl(&mut self) {
        if !self.owns {
            // Empty range: nothing was taken; `cow` is a throwaway empty vector.
            return;
        }
        let mut rb = CowDrainRollback {
            cow: &mut self.cow,
            slot: &mut *self.slot,
            begin: self.begin,
            end: self.end,
            original_len: self.original_len,
            pos: self.pos,
            back: self.back,
        };
        if rb.cow.is_heap() {
            // Heap state: hoist the block pointer (provenance is in the
            // independent allocation; the loop only mutates `rb.pos`).
            // Safety: owned heap state (the drain constructor materializes).
            let base = unsafe { rb.cow.base_ptr_owned() };
            while rb.pos < rb.back {
                // Safety: `pos < back <= end <= original_len`; slot `pos` holds
                // an element not yet yielded.
                let idx = rb.pos;
                rb.pos += 1;
                unsafe { ptr::drop_in_place(base.add(idx)) };
            }
        } else {
            while rb.pos < rb.back {
                // Inline state: the elements live inside `rb.cow`, so re-derive
                // per element (per-location Stacked-Borrows rule).
                // Safety: as above.
                let p = unsafe { rb.cow.base_ptr_owned().add(rb.pos) };
                rb.pos += 1;
                unsafe { ptr::drop_in_place(p) };
            }
        }
    }
}

/// Rollback guard for [`CowDrain`]: shifts the tail back over the drained
/// region, restores the length, and moves the container back into the caller's
/// slot. Every step is panic-free, so the rollback always completes.
struct CowDrainRollback<'f, 'a, T, const N: usize> {
    cow: &'f mut ManuallyDrop<CovariantCow<'a, T, N>>,
    slot: &'f mut CovariantCow<'a, T, N>,
    begin: usize,
    end: usize,
    original_len: usize,
    pos: usize,
    back: usize,
}

impl<T, const N: usize> Drop for CowDrainRollback<'_, '_, T, N> {
    fn drop(&mut self) {
        if self.begin != self.end {
            // Safety: `[end, original_len)` is the untouched initialized tail;
            // `[begin, end)` is dead from the container's point of view.
            // `ptr::copy` handles the overlap; `base` is derived once and
            // consumed before `self.cow` is reborrowed again.
            unsafe {
                let base = self.cow.base_ptr();
                ptr::copy(
                    base.add(self.end),
                    base.add(self.begin),
                    self.original_len - self.end,
                );
            }
            let c = &mut **self.cow;
            // Safety: exactly `original_len - (end - begin)` leading slots are
            // initialized (kept prefix + shifted tail); this only shrinks, the
            // cow is owned (the drain constructor materializes), and `set_len`
            // preserves the state tag while its debug assertion pins the
            // capacity bound.
            unsafe { c.set_len(self.original_len - (self.end - self.begin)) };
        }
        if self.cow.is_heap() {
            // Heap state: write back only `(ptr, cap)` + the tagged length,
            // mirroring the state swap in `make_drain` (a whole-struct move
            // would memcpy the dead `N · size_of::<T>()` inline array). The
            // placeholder in the slot is an empty owned inline cow with no
            // resources, so skipping the drop an assignment would run is
            // unobservable.
            //
            // Safety: `self.cow` still owns the block (it was moved, not
            // copied, out of the slot by `make_drain`); this transfers
            // ownership back. `self.cow` is `ManuallyDrop` and never read
            // again, so the block is freed exactly once, by the slot.
            let (ptr, cap) = unsafe { self.cow.data.heap };
            self.slot.data.heap = (ptr, cap);
            self.slot.len = self.cow.len;
        } else {
            // Hand the container back (the placeholder in the slot is an empty
            // inline cow, so overwriting it drops nothing).
            // Safety: `self.cow` is taken exactly once, here.
            *self.slot = unsafe { ManuallyDrop::take(self.cow) };
        }
    }
}

impl<'r, 'a, T, const N: usize> Iterator for CowDrain<'r, 'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.pos == self.back {
            return None;
        }
        // Safety: slot `pos` is initialized and not yet yielded; its slot
        // becomes dead and is overwritten by the rollback without re-dropping.
        // The pointer is derived after the cursor write, matching
        // `CowIntoIter::next`.
        let idx = self.pos;
        self.pos = idx + 1;
        // Safety: owned state (the constructor materializes).
        Some(unsafe { self.cow.base_ptr_owned().add(idx).read() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.back - self.pos;
        (n, Some(n))
    }
}

impl<'r, 'a, T, const N: usize> DoubleEndedIterator for CowDrain<'r, 'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.pos == self.back {
            return None;
        }
        self.back -= 1;
        // Safety: as in `next`; the pointer is derived after the cursor write,
        // and the state is owned (the constructor materializes).
        unsafe {
            let p = self.cow.base_ptr_owned().add(self.back);
            Some(p.read())
        }
    }
}

impl<'r, 'a, T, const N: usize> ExactSizeIterator for CowDrain<'r, 'a, T, N> {}
impl<'r, 'a, T, const N: usize> FusedIterator for CowDrain<'r, 'a, T, N> {}

// Auto traits: a `CowDrain` always owns its cow (the constructor materializes
// a borrowed cow first for any non-empty range; an empty range keeps a
// throwaway owned cow), exactly as for `CowIntoIter` — see the `Send`/`Sync`
// impls there for the invariant and its rationale. Bounds mirror the owned
// cow itself (`T: Send` / `T: Sync`), not the stricter `Send` of a borrowing
// `CovariantCow`.
//
// Safety: as for `CowIntoIter`: the iterator only touches owned storage.
unsafe impl<'r, 'a, T: Send, const N: usize> Send for CowDrain<'r, 'a, T, N> {}
unsafe impl<'r, 'a, T: Sync, const N: usize> Sync for CowDrain<'r, 'a, T, N> {}

impl<'r, 'a, T: fmt::Debug, const N: usize> fmt::Debug for CowDrain<'r, 'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

// Same drop obligations as `Drain` (reads only the tag/length bookkeeping, runs
// `T`'s drop glue, shifts raw bytes, moves the container back — never reading any
// `T` contents nor anything reachable through `'a`), so the same `#[may_dangle]`
// annotations apply.
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<'r, #[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for CowDrain<'r, 'a, T, N> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'r, 'a, T, const N: usize> Drop for CowDrain<'r, 'a, T, N> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

// --------------------------------------------------------------------------
// CowDrainLazy
// --------------------------------------------------------------------------

/// Builds an owned-state [`CowDrain`] for an already-validated `(begin, end)`
/// range. Free function (rather than a method) so callers that own a
/// `&'r mut CovariantCow` binding can move the borrow into the returned drain
/// with its full `'r` lifetime.
fn make_drain_impl<'r, 'a, T, const N: usize>(
    slot: &'r mut CovariantCow<'a, T, N>,
    begin: usize,
    end: usize,
    len: usize,
) -> CowDrain<'r, 'a, T, N> {
    // Contract: a non-empty range requires an owned cow. Taking
    // `slot.data.heap` / `mem::take` from a borrowed cow would silently drop
    // the borrow and hand `Drop` a `*mut T` derived from a shared tag (UB).
    debug_assert!(
        begin == end || !slot.is_borrowed(),
        "make_drain: non-empty range requires an owned cow"
    );
    if begin == end {
        // Empty range: a pure no-op. The container is never taken.
        return CowDrain {
            cow: ManuallyDrop::new(CovariantCow::new()),
            slot,
            owns: false,
            begin,
            pos: begin,
            end,
            back: end,
            original_len: len,
        };
    }
    // As in `CovariantVec::drain`: a heap-backed container is handed over as
    // a state swap (only `(ptr, cap)` + the tagged length) rather than via
    // `mem::take`, which would memcpy the `N · size_of::<T>()` inline array
    // — dead padding in the heap state — and copy it back on rollback.
    let cow = if slot.is_heap() {
        // Safety: owned heap state (`make_drain` is only reached with a
        // materialized cow when the range is non-empty).
        let (ptr, cap) = unsafe { slot.data.heap };
        let raw_len = slot.len;
        // Leave the slot an empty inline cow by writing only `len`; see
        // `CovariantVec::drain` for why the stale union bytes are inert.
        slot.len = 0;
        // Safety: `ptr`/`cap` describe the live block the slot owned and
        // `raw_len` is its tagged length; ownership moves into the returned
        // cow, and `slot` no longer refers to the block.
        CovariantCow {
            data: Data3 { heap: (ptr, cap) },
            len: raw_len,
        }
    } else {
        mem::take(slot)
    };
    CowDrain {
        cow: ManuallyDrop::new(cow),
        slot,
        owns: true,
        begin,
        pos: begin,
        end,
        back: end,
        original_len: len,
    }
}

/// Draining iterator returned by [`CovariantCow::drain`].
///
/// It has two modes. On an **owned** cow it is exactly a [`CowDrain`] (the
/// whole container is moved out and restored on drop). On a **borrowed** cow
/// it is lazy: the kept prefix and suffix are cloned into a fresh owned cow at
/// construction, while the removed elements are cloned from the borrowed slice
/// only as they are yielded. A partial drain therefore never clones the
/// elements it does not yield.
///
/// The borrowed mode is safe for an empty range too: it clones nothing and
/// leaves the cow borrowed.
///
/// `Send` needs `T: Send + Sync` because the borrowed mode holds a shared
/// `&'a [T]`; use [`CovariantCow::drain_owned`] for an owned `T: Send +
/// !Sync` drain.
pub struct CowDrainLazy<'r, 'a, T, const N: usize> {
    inner: CowDrainLazyInner<'r, 'a, T, N>,
}

enum CowDrainLazyInner<'r, 'a, T, const N: usize> {
    Empty,
    Owned(CowDrain<'r, 'a, T, N>),
    Borrowed {
        slot: &'r mut CovariantCow<'a, T, N>,
        src: &'a [T],
        /// Drain start (kept prefix is `[0, begin)`).
        begin: usize,
        /// Drain end (kept suffix is `[end, len)`).
        end: usize,
        /// Next front element to yield.
        pos: usize,
        /// One past the next back element to yield.
        back: usize,
        /// Owned `prefix + suffix`, installed into the slot on drop (or by
        /// `keep_rest`). `None` for the inert empty-range case.
        kept: Option<CovariantCow<'a, T, N>>,
    },
}

impl<'r, 'a, T, const N: usize> CowDrainLazy<'r, 'a, T, N> {
    fn from_owned(drain: CowDrain<'r, 'a, T, N>) -> Self {
        CowDrainLazy {
            inner: CowDrainLazyInner::Owned(drain),
        }
    }

    fn from_borrowed_empty(slot: &'r mut CovariantCow<'a, T, N>, src: &'a [T], at: usize) -> Self {
        CowDrainLazy {
            inner: CowDrainLazyInner::Borrowed {
                slot,
                src,
                begin: at,
                end: at,
                pos: at,
                back: at,
                kept: None,
            },
        }
    }

    fn from_borrowed_kept(
        slot: &'r mut CovariantCow<'a, T, N>,
        src: &'a [T],
        begin: usize,
        end: usize,
        kept: CovariantCow<'a, T, N>,
    ) -> Self {
        CowDrainLazy {
            inner: CowDrainLazyInner::Borrowed {
                slot,
                src,
                begin,
                end,
                pos: begin,
                back: end,
                kept: Some(kept),
            },
        }
    }

    /// Converts a borrowed lazy drain into the fully-owned [`CowDrain`] form,
    /// cloning the *whole* original cow (needed to hand out `&mut [T]` over
    /// the removed region). Used only by `as_mut_slice`, so the lazy path is
    /// not disturbed.
    fn materialize_owned(&mut self)
    where
        T: Clone,
    {
        let inner = mem::replace(&mut self.inner, CowDrainLazyInner::Empty);
        let CowDrainLazyInner::Borrowed {
            slot,
            src,
            begin,
            end,
            pos,
            back,
            kept,
        } = inner
        else {
            // Owned or already materialized: put it back.
            self.inner = inner;
            return;
        };
        drop(kept);
        let len = src.len();
        let mut full = CovariantCow::new();
        full.reserve_owned(len);
        // Safety: `full` is owned, empty and has capacity for `len`;
        // `src` is a distinct initialized borrowed slice.
        unsafe { T::clone_fill(src.as_ptr(), &mut full, len) };
        *slot = full;
        // Move the slot borrow into the drain with its full `'r` lifetime.
        let mut owned = make_drain_impl(slot, begin, end, len);
        // Preserve how much of the removed run was already yielded from each
        // end; the drain drop only reclaims `[pos, back)`.
        owned.pos = pos;
        owned.back = back;
        self.inner = CowDrainLazyInner::Owned(owned);
    }

    /// The not-yet-yielded elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            CowDrainLazyInner::Owned(d) => d.as_slice(),
            CowDrainLazyInner::Borrowed { src, pos, back, .. } => &src[*pos..*back],
            CowDrainLazyInner::Empty => &[],
        }
    }

    /// The not-yet-yielded elements, mutably. The borrowed mode materializes
    /// the whole cow first (cloning it), because the removed elements live in
    /// the borrower's shared slice and cannot be mutated in place.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        self.materialize_owned();
        match &mut self.inner {
            CowDrainLazyInner::Owned(d) => d.as_mut_slice(),
            _ => unreachable!(),
        }
    }

    /// Keeps every not-yet-yielded element in the container (std parity with
    /// `Vec::Drain::keep_rest`). In the borrowed mode the remaining removed
    /// elements are cloned into the kept cow at the insertion point.
    pub fn keep_rest(&mut self)
    where
        T: Clone,
    {
        match &mut self.inner {
            CowDrainLazyInner::Owned(d) => d.keep_rest(),
            CowDrainLazyInner::Borrowed {
                slot,
                src,
                begin,
                pos,
                back,
                kept,
                ..
            } => {
                let Some(mut k) = kept.take() else {
                    // Inert empty-range drain: the container was never taken.
                    return;
                };
                let n = *back - *pos;
                if n > 0 {
                    k.reserve_owned(n);
                    for j in *pos..*back {
                        let at = *begin + (j - *pos);
                        k.insert_owned(at, src[j].clone());
                    }
                }
                // The kept run is now part of the container; the remaining
                // removed elements are no longer yielded.
                *pos = *back;
                **slot = k;
            }
            CowDrainLazyInner::Empty => {}
        }
    }

    fn drop_impl(&mut self) {
        if let CowDrainLazyInner::Borrowed { slot, kept, .. } = &mut self.inner {
            if let Some(k) = kept.take() {
                // Install the owned prefix + suffix; the removed elements are
                // the borrower's and are simply not part of the cow.
                **slot = k;
            }
        }
        // Owned mode: the contained `CowDrain`'s own `Drop` does the work when
        // the enum field is dropped after this body.
    }
}

impl<'r, 'a, T: Clone, const N: usize> Iterator for CowDrainLazy<'r, 'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match &mut self.inner {
            CowDrainLazyInner::Owned(d) => d.next(),
            CowDrainLazyInner::Borrowed { src, pos, back, .. } => {
                if *pos == *back {
                    return None;
                }
                let idx = *pos;
                *pos += 1;
                Some(src[idx].clone())
            }
            CowDrainLazyInner::Empty => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            CowDrainLazyInner::Owned(d) => d.size_hint(),
            CowDrainLazyInner::Borrowed { pos, back, .. } => {
                let n = *back - *pos;
                (n, Some(n))
            }
            CowDrainLazyInner::Empty => (0, Some(0)),
        }
    }
}

impl<'r, 'a, T: Clone, const N: usize> DoubleEndedIterator for CowDrainLazy<'r, 'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        match &mut self.inner {
            CowDrainLazyInner::Owned(d) => d.next_back(),
            CowDrainLazyInner::Borrowed { src, pos, back, .. } => {
                if *pos == *back {
                    return None;
                }
                *back -= 1;
                let idx = *back;
                Some(src[idx].clone())
            }
            CowDrainLazyInner::Empty => None,
        }
    }
}

impl<'r, 'a, T: Clone, const N: usize> ExactSizeIterator for CowDrainLazy<'r, 'a, T, N> {}
impl<'r, 'a, T: Clone, const N: usize> FusedIterator for CowDrainLazy<'r, 'a, T, N> {}

// Auto traits: the owned mode moves `T` (Send needs `T: Send`), but the
// borrowed mode shares `&'a [T]` across threads (Send and Sync both need
// `T: Sync`). The stricter combined bound is required because the mode is a
// runtime property; `drain_owned` is the owned-only entry point.
unsafe impl<'r, 'a, T: Send + Sync, const N: usize> Send for CowDrainLazy<'r, 'a, T, N> {}
unsafe impl<'r, 'a, T: Sync, const N: usize> Sync for CowDrainLazy<'r, 'a, T, N> {}

impl<'r, 'a, T: fmt::Debug, const N: usize> fmt::Debug for CowDrainLazy<'r, 'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

#[cfg(feature = "dropck_eyepatch")]
unsafe impl<'r, #[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop
    for CowDrainLazy<'r, 'a, T, N>
{
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'r, 'a, T, const N: usize> Drop for CowDrainLazy<'r, 'a, T, N> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

// --------------------------------------------------------------------------
// `retain` support (shared with `CovariantVec`)
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> crate::BackshiftTarget for CovariantCow<'a, T, N> {
    type Elem = T;

    #[inline]
    fn backshift_raw_len(&self) -> usize {
        self.raw_len()
    }

    #[inline]
    unsafe fn backshift_base(&mut self) -> *mut T {
        // `base_ptr` asserts the owned state; `retain_mut` materializes first.
        self.base_ptr()
    }

    #[inline]
    unsafe fn backshift_set_len(&mut self, n: usize) {
        // Preserve both tag bits (owned states have `BORROWED_TAG == 0`).
        self.len = (self.len & !LEN_MASK) | n;
    }
}

// --------------------------------------------------------------------------
// CowSplice
// --------------------------------------------------------------------------

/// A splicing iterator over a [`CovariantCow`].
///
/// Created by [`CovariantCow::splice`] / [`CovariantCow::splice_owned`].
/// Yields the removed elements (via [`CowDrain`]) as the iterator is
/// consumed; when the `CowSplice` is dropped, the not-yet-yielded removed
/// elements are dropped and the replacement is spliced in before the
/// [`CowDrain`] rolls the cow back. The mechanics — including the in-place
/// refill of the drained region, the chunked growth for a longer
/// replacement, and the panic behavior — are identical to the two-state
/// [`Splice`](crate::Splice); the only difference is that the container is a
/// three-state cow and is always owned while a non-empty `CowSplice` exists
/// (a borrowed cow is materialized at construction). Like `Splice`, a
/// replacement longer than the range may reallocate, so an inline cow can
/// spill to the heap. An empty-range `CowSplice` leaves a borrowed cow
/// borrowed when the replacement reported an empty upper size hint, so its
/// insert path expects the iterator to stay empty.
///
/// # Panics
///
/// A panic from a removed element's destructor follows [`CowDrain`]'s
/// unwind path (the remaining removed elements are reclaimed without running
/// their destructors). A panic from the replacement iterator keeps every
/// replacement element already written and rolls the tail back over only the
/// still-dead remainder — std's behavior. On the empty-range borrowed path, a
/// replacement that reports an empty upper size hint but then yields panics
/// with `"splice replacement yielded an element after reporting an empty size
/// hint"`; the cow is left unchanged. The replacement is driven from `Drop`,
/// so any of these panics aborts if the `CowSplice` is itself being dropped
/// during unwinding (std parity).
///
/// # Leaking
///
/// Forgetting a non-empty `CowSplice` leaks the whole cow (like forgetting a
/// [`CowDrain`]); an **empty** range never takes the cow, so forgetting it
/// leaks nothing.
pub struct CowSplice<'r, 'a, T, const N: usize, I: Iterator<Item = T>> {
    drain: CowDrain<'r, 'a, T, N>,
    replace_with: I,
}

impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> Iterator for CowSplice<'r, 'a, T, N, I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.drain.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> DoubleEndedIterator
    for CowSplice<'r, 'a, T, N, I>
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.drain.next_back()
    }
}

impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> ExactSizeIterator
    for CowSplice<'r, 'a, T, N, I>
{
}

impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> FusedIterator
    for CowSplice<'r, 'a, T, N, I>
{
}

impl<'r, 'a, T: fmt::Debug, const N: usize, I: Iterator<Item = T>> fmt::Debug
    for CowSplice<'r, 'a, T, N, I>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CowSplice")
            .field(&self.drain.as_slice())
            .finish()
    }
}

/// Fill guard for [`CowSplice`]: the cow twin of `SpliceFill`, tracking the
/// dead gap while the replacement is written and compacting it — publishing
/// the final length — when it drops, on the normal path *and* during
/// unwinding.
///
/// The cow lives in `drain.cow` for a non-empty splice and in `drain.slot`
/// for an empty-range splice (which never takes the cow), so every accessor
/// dispatches on `drain.owns`. Only owned cows reach this guard: the
/// borrowed empty-range shortcut is handled separately by `drop_impl`.
struct CowSpliceFill<'f, 'r, 'a, T, const N: usize> {
    drain: &'f mut CowDrain<'r, 'a, T, N>,
}

impl<'f, 'r, 'a, T, const N: usize> CowSpliceFill<'f, 'r, 'a, T, N> {
    /// The cow being filled (always owned).
    #[inline]
    fn cow(&mut self) -> &mut CovariantCow<'a, T, N> {
        if self.drain.owns {
            &mut self.drain.cow
        } else {
            &mut *self.drain.slot
        }
    }

    /// Reserves room for `additional` more slots.
    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.cow().reserve_owned(additional);
    }

    /// Shifts the live tail right by `slots`, growing the dead gap at
    /// `begin` by that many slots, and publishes the grown length.
    ///
    /// # Safety
    ///
    /// Capacity for `original_len + slots` slots; `slots >= 1`.
    unsafe fn open_gap(&mut self, slots: usize) {
        let end = self.drain.end;
        let original_len = self.drain.original_len;
        // Safety: `[end, original_len)` is the live tail and
        // `[end + slots, original_len + slots)` is spare capacity. `ptr::copy`
        // handles the overlap (the destination is to the right).
        unsafe {
            let base = self.cow().base_ptr_owned();
            ptr::copy(base.add(end), base.add(end + slots), original_len - end);
        }
        self.drain.end = end + slots;
        self.drain.original_len = original_len + slots;
        // Publish the grown length now (see `SpliceFill::open_gap`): the next
        // chunk's `reserve_owned` sizes from the cow's own length, and the
        // shifted tail must lie inside it. The guard closes the dead gap
        // before the cow is observable again, including on the unwind path.
        //
        // Safety: slots `[0, original_len + slots)` are initialized; the cow
        // is owned and `set_len` preserves the state tag.
        unsafe { self.cow().set_len(original_len + slots) };
    }

    /// Writes `x` into the dead gap at `begin` and advances the gap start.
    ///
    /// # Safety
    ///
    /// `begin < end`; the slot is dead.
    unsafe fn write(&mut self, x: T) {
        let at = self.drain.begin;
        // Re-derive the base per write (Stacked Borrows, as in `SpliceFill`).
        // Safety: `at < end <= original_len <= capacity` and the slot is dead.
        unsafe {
            let base = self.cow().base_ptr_owned();
            base.add(at).write(x);
        }
        self.drain.begin = at + 1;
    }
}

impl<T, const N: usize> Drop for CowSpliceFill<'_, '_, '_, T, N> {
    fn drop(&mut self) {
        if self.drain.begin != self.drain.end {
            // The replacement ended before the gap closed (an exhausted
            // zero-hint iterator, a lying lower bound, or a non-fused
            // iterator). Compact the live tail over the dead remainder.
            let begin = self.drain.begin;
            let end = self.drain.end;
            let tail = self.drain.original_len - end;
            // Safety: `[end, original_len)` is the live tail and
            // `[begin, begin + tail)` lies at or before it (`begin < end`), so
            // `ptr::copy` handles the overlap.
            unsafe {
                let base = self.cow().base_ptr_owned();
                ptr::copy(base.add(end), base.add(begin), tail);
            }
            self.drain.end = begin;
            self.drain.original_len = begin + tail;
        }
        let len = self.drain.original_len;
        // Safety: `[0, len)` is initialized and `len <= capacity`; the cow is
        // owned, and `set_len` preserves the state tag.
        unsafe { self.cow().set_len(len) };
    }
}

impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> CowSplice<'r, 'a, T, N, I> {
    /// Shared `Drop` body; mirrors [`Splice`](crate::Splice)'s
    /// `drop_impl` with the cow's `CowDrain` geometry.
    fn drop_impl(&mut self) {
        // Drop the not-yet-yielded removed elements first.
        while self.drain.next().is_some() {}
        if !self.drain.owns && self.drain.slot.is_borrowed() {
            // Empty range on a borrowed cow: the constructor's upper-hint
            // shortcut left it borrowed, and `Drop` has no `T: Clone` to
            // materialize it. An honest replacement is empty, so this pulls
            // zero times; a replacement that yields is a logic error, named
            // here rather than failing `insert_owned`'s generic assertion.
            // (The slot is borrowed, so there is no owned storage to insert
            // into without materializing: the old per-element `insert_owned`
            // loop could never succeed.)
            let _ = self.replace_with.next().map(|x| {
                drop(x);
                panic!("splice replacement yielded an element after reporting an empty size hint")
            });
            return;
        }

        {
            let mut fill = CowSpliceFill {
                drain: &mut self.drain,
            };
            // Phase 1: refill the dead hole in place. Advance `begin` after
            // each write so a panic keeps the written prefix and leaves the
            // rollback a correct geometry (see `Splice::drop_impl` for the
            // full argument).
            while fill.drain.begin < fill.drain.end {
                match self.replace_with.next() {
                    Some(x) => {
                        // Safety: `begin < end`, so the slot is in the dead
                        // gap.
                        unsafe { fill.write(x) };
                    }
                    None => break,
                }
            }

            // Phase 2: the cow twin of `Splice::drop_impl`'s chunked growth —
            // one reserve, one tail shift, and one fill per chunk, with the
            // chunk size doubling for zero-lower-bound iterators.
            let mut chunk = crate::SPLICE_CHUNK;
            'grow: loop {
                // Pull before opening a gap so an exhausted iterator costs no
                // shift (and a lying huge hint is never consulted).
                let Some(x) = self.replace_with.next() else {
                    break;
                };
                let (lower, upper) = self.replace_with.size_hint();
                let mut slots = chunk.max(lower.saturating_add(1));
                if let Some(upper) = upper {
                    slots = slots.min(upper.saturating_add(1));
                }
                // Panics (capacity overflow / allocation failure) happen
                // before any geometry change, so the guard sees the phase-1
                // state; `x` is dropped by the unwind.
                fill.reserve(slots);
                // Safety: capacity for `original_len + slots`; `slots >= 1`.
                unsafe { fill.open_gap(slots) };
                // Safety: the gap is open and `slots >= 1`.
                unsafe { fill.write(x) };
                while fill.drain.begin < fill.drain.end {
                    match self.replace_with.next() {
                        Some(x) => {
                            // Safety: `begin < end`.
                            unsafe { fill.write(x) };
                        }
                        None => break 'grow,
                    }
                }
                chunk = chunk.saturating_mul(2);
            }
        }
        // The guard has compacted the gap and published the length; the
        // `CowDrain` rollback now takes its empty-range skip and hands the
        // cow back.
        self.drain.pos = self.drain.begin;
        self.drain.back = self.drain.begin;
    }
}

// Same drop obligations as `CowDrain`: the body moves raw bytes, drops
// *owned* removed elements, and never reads any `T` nor anything reachable
// through `'a` (the replacement is moved in, not cloned).
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<'r, #[may_dangle] 'a, #[may_dangle] T, const N: usize, I: Iterator<Item = T>> Drop
    for CowSplice<'r, 'a, T, N, I>
{
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'r, 'a, T, const N: usize, I: Iterator<Item = T>> Drop for CowSplice<'r, 'a, T, N, I> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

// --------------------------------------------------------------------------
// CowExtractIf
// --------------------------------------------------------------------------

/// An iterator that moves the matching elements out of a [`CovariantCow`]
/// according to a predicate, keeping the rest in their original order.
///
/// Created by [`CovariantCow::extract_if`] /
/// [`CovariantCow::extract_if_owned`]. The scan is the same in-place
/// backshift scan as [`ExtractIf`](crate::ExtractIf); a borrowed cow is
/// materialized at construction, so the rollback only ever sees owned storage.
/// The length is published as `0` for the scan's duration (forget → safe
/// leak, never double-drop), and dropping the iterator early — normally or
/// while unwinding — compacts the untouched tail and restores the length.
pub struct CowExtractIf<'r, 'a, T, const N: usize, F> {
    cow: &'r mut CovariantCow<'a, T, N>,
    pred: F,
    /// Index of the next slot to examine, in the *original* coordinates.
    idx: usize,
    /// Number of elements moved out so far: `[idx - del, idx)` are dead holes.
    del: usize,
    /// Length of the cow when the scan started.
    original_len: usize,
}

impl<'r, 'a, T, const N: usize, F> Iterator for CowExtractIf<'r, 'a, T, N, F>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        // One derivation per call (the predicate reborrows the storage, and
        // this loop performs no other cow access) — the same per-location
        // discipline as `ExtractIf::next`.
        let base = self.cow.base_ptr();
        while self.idx < self.original_len {
            // Safety: `idx < original_len <= capacity`; slot `idx` is in
            // the untouched, initialized tail.
            let cur = unsafe { base.add(self.idx) };
            if (self.pred)(unsafe { &mut *cur }) {
                // Safety: initialized and not yet processed; the value is
                // moved out and its slot becomes a dead hole counted by
                // `del` (ZST protocol site 3).
                let out = unsafe { ptr::read(cur) };
                self.idx += 1;
                self.del += 1;
                return Some(out);
            }
            if self.del > 0 {
                // Safety: distinct slots in the same buffer (`hole < cur`).
                unsafe { ptr::copy_nonoverlapping(cur, base.add(self.idx - self.del), 1) };
            }
            self.idx += 1;
        }
        None
    }
}

impl<'r, 'a, T, const N: usize, F> FusedIterator for CowExtractIf<'r, 'a, T, N, F> where
    F: FnMut(&mut T) -> bool
{
}

// Auto traits: the constructor materializes a borrowed cow, so this type only
// ever observes an **owned** cow. Moving it between threads therefore moves
// owned `T`s (Send needs `T: Send`) plus the `&mut` borrow, exactly like
// `CowDrain`/`CowIntoIter`; the stiffer `T: Send + Sync` of the borrowed
// `CovariantCow` state is not needed. `Sync` shares `&CowExtractIf`; the only
// interior mutability is the predicate, hence the `F: Sync` bound.
//
// Safety: `extract_if` materializes first and `extract_if_owned` asserts
// `!is_borrowed`, so no path can observe the borrowed union variant.
unsafe impl<'r, 'a, T: Send, const N: usize, F: Send> Send for CowExtractIf<'r, 'a, T, N, F> {}
unsafe impl<'r, 'a, T: Sync, const N: usize, F: Sync> Sync for CowExtractIf<'r, 'a, T, N, F> {}

impl<'r, 'a, T, const N: usize, F> CowExtractIf<'r, 'a, T, N, F> {
    /// Shared `Drop` body: compacts the untouched tail over the dead holes
    /// and restores the length through the cow's `BackshiftTarget` impl
    /// (which preserves both tag bits). It runs no destructor, so it cannot
    /// panic, and it runs on the normal *and* the unwinding path.
    fn rollback(&mut self) {
        if self.idx < self.original_len && self.del > 0 {
            // Safety: `[idx, original_len)` is the untouched, initialized
            // tail; `[idx - del, idx)` is dead. `ptr::copy` handles overlap.
            unsafe {
                let base = self.cow.backshift_base();
                ptr::copy(
                    base.add(self.idx),
                    base.add(self.idx - self.del),
                    self.original_len - self.idx,
                );
            }
        }
        // Safety: exactly `original_len - del` leading slots are initialized
        // (retained prefix + compacted tail); this only shrinks, and the
        // setter preserves the state tags.
        unsafe { self.cow.backshift_set_len(self.original_len - self.del) };
    }
}

// The drop body shifts raw bytes and fixes the length; it never reads any `T`
// nor anything reachable through `'a` (the predicate is not called on this
// path), so the same `#[may_dangle]` annotations as `CowDrain` apply.
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<'r, #[may_dangle] 'a, #[may_dangle] T, const N: usize, F> Drop
    for CowExtractIf<'r, 'a, T, N, F>
{
    fn drop(&mut self) {
        self.rollback();
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'r, 'a, T, const N: usize, F> Drop for CowExtractIf<'r, 'a, T, N, F> {
    fn drop(&mut self) {
        self.rollback();
    }
}
