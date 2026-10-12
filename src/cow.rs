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
use crate::{try_heap_layout, TryReserveError, HEAP_TAG};

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
/// The fallible [`try_reserve`](Self::try_reserve) /
/// [`try_reserve_exact`](Self::try_reserve_exact) (and their `_owned` twins)
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
    fn heap_cap(&self) -> usize {
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
    pub fn capacity(&self) -> usize {
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
                // `new_len` values.
                for _ in 0..(additional - 1) {
                    assert!(self.raw_len() < MAX_LEN, "capacity overflow");
                    self.len = self.raw_len() + 1;
                    mem::forget(value.clone());
                }
                assert!(self.raw_len() < MAX_LEN, "capacity overflow");
                self.len = self.raw_len() + 1;
                mem::forget(value);
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
                    for _ in 0..(additional - 1) {
                        let v = value.clone();
                        let l = self.raw_len();
                        // Safety: owned state with capacity for `additional`
                        // more elements, so slot `[l]` is in bounds and
                        // uninitialized. The pointer is re-derived each
                        // iteration because a write to `self.len` is a use of
                        // `self` that invalidates hoisted raw derivations under
                        // Stacked Borrows.
                        unsafe { self.base_ptr().add(l).write(v) };
                        self.len = (self.len & HEAP_TAG) | (l + 1);
                    }
                    // Safety: as above, for the final uninitialized slot; moving
                    // `value` in consumes it without running its destructor here.
                    let l = self.raw_len();
                    unsafe { self.base_ptr().add(l).write(value) };
                    self.len = (self.len & HEAP_TAG) | (l + 1);
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
        if needed <= len {
            // `additional == 0`: the borrowed slice is already exactly full.
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
        if needed <= len {
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
    /// original is not dropped during the call. This implementation seeds slot
    /// 0 with the original and clones *that* slot instead — same counts, with
    /// the original in a different position (unobservable for a
    /// value-preserving `Clone`).
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
            // Move the owned `elem` into slot 0, then clone *that* slot for
            // the rest (exactly `n` values, hence `n` destructors —
            // `vec![elem; n]` semantics). Under `T: Copy` this is one length
            // publication plus the O(log n) doubling copies of `spec::bulk`;
            // the stable path below is the same accounting, element-wise.
            // Safety: capacity `n` reserved, the cow is owned and empty, and
            // slots `[0, n)` are uninitialized. For a ZST the write consumes
            // `elem` (ZST protocol site 1).
            unsafe { T::seed_fill(elem, &mut out, n) };
        }
        #[cfg(not(feature = "specialization"))]
        {
            // Move the owned `elem` into slot 0, then clone *that* slot for the
            // rest (exactly `n` values, hence `n` destructors — `vec![elem; n]`
            // semantics). The length writes preserve the heap tag, since `reserve`
            // may have spilled.
            let dst0 = out.base_ptr();
            // Safety: capacity `n` reserved, slot 0 uninitialized. For a ZST this
            // consumes `elem` (ZST protocol site 1).
            unsafe { dst0.write(elem) };
            out.len = (out.len & HEAP_TAG) | 1;
            for i in 1..n {
                // Safety: slot 0 is initialized; re-derived each iteration because
                // the `out.len` write below invalidates a hoisted pointer inline.
                // (ZST protocol: slot 0 was never written, but reading a ZST through
                // a `&T` touches no bytes and the clone is counted.)
                let v = unsafe { (&*out.base_ptr_const()).clone() };
                // Safety: slot [i] is uninitialized (capacity `n` reserved).
                unsafe { out.base_ptr().add(i).write(v) };
                out.len = (out.len & HEAP_TAG) | (i + 1);
            }
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
    /// update (`push`/`insert`/`resize`/`extend`) adopts, like `to_mut`;
    /// `Clone` is the one deliberate exception, because calling `T::clone` is
    /// its entire contract.
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

    /// Builds the [`CowDrain`] for already-validated `(begin, end)` over a
    /// length-`len` container: inert for an empty range, otherwise taking the
    /// whole container out of `self` (the caller must have materialized any
    /// borrowed state first).
    fn make_drain(&mut self, begin: usize, end: usize, len: usize) -> CowDrain<'_, 'a, T, N> {
        if begin == end {
            // Empty range: a pure no-op. The container is never taken.
            return CowDrain {
                cow: ManuallyDrop::new(CovariantCow::new()),
                slot: self,
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
        let cow = if self.is_heap() {
            // Safety: owned heap state (`make_drain` is only reached with a
            // materialized cow when the range is non-empty).
            let (ptr, cap) = unsafe { self.data.heap };
            let raw_len = self.len;
            // Leave the slot an empty inline cow by writing only `len`; see
            // `CovariantVec::drain` for why the stale union bytes are inert.
            // (`raw_len` has `BORROWED_TAG` clear — an owned cow is never
            // borrowed — and `HEAP_TAG` set.)
            self.len = 0;
            // Safety: `ptr`/`cap` describe the live block this cow owned and
            // `raw_len` is its tagged length; ownership moves into the returned
            // cow, and `self` no longer refers to the block.
            CovariantCow {
                data: Data3 { heap: (ptr, cap) },
                len: raw_len,
            }
        } else {
            mem::take(self)
        };
        CowDrain {
            cow: ManuallyDrop::new(cow),
            slot: self,
            owns: true,
            begin,
            pos: begin,
            end,
            back: end,
            original_len: len,
        }
    }

    /// Removes the specified range and returns an iterator over the removed
    /// elements, in order, from either end.
    ///
    /// A **borrowed** cow is materialized first (hence the `T: Clone` bound) —
    /// but only for a **non-empty** range. The range is validated first, so an
    /// empty range is a pure no-op that leaves a borrowed cow borrowed and
    /// clones nothing, and a reversed/out-of-bounds range panics without paying
    /// for a materialization.
    ///
    /// The design then mirrors [`CovariantVec::drain`](crate::CovariantVec::drain):
    /// the whole container is
    /// moved into the returned [`CowDrain`] and restored on drop, so the tail is
    /// shifted back and the length restored — even while unwinding. See
    /// [`CovariantVec::drain`](crate::CovariantVec::drain) for the aliasing
    /// rationale and the `mem::forget` semantics (forgetting a non-empty
    /// `CowDrain` leaks the whole cow; an empty range never takes it).
    ///
    /// Use [`Self::drain_owned`] when the cow is known to be owned and `T: Clone`
    /// is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if the range's start is greater than its end, or its end is
    /// greater than the length.
    #[track_caller]
    pub fn drain<R>(&mut self, range: R) -> CowDrain<'_, 'a, T, N>
    where
        R: RangeBounds<usize>,
        T: Clone,
    {
        // Validate the range *before* materializing: `raw_len` already masks
        // off the tag bits, so it is correct for the borrowed state too.
        let len = self.raw_len();
        let (begin, end) = crate::drain_bounds(&range, len);
        if begin != end && self.is_borrowed() {
            self.to_mut();
        }
        self.make_drain(begin, end, len)
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
    #[track_caller]
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Clone,
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|x| f(x));
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
            // HEAP_TAG-collision bound is enforced here by the reserve path;
            // `push` and the `extend` paths instead carry the explicit
            // `raw_len < MAX_LEN` assert themselves.
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
        // pointer and capacity stay in registers. The loop is single-entry on
        // purpose: a back edge re-entering it would force LLVM to reload the
        // state at the join, so the lying-`size_hint` remainder falls through
        // to the general loop instead. The inline state is handled there too
        // (it must re-derive per element).
        if self.is_heap() {
            let base = self.base_ptr();
            let cap = self.heap_cap();
            let mut len = self.raw_len();
            while len < cap {
                let Some(x) = it.next() else { return };
                // Safety: heap state with `len < cap`, so slot `len` is in
                // bounds and uninitialized; `write` moves `x` in without
                // dropping it. `base` stays valid across the publication
                // below (independent block; see above).
                unsafe { base.add(len).write(x) };
                len += 1;
                // Published before the next `it.next()` (user code, may
                // panic). Owned states have `BORROWED_TAG == 0`, so
                // `HEAP_TAG | len` is the tag-consistent form.
                self.len = HEAP_TAG | len;
            }
            // Reservation exhausted: the general loop handles the remainder.
        }
        // General loop (inline state, or the exhausted-heap remainder):
        // per-element re-derivation — the length publication invalidates a
        // hoisted pointer when the elements live inside `self` — and
        // `push_owned` may spill, after which the per-element `is_heap` test
        // picks up the heap branch.
        for x in it {
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
            }
        }
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
        // Safe operation: `base_ptr` dispatches on the tag, uses place access,
        // and itself asserts the non-borrowed state the caller promises.
        self.base_ptr()
    }

    #[inline]
    unsafe fn publish_len(&mut self, n: usize) {
        // Safety: `n` slots were just initialized, and `n <= capacity`. The
        // fill only ever runs in an owned state, which `set_len` re-asserts.
        unsafe { self.set_len(n) };
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
        // `self` holding exactly the clones made so far (the fill publishes the
        // length per element), which `Drop` cleans up — no double drops. The
        // published length stays `0` until then, so the partially-filled
        // prefix is invisible to destructors mid-panic.
        //
        // The ZST case: `clone_fill` reads and writes no bytes, but it still
        // runs `T::clone` `len` times — deliberate (callers count those
        // clones), and the one place this crate's ZST handling *does not*
        // adopt the count. Contrast `to_mut`, which adopts without cloning;
        // the asymmetry is documented there.
        self.clear();
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

        if vec.capacity() > N {
            // Take over the allocation (cap may be <= N only when `N == 0`,
            // which cannot happen since cap > N).
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
        let Some(first) = it.next() else {
            return;
        };
        self.materialize_with_min_capacity(lo.max(1));
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
        let Some(first) = it.next() else {
            return;
        };
        self.materialize_with_min_capacity(lo.max(1));
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
    type IntoIter = CowIntoIter<'a, T, N>;
    fn into_iter(mut self) -> Self::IntoIter {
        // A borrowed cow must be materialized (hence the `T: Clone` bound);
        // owned states are handed straight to the iterator with no allocation
        // or copy. For a no-`Clone` owned cow use `into_iter_owned` directly.
        if self.is_borrowed() {
            self.to_mut();
        }
        self.into_iter_owned()
    }
}

// --------------------------------------------------------------------------
// CowIntoIter
// --------------------------------------------------------------------------

/// Consuming iterator over an **owned** [`CovariantCow`].
///
/// Created by [`CovariantCow::into_iter_owned`], or by `IntoIterator` on a
/// `CovariantCow` (which materializes a borrowed cow first). Unlike routing
/// through a `Vec`, this moves elements directly out of the cow's inline or
/// heap storage, so neither owned state allocates or copies on consumption.
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
        // Safety: [head, tail) are initialized; `&mut self` excludes aliases.
        unsafe { slice::from_raw_parts_mut(self.cow.base_ptr().add(self.head), n) }
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
            let p = unsafe { self.cow.base_ptr().add(self.head) };
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
        // Safety: slot [head] is initialized.
        let p = unsafe { self.cow.base_ptr().add(self.head) };
        self.head += 1;
        Some(unsafe { p.read() })
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
        let p = unsafe { self.cow.base_ptr().add(self.tail) };
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
// CowDrain
// --------------------------------------------------------------------------

/// A draining iterator over a [`CovariantCow`].
///
/// Created by [`CovariantCow::drain`]. For a **non-empty** range the whole
/// container is moved out and restored on drop (even while unwinding), so the
/// tail is shifted back and the length restored; the cow is materialized
/// before draining, so it is always in an owned state while such a `CowDrain`
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
        // Safety: as `as_slice`, plus `&mut self` excludes aliases.
        unsafe { slice::from_raw_parts_mut(self.cow.base_ptr().add(self.pos), n) }
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
        while rb.pos < rb.back {
            // Safety: `pos < back <= end <= original_len`; slot `pos` holds an
            // element not yet yielded. (`pos` advances before the drop only for
            // this loop's bookkeeping — `CowDrainRollback` does not read `pos`,
            // so the ordering is not load-bearing for the unwind path.)
            let p = unsafe { rb.cow.base_ptr().add(rb.pos) };
            rb.pos += 1;
            unsafe { ptr::drop_in_place(p) };
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
            // Preserve the tag bits (owned states have `BORROWED_TAG == 0`).
            c.len = (c.len & HEAP_TAG) | (self.original_len - (self.end - self.begin));
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
        unsafe {
            let p = self.cow.base_ptr().add(self.pos);
            self.pos += 1;
            Some(p.read())
        }
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
        // Safety: as in `next`.
        unsafe {
            let p = self.cow.base_ptr().add(self.back);
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
