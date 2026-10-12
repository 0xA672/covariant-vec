//! # `covariant-vec`
//!
//! A small-vector (`inline` + `heap` two-state) container whose state
//! discrimination bit is encoded in the **highest bit of the `len` field**,
//! rather than in a data-embedded sentinel.
//!
//! ## Two points worth calling out
//!
//! 1. **Dropck:** every generic parameter of a type that has a `Drop` impl is
//!    subject to the strict drop-check unless marked `#[may_dangle]` —
//!    **lifetime parameters included**. `PhantomData<&'a ()>` is merely how
//!    `'a` is threaded into the type (for variance anchoring); the strictness
//!    is a property of the parameter itself, not of the `PhantomData` field.
//!    The `Drop` impl below therefore marks **both** `'a` and `T`. Without the
//!    `'a` annotation the motivating use case (`CovariantVec<'src, Label<'src>,
//!    N>` dropped together with `'src`) does not compile even on nightly.
//!
//! 2. **`Send`/`Sync`:** `NonNull<u8>` carries `!Send`/`!Sync`, which would
//!    otherwise poison the whole type. Explicit `unsafe impl`s restore the
//!    intended `Vec`-like semantics (`Send ⇔ T: Send`, `Sync ⇔ T: Sync`).
//!
//! ## Compile-time properties
//!
//! Auto traits propagate from `T`, so a non-`Sync` element makes the container
//! non-`Sync` (the negative direction is checked with `compile_fail` doctests):
//!
//! ```compile_fail
//! use std::cell::Cell;
//! use covariant_vec::CovariantVec;
//! fn assert_sync<T: Sync>() {}
//! // `Cell<i32>` is `!Sync`, so `CovariantVec<Cell<i32>>` is `!Sync`.
//! assert_sync::<CovariantVec<'static, Cell<i32>, 4>>();
//! ```
//!
//! ```compile_fail
//! use std::rc::Rc;
//! use covariant_vec::CovariantVec;
//! fn assert_send<T: Send>() {}
//! // `Rc<i32>` is `!Send`, so `CovariantVec<Rc<i32>>` is `!Send`.
//! assert_send::<CovariantVec<'static, Rc<i32>, 4>>();
//! ```
//!
//! The container is also covariant in both `'a` and `T` (see `tests/variance.rs`).
//!
//! ## The ZST ownership protocol
//!
//! For a zero-sized `T` the container stores **only a count**, never element
//! bytes: ZSTs never spill, so there is no storage to write to. Ownership is
//! therefore tracked by the length field alone, and the following three
//! conventions must be preserved together:
//!
//! 1. **`push`/`push_owned` consume the value without storing it** — the
//!    argument is `mem::forget`-ed and the length incremented. The element is
//!    conceptually "owned by the count".
//! 2. **Destructor runs happen over `raw_len` slots that were never written**
//!    — `drop_contents`/`truncate`/`retain_mut`/`Drain::drop` call
//!    `drop_in_place` over `raw_len` slots of a ZST array. This is sound only
//!    because a ZST's drop glue touches no bytes, and it runs exactly one
//!    destructor per counted element.
//! 3. **`pop`/`IntoIter::next`/`Drain::next` materialize a value by `ptr::read`
//!    from such an unwritten slot** — again sound only for ZSTs (no bytes are
//!    read, and the count is decremented so the element is not dropped later).
//!
//! The invariant is "exactly one conceptual destructor per element", and it is
//! balanced only because `mem::forget` (site 1) is matched by the `raw_len`
//! accounting (sites 2/3). A future change that makes `push` *store* the ZST
//! would double-count; a change that stops `mem::forget`-ing would double-drop.
//! Every site is annotated with a `ZST protocol` safety comment.
//!
//! ## Toolchain / MSRV
//!
//! * The minimum supported Rust version is **1.80** (inline `const {}` blocks
//!   need 1.79; bare `size_of`/`align_of` in the prelude need 1.80; const
//!   generics and `addr_of!` need 1.51+).
//! * The `dropck_eyepatch` feature is **nightly only**: it enables
//!   `#![feature(dropck_eyepatch)]`. Stable builds must leave it off (the
//!   default), where dropck is strict. The `cow` feature is stable-safe.
//! * The `specialization` feature is **nightly only** too: it enables
//!   `#![feature(specialization)]` — the *full* feature, because the sound
//!   subset rejects trait-bound specialization outright — and switches the
//!   clone loops to the `T: Copy` bulk copy described below. Stable builds
//!   must leave it off (the default), where every clone loop is element-wise.
//!   Its **minimum known-good nightly is `nightly-2026-01-16`**
//!   (1.94.0-nightly): the accepted shape of this specialization has moved
//!   between nightlies before, so CI pins that date and runs the floating
//!   nightly as a canary. See the feature's entry in `Cargo.toml`.
//!
//! ## `T: Copy` fast paths (nightly only)
//!
//! Every clone loop in the crate — `Clone` on both containers,
//! `CovariantCow::to_mut` and `CovariantCow::try_materialize` — runs through
//! one shared helper. With the `specialization` feature on, a `T: Copy`
//! element is filled with a single `ptr::copy_nonoverlapping` instead of
//! element by element.
//!
//! The specialization is keyed **exactly** on `T: Copy`, matching what `std`'s
//! `Vec` does, so it is unobservable: `Copy` is a contract that `clone` is a
//! bitwise copy. It is deliberately *not* keyed on `!needs_drop::<T>()` — that
//! is a strict superset of `Copy`, and a no-drop `!Copy` type with a
//! side-effectful `Clone` would silently stop running `clone`.
//!
//! The feature changes speed only, never behaviour or panic safety: the
//! element-wise loop publishes the length *after* each element, so a panic
//! inside `T::clone` leaves the destination holding exactly the clones made so
//! far, and that is the only loop stable builds ever run. See the crate's
//! `clone_fill` module for the mechanism and the soundness argument.
//!
//! ## Fallible allocation
//!
//! The `try_reserve` / `try_reserve_exact` family (on [`CovariantVec`] and,
//! under the `cow` feature, `CovariantCow`) reports failure as a
//! [`TryReserveError`] instead of panicking on capacity overflow or aborting on
//! allocation failure. [`TryReserveError::kind`] distinguishes the two cases.
//!
//! This needs no unstable API: the global allocator's `alloc`/`realloc`
//! already report failure by returning a **null pointer** (it is
//! `handle_alloc_error`, called explicitly by the infallible paths, that
//! aborts). A fallible path therefore only has to null-check the returned
//! pointer. [`alloc::collections::TryReserveError`] has no public constructor,
//! so this crate defines its own error type with the same two failure modes.
//!
//! ## Deliberately not provided
//!
//! * **`append` / `split_off` / `splice` / `extract_if`** — omitted for now to
//!   keep the unsafe surface small. `sort*`/`dedup*` are already reachable
//!   through [`DerefMut`], and a `Vec` round-trip covers the rest.
//! * **A stable `T: Copy` bulk-copy fast path** — specialization is the only
//!   mechanism that keys on `T: Copy` exactly, and it is nightly-only; on
//!   stable the generic element-wise path is used for every type (correct,
//!   just not maximally fast for `Copy` elements). See the `specialization`
//!   feature for the opt-in.

#![cfg_attr(feature = "dropck_eyepatch", feature(dropck_eyepatch))]
// `specialization` is an incomplete feature, so enabling it also silences the
// `incomplete_features` lint. Both attributes are scoped to the feature: a
// stable build never sees them. See `src/clone_fill.rs` for why the full
// feature (not `min_specialization`) is required and why this crate's use of
// it is sound.
#![cfg_attr(feature = "specialization", feature(specialization))]
#![cfg_attr(feature = "specialization", allow(incomplete_features))]
#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

#[cfg(test)]
extern crate std;

use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{Bound, Deref, DerefMut, RangeBounds};
use core::ptr::{self, addr_of, addr_of_mut, NonNull};
use core::slice;

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use alloc::vec::Vec;

mod clone_fill;

#[cfg(feature = "cow")]
mod cow;
#[cfg(feature = "cow")]
pub use cow::{CovariantCow, CowDrain, CowIntoIter};

#[cfg(feature = "specialization")]
use clone_fill::CloneRepeat;
use clone_fill::{CloneFill, CloneSink};

/// Heap-state tag: the highest bit of the `len` field.
pub(crate) const HEAP_TAG: usize = 1 << (usize::BITS - 1);

/// The allocation layout for `cap` elements of `T`.
///
/// Centralizes the byte-size check so the `.expect` lives in one place. Note
/// that the check is on **bytes**: `Layout::array::<T>` fails once
/// `cap * size_of::<T>()` would exceed `isize::MAX`, which is possible for
/// `size_of::<T>() > 1` while `cap <= isize::MAX` still holds (e.g.
/// `with_capacity(isize::MAX)` on `CovariantVec<u64, _>`). Bounding the
/// **element count** (`cap <= isize::MAX`) is the capacity-planning callers'
/// job; this function only guarantees the byte round-trip, and panics
/// `"capacity overflow"` otherwise. Do not replace the `.expect` with
/// `Layout::from_size_align_unchecked` — for multi-byte `T` the expectation
/// is genuinely reachable.
#[inline]
pub(crate) fn heap_layout<T>(cap: usize) -> Layout {
    Layout::array::<T>(cap).expect("capacity overflow")
}

/// Fallible twin of [`heap_layout`]: an over-large request becomes
/// [`TryReserveErrorKind::CapacityOverflow`] instead of a panic.
#[inline]
pub(crate) fn try_heap_layout<T>(cap: usize) -> Result<Layout, TryReserveError> {
    Layout::array::<T>(cap).map_err(|_| TryReserveError::capacity_overflow())
}

/// std's `RawVec` floor for the **first** heap allocation, so a small or empty
/// inline buffer does not grow one element at a time. Mirrors
/// `alloc::raw_vec::min_non_zero_cap`.
#[inline]
pub(crate) const fn min_non_zero_cap<T>() -> usize {
    let size = size_of::<T>();
    if size == 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// Computes and validates the `(begin, end)` half-open bounds of `range` over a
/// container of length `len`. Shared by [`CovariantVec::drain`] and (under the
/// `cow` feature) `CovariantCow::drain`/`drain_owned`, so the range semantics
/// and panic messages live in one place.
#[track_caller]
pub(crate) fn drain_bounds<R: RangeBounds<usize>>(range: &R, len: usize) -> (usize, usize) {
    let begin = match range.start_bound() {
        Bound::Included(&i) => i,
        // `saturating_add` (not `+ 1`): an out-of-range `Excluded(usize::MAX)`
        // start would otherwise overflow. It saturates to `usize::MAX`, which
        // then trips the `begin <= len` assert below as a clean panic (std
        // reports the same `StartPastEnd` case). The mathematically correct
        // `MAX + 1` is unrepresentable, but no real container is that long.
        Bound::Excluded(&i) => i.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&i) => i.saturating_add(1),
        Bound::Excluded(&i) => i,
        Bound::Unbounded => len,
    };
    // Check order and messages mirror `core::slice::index::slice_index_fail`
    // (what `Vec::drain` reports): start past `len`, then end past `len`, then
    // the start-vs-end order.
    assert!(
        begin <= len,
        "range start index {begin} out of range for slice of length {len}"
    );
    assert!(
        end <= len,
        "range end index {end} out of range for slice of length {len}"
    );
    assert!(
        begin <= end,
        "slice index starts at {begin} but ends at {end}"
    );
    (begin, end)
}

/// The error type returned by the `try_reserve*` family.
///
/// [`alloc::collections::TryReserveError`] has no public constructor, so a
/// `no_std` + `alloc` crate that drives the global allocator directly cannot
/// produce one. This type carries the same two failure modes; see
/// [`TryReserveErrorKind`].
///
/// On compilers >= 1.81 this type also implements [`core::error::Error`]. The
/// crate's MSRV is 1.80, so the impl is gated on a `build.rs`-emitted cfg; on
/// an older compiler it is simply absent. The build script prints a warning if
/// it cannot determine the compiler version, so the absence is never silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

/// Why a [`TryReserveError`] was returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TryReserveErrorKind {
    /// The requested capacity is not representable: `len + additional`
    /// overflowed `usize`, or the resulting allocation would exceed
    /// `isize::MAX` bytes (the length field could not hold it, or
    /// [`Layout::array`] rejected it).
    CapacityOverflow,
    /// The global allocator returned a null pointer for this layout.
    AllocError {
        /// The layout the allocator could not satisfy.
        layout: Layout,
    },
}

impl TryReserveError {
    #[inline]
    pub(crate) fn capacity_overflow() -> Self {
        TryReserveError {
            kind: TryReserveErrorKind::CapacityOverflow,
        }
    }

    #[inline]
    pub(crate) fn alloc_error(layout: Layout) -> Self {
        TryReserveError {
            kind: TryReserveErrorKind::AllocError { layout },
        }
    }

    /// The reason the reservation failed.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> TryReserveErrorKind {
        self.kind
    }

    /// Converts a failed reservation into the infallible behavior: panic on
    /// capacity overflow, abort through the allocation-error hook on allocation
    /// failure — exactly what the infallible `reserve`/`push` paths do.
    #[inline]
    #[track_caller]
    pub(crate) fn abort(self) -> ! {
        match self.kind {
            TryReserveErrorKind::CapacityOverflow => panic!("capacity overflow"),
            TryReserveErrorKind::AllocError { layout } => handle_alloc_error(layout),
        }
    }
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TryReserveErrorKind::CapacityOverflow => f.write_str("capacity overflow"),
            TryReserveErrorKind::AllocError { layout } => {
                write!(f, "memory allocation of {} bytes failed", layout.size())
            }
        }
    }
}

// `core::error::Error` stabilized in Rust 1.81, newer than this crate's MSRV
// (1.80), so the impl is gated on a cfg emitted by `build.rs` from the
// building compiler's version. `source()` defaults to `None`, which is right:
// neither kind wraps another error.
#[cfg(has_error_trait)]
impl core::error::Error for TryReserveError {}

/// Overlapping storage body.
///
/// * `inline` — the inline array, wrapped so it is a legal union field and so
///   that an accidental read of the wrong variant yields `MaybeUninit` (any
///   bit pattern is valid) rather than immediate UB.
/// * `heap` — an erased `(pointer, capacity)`; `T` deliberately does not
///   appear so the heap side contributes no variance constraint on `T`.
///
/// The `repr(C)` is load-bearing *specification*, not interop: a `repr(Rust)`
/// union carries no layout guarantees at all (the reference promises only
/// alignment divisible by the field's and no padding beyond what soundness
/// needs), whereas `repr(C)` pins every field to offset 0 and the size to the
/// largest field rounded up to its alignment — the exact union layout the
/// state encoding and `tests/layout.rs` assume.
#[repr(C)]
union Data<T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    heap: (NonNull<u8>, usize),
}

/// An inline-or-heap vector with a compile-time inline capacity of `N`.
///
/// * `'a` — a lifetime anchor; covariant.
/// * `T` — element type; covariant (variance passes only through the inline
///   side).
/// * `N` — inline capacity; may be `0`.
#[repr(C)]
pub struct CovariantVec<'a, T, const N: usize> {
    data: Data<T, N>,
    /// Compound field: high bit is [`HEAP_TAG`], the low bits are the length.
    /// Meaning defined by the state invariants.
    len: usize,
    /// Covariance anchor for `'a`; zero-sized, not part of the layout.
    marker: PhantomData<&'a ()>,
}

impl<'a, T, const N: usize> CovariantVec<'a, T, N> {
    // ----------------------------------------------------------------------
    // Discrimination helpers
    // ----------------------------------------------------------------------

    /// `true` iff the storage currently lives on the heap.
    #[inline]
    #[must_use]
    pub const fn is_heap(&self) -> bool {
        self.len & HEAP_TAG != 0
    }

    /// The number of initialized elements, ignoring the tag bit.
    #[inline]
    #[must_use]
    const fn raw_len(&self) -> usize {
        self.len & !HEAP_TAG
    }

    /// The heap capacity. Only meaningful in the heap state.
    #[inline]
    fn heap_cap(&self) -> usize {
        debug_assert!(self.is_heap());
        // Safety: the tag says we are in the heap variant.
        unsafe { self.data.heap.1 }
    }

    /// Raw pointer to the first element slot.
    #[inline]
    fn base_ptr(&mut self) -> *mut T {
        if self.is_heap() {
            // Safety: heap state.
            unsafe { self.data.heap.0.cast::<T>().as_ptr() }
        } else {
            // `addr_of_mut!` performs only place access — no intermediate
            // reference — so it is safe even on a union field; `MaybeUninit`
            // accepts any bits.
            addr_of_mut!(self.data.inline).cast::<T>()
        }
    }

    /// `*const` twin of [`Self::base_ptr`].
    #[inline]
    fn base_ptr_const(&self) -> *const T {
        if self.is_heap() {
            // Safety: heap state.
            unsafe { self.data.heap.0.cast::<T>().as_ptr() as *const T }
        } else {
            // Place access only.
            addr_of!(self.data.inline).cast::<T>()
        }
    }

    // ----------------------------------------------------------------------
    // Growth
    // ----------------------------------------------------------------------

    /// Next capacity given the current one.
    ///
    /// `from <= N` selects the first-spill capacity `max(N + 1, 2N)`, floored to
    /// [`min_non_zero_cap`] (so `N = 0` does not crawl `1 -> 2 -> 4 -> 8`); above
    /// that we double. Always clamped to `isize::MAX`.
    #[inline]
    const fn grow_to(from: usize) -> usize {
        let candidate = if from <= N {
            let a = N.saturating_add(1);
            let b = N.saturating_mul(2);
            let base = if a > b { a } else { b };
            let floor = min_non_zero_cap::<T>();
            if base > floor {
                base
            } else {
                floor
            }
        } else {
            from.saturating_mul(2)
        };
        if candidate > isize::MAX as usize {
            isize::MAX as usize
        } else {
            candidate
        }
    }

    // ----------------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------------

    /// Creates an empty inline vector (`len = 0`, tag = 0).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        // Compile-time precondition for unambiguous discrimination:
        // the inline length range `[0, N]` must not overlap the tag bit.
        const { assert!(N <= isize::MAX as usize, "N must fit in isize") };
        CovariantVec {
            data: Data {
                inline: ManuallyDrop::new(MaybeUninit::uninit()),
            },
            len: 0,
            marker: PhantomData,
        }
    }

    /// Creates a vector with at least `capacity` capacity.
    ///
    /// `capacity <= N` stays inline; a larger request goes straight to the
    /// heap state.
    ///
    /// A small or zero request still reports the full inline capacity:
    /// `with_capacity(0).capacity() == N` (the inline buffer always exists),
    /// which differs from `Vec::with_capacity(0).capacity() == 0`.
    ///
    /// For a zero-sized element nothing is allocated (ZSTs never spill, and
    /// [`capacity`](Self::capacity) reports `isize::MAX` regardless of the
    /// requested value), but the request is still bounds-checked: like
    /// [`reserve`](Self::reserve), a capacity above the representable maximum
    /// panics with `"capacity overflow"` rather than being silently accepted.
    #[must_use]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut out = Self::new();
        if size_of::<T>() == 0 {
            // No allocation, but the length field still bounds the capacity.
            assert!(capacity <= isize::MAX as usize, "capacity overflow");
        } else if capacity > N {
            out.alloc_heap(capacity);
        }
        out
    }

    /// Allocates a heap block of `cap` slots and switches to the heap state
    /// (with `raw_len == 0`).
    ///
    /// Preconditions: `size_of::<T>() != 0`, `cap > 0`, `cap <= isize::MAX`.
    #[track_caller]
    fn alloc_heap(&mut self, cap: usize) {
        debug_assert!(size_of::<T>() != 0);
        debug_assert!(cap > 0);
        assert!(cap <= isize::MAX as usize, "capacity overflow");
        let layout = heap_layout::<T>(cap);
        // Safety: valid non-zero layout.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        // Safety: destructive state switch; the inline view is void.
        unsafe {
            self.data.heap = (NonNull::new_unchecked(ptr), cap);
        }
        self.len = HEAP_TAG;
    }

    // ----------------------------------------------------------------------
    // Capacity / length
    // ----------------------------------------------------------------------

    /// The current capacity.
    ///
    /// For a zero-sized element this reports `isize::MAX`, **not** the
    /// `usize::MAX` that `Vec<ZST>` reports: the length lives in the tag word,
    /// so the largest representable length is `isize::MAX`. Consequently a
    /// `reserve` that would exceed `isize::MAX` panics with `"capacity
    /// overflow"`, where `Vec<ZST>::reserve` would silently succeed.
    ///
    /// Note the cross-type asymmetry: `CovariantCow<_, ZST, _>` spends one more
    /// bit of its length field on the borrowed tag, so its ZST capacity is
    /// `2^62 - 1` and its `From<Vec<ZST>>` overflow threshold is correspondingly
    /// lower. The two types are otherwise identical here.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            // ZSTs never spill; capacity is conceptually unbounded up to the
            // representable-length limit.
            isize::MAX as usize
        } else if self.is_heap() {
            self.heap_cap()
        } else {
            N
        }
    }

    /// The number of elements.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw_len()
    }

    /// `true` iff there are no elements.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw_len() == 0
    }

    // ----------------------------------------------------------------------
    // Access
    // ----------------------------------------------------------------------

    /// The elements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // Safety: non-null, aligned, `raw_len` valid elements.
        unsafe { slice::from_raw_parts(self.base_ptr_const(), self.raw_len()) }
    }

    /// The elements as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.raw_len();
        // Safety: as above, and `&mut self` guarantees no aliases.
        unsafe { slice::from_raw_parts_mut(self.base_ptr(), len) }
    }

    /// A raw pointer to the first element slot.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.base_ptr_const()
    }

    /// A mutable raw pointer to the first element slot.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.base_ptr()
    }

    /// The spare capacity as a slice of `MaybeUninit<T>`, starting at `len()`
    /// and running for `capacity() - len()` elements (std parity).
    ///
    /// This is the intended low-level append workflow: write into the returned
    /// slots (e.g. with [`MaybeUninit::write`]) and then publish the new count
    /// with [`set_len`](Self::set_len). It replaces manual
    /// `as_mut_ptr().add(len).write(..)` arithmetic and, unlike `as_mut_ptr`,
    /// hands out a bounds-checked slice. Elements in the returned slice are
    /// *uninitialized*; reading them is UB until written and published.
    #[inline]
    #[must_use]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let len = self.raw_len();
        let cap = self.capacity();
        // Safety: the allocation has `capacity()` slots; slots `[len, cap)` are
        // uninitialized but valid for writes viewed as `MaybeUninit<T>`. For a
        // ZST `cap - len` is a huge count of zero-sized slots, which is sound
        // (any aligned pointer is a valid ZST slice base).
        unsafe {
            slice::from_raw_parts_mut(self.base_ptr().cast::<MaybeUninit<T>>().add(len), cap - len)
        }
    }

    /// Iterator over shared references.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Iterator over mutable references.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    // ----------------------------------------------------------------------
    // Mutation
    // ----------------------------------------------------------------------

    /// Moves the `len` inline elements into a fresh heap block of `new_cap`
    /// and switches the union to the heap variant.
    ///
    /// # Safety
    ///
    /// * currently inline (`!is_heap()`),
    /// * `size_of::<T>() != 0`,
    /// * `new_cap > N` (hence `new_cap >= raw_len`).
    #[track_caller]
    unsafe fn spill(&mut self, new_cap: usize) {
        let len = self.raw_len();
        let layout = heap_layout::<T>(new_cap);
        // Safety: valid non-zero layout.
        let new_ptr = unsafe { alloc(layout) };
        if new_ptr.is_null() {
            handle_alloc_error(layout);
        }
        // Safety: `read`-style migration of `len` valid elements; no user code
        // runs inside this window.
        unsafe {
            let src = addr_of_mut!(self.data.inline).cast::<T>();
            let dst = new_ptr as *mut T;
            ptr::copy_nonoverlapping(src, dst, len);
            self.data.heap = (NonNull::new_unchecked(new_ptr), new_cap);
        }
        self.len = HEAP_TAG | len;
    }

    /// Appends an element.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, value: T) {
        // ZST fast path: never spills.
        if size_of::<T>() == 0 {
            // A ZST can never be heap-backed, so `raw_len + 1` must not carry
            // the heap tag. A real (not debug) assert: in release a violated
            // invariant would silently clear `HEAP_TAG` and leak the heap block
            // instead of failing loudly. The branch is const-folded per `T`, so
            // the check is free on non-ZST paths.
            assert!(!self.is_heap(), "ZST vectors are never heap-backed");
            assert!(self.raw_len() < isize::MAX as usize, "capacity overflow");
            self.len = self.raw_len() + 1;
            // ZST protocol (site 1): a ZST owns no bytes, so the count *is* the
            // ownership. `mem::forget` transfers the value's one conceptual
            // destructor to the count; it is run later by `drop_contents` (or
            // `truncate`/`retain_mut`/`Drain::drop`) over `raw_len` unwritten
            // slots. Storing the value instead would double-count the drop.
            mem::forget(value);
            return;
        }

        let len = self.raw_len();
        if !self.is_heap() {
            if len < N {
                // Path A: inline not full.
                // Safety: slot [len] is uninitialized.
                unsafe {
                    let p = addr_of_mut!(self.data.inline).cast::<T>().add(len);
                    p.write(value);
                }
                self.len = len + 1;
            } else {
                // Path B: inline full → spill.
                // Spilling migrates exactly the `N` initialized inline slots;
                // the new slot index `N` below relies on `len == N` (an inline
                // container is either `len < N` (Path A) or exactly full).
                debug_assert_eq!(len, N);
                let new_cap = Self::grow_to(N);
                // Defense in depth: `grow_to` saturates at `isize::MAX`;
                // if `N == isize::MAX` there is no room for the N+1-th element.
                assert!(new_cap > N, "capacity overflow");
                // Safety: preconditions of `spill` hold (inline, non-ZST, cap > N).
                unsafe { self.spill(new_cap) };
                // Safety: heap slot [N] is uninitialized.
                unsafe {
                    self.data.heap.0.cast::<T>().as_ptr().add(N).write(value);
                }
                self.len = HEAP_TAG | (N + 1);
            }
        } else {
            // Path C: heap state.
            let cap = self.heap_cap();
            if len < cap {
                // Safety: heap slot [len] is uninitialized.
                unsafe {
                    self.data.heap.0.cast::<T>().as_ptr().add(len).write(value);
                }
                self.len = HEAP_TAG | (len + 1);
            } else {
                let new_cap = Self::grow_to(cap);
                // Defense in depth: `grow_to` saturates at `isize::MAX`,
                // so a saturated `new_cap == cap` would write out of bounds.
                assert!(new_cap > cap, "capacity overflow");
                let new_layout = heap_layout::<T>(new_cap);
                let old_layout = heap_layout::<T>(cap);
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

    /// Removes and returns the last element. Never shrinks back to inline.
    #[inline]
    #[must_use]
    pub fn pop(&mut self) -> Option<T> {
        let len = self.raw_len();
        if len == 0 {
            return None;
        }
        // Safety: slot [len - 1] is initialized. ZST protocol (site 3): for a
        // ZST the slot was never written, but `read` of a ZST touches no bytes
        // and the count is decremented below, so the element's conceptual
        // destructor is not run twice.
        let value = unsafe { self.base_ptr().add(len - 1).read() };
        self.len = (self.len & HEAP_TAG) | (len - 1);
        Some(value)
    }

    /// Bulk-appends every item of `iter`.
    ///
    /// Reserves the iterator's lower-bound hint (trusted, as `Vec` does), then
    /// writes directly into the spare capacity for as long as it lasts,
    /// falling back to `push` (which spills/reallocates) for any remainder.
    /// Hoisting the state/tag checks out of the common case — where the
    /// reservation already covered the whole iterator — makes this cheaper
    /// than a bare `for x in iter { self.push(x) }` loop.
    ///
    /// A **heap-backed** destination takes a dedicated fast path with the
    /// block pointer and capacity hoisted out of the loop: the pointer has
    /// provenance in the independent block, so the per-element length
    /// publication (a write to `self.len`, a different location) does not
    /// invalidate it under Stacked Borrows. An **inline** destination must
    /// re-derive the slot pointer and re-test the tag after every publication
    /// — the elements live inside `self` (see `retain_mut_impl` for the
    /// per-location rule) — and may spill mid-iteration, so it keeps the
    /// general per-element form below. The split is not something the
    /// optimizer recovers on its own: in the release asm the general loop
    /// reloads the block pointer and capacity and re-tests the tag per
    /// element, while the heap loop below keeps both in registers.
    fn extend_from_iter<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut it = iter.into_iter();
        let (lo, _) = it.size_hint();
        self.reserve(lo);
        if size_of::<T>() == 0 {
            // A ZST never allocates, so each item only bumps the count; the
            // value's one conceptual destructor is transferred to the count
            // (ZST protocol site 1), exactly as in `push`.
            for x in it {
                assert!(!self.is_heap(), "ZST vectors are never heap-backed");
                assert!(self.raw_len() < isize::MAX as usize, "capacity overflow");
                self.len = self.raw_len() + 1;
                mem::forget(x);
            }
            return;
        }
        if self.is_heap() {
            // Single-entry fast loop: `base`/`cap`/`len` are hoisted and stay
            // in registers. No back edge re-enters it (a re-entering loop
            // would force LLVM to reload the state at the join — verified in
            // the release asm), so the lying-`size_hint` remainder falls
            // through to the general loop below instead, which re-derives and
            // pushes as needed.
            let base = self.base_ptr();
            let cap = self.heap_cap();
            let mut len = self.raw_len();
            while len < cap {
                let Some(x) = it.next() else { return };
                // Safety: heap state with `len < cap`, so slot `len` is in
                // bounds and uninitialized; `write` moves `x` in without
                // dropping it. `base` stays valid across the publication
                // below: it points into the independent block, and writing
                // `self.len` does not touch that allocation's Stacked-Borrows
                // stacks.
                unsafe { base.add(len).write(x) };
                len += 1;
                // Published before the next `it.next()` (user code, may
                // panic), so a panic leaves exactly the elements written so
                // far counted. `len <= cap <= MAX`, so no tag bit is touched.
                self.len = HEAP_TAG | len;
            }
            // Reservation exhausted: an iterator that yielded more than its
            // `size_hint` promised. The general loop handles the remainder.
        }
        // General loop (inline state, or the exhausted-heap remainder):
        // per-element re-derivation — the length publication invalidates a
        // hoisted pointer when the elements live inside `self` — and `push`
        // may spill, after which the per-element `is_heap` test picks up the
        // heap branch.
        for x in it {
            let len = self.raw_len();
            if len < self.capacity() {
                // Safety: `len < capacity`, so slot `[len]` is in bounds and
                // uninitialized; `write` moves `x` in without dropping it, and
                // the tag bits are preserved by the masked length write.
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
                // Spare capacity exhausted; `push` spills or reallocates.
                self.push(x);
            }
        }
    }

    /// Appends exactly `additional` elements produced by `f(k)` for `k` in
    /// `0..additional`, assuming capacity for them has already been reserved by
    /// the caller.
    ///
    /// Writes directly into the spare slots (no per-element state/capacity
    /// re-check, unlike `push`) and advances the length *after* each write, so a
    /// panic inside `f` leaves the container holding exactly the elements
    /// produced so far, which its `Drop` cleans up.
    fn fill_with<F>(&mut self, additional: usize, mut f: F)
    where
        F: FnMut(usize) -> T,
    {
        for k in 0..additional {
            let len = self.raw_len();
            debug_assert!(len < self.capacity());
            let value = f(k);
            // Safety: `len < capacity` (reserved by the caller), so slot `[len]`
            // is in bounds and uninitialized; `write` moves the value in without
            // dropping it, and the tag bits are preserved by the masked length
            // write. For a ZST this write is a no-op that still *consumes* the
            // value (ZST protocol site 1: the count owns its destructor).
            unsafe {
                let p = if self.is_heap() {
                    self.data.heap.0.cast::<T>().as_ptr().add(len)
                } else {
                    addr_of_mut!(self.data.inline).cast::<T>().add(len)
                };
                p.write(value);
            }
            self.len = (self.len & HEAP_TAG) | (len + 1);
        }
    }

    /// Shortens the vector, dropping the removed elements.
    pub fn truncate(&mut self, new_len: usize) {
        let len = self.raw_len();
        if new_len >= len {
            return;
        }
        // Publish the new length *before* running destructors (std parity). If
        // a destructor panics mid-slice, the already-dropped slots are no
        // longer counted, so the container's later `Drop` leaks the un-dropped
        // tail instead of re-dropping the slots that did run — a double drop is
        // UB. (The alternative, dropping first and publishing after, is exactly
        // the bug std fixed in `Vec::truncate`.)
        self.len = (self.len & HEAP_TAG) | new_len;
        if mem::needs_drop::<T>() {
            // Derive the pointer *after* the `len` write: under Stacked Borrows
            // that write is a use of `self` and would invalidate an earlier raw
            // derivation.
            // Safety: `[new_len, len)` were initialized; the container no longer
            // claims them, so this drop neither aliases nor re-drops.
            unsafe {
                let p = self.base_ptr().add(new_len);
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len - new_len));
            }
        }
    }

    /// Drops all elements; the storage state is unchanged.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Resizes the vector to `new_len`, cloning `value` to fill any new slots.
    ///
    /// Parity with `Vec::resize`: when growing, `value` is cloned into the first
    /// `new_len - len - 1` new slots and the **original is moved** into the last
    /// new slot, so `T::clone` runs `new_len - len - 1` times and the total
    /// number of destructors is exactly `new_len` (the moved original adds no
    /// extra drop). This mirrors [`from_elem`](Self::from_elem), which likewise
    /// moves its argument into one slot and clones only the rest.
    #[track_caller]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        let len = self.raw_len();
        if new_len > len {
            let additional = new_len - len;
            self.reserve(additional);
            // Clone into the first `additional - 1` slots; the original `value`
            // is moved into the last slot. `tail_fill` is exactly that shape;
            // for `T: Copy` (specialization) it is one length publication plus
            // the O(log len) doubling copies of `spec::bulk`, which is sound
            // because no user code runs in between and a moved original is
            // unobservable for a type with no destructor.
            #[cfg(feature = "specialization")]
            {
                let l = self.raw_len();
                // Safety: `reserve(additional)` guaranteed capacity for
                // `l + additional` slots, and slots `[l, l + additional)` are
                // uninitialized. `additional >= 1` because `new_len > len`.
                // For a ZST the writes are no-ops and the ownership transfer
                // happens through the published count.
                unsafe { T::tail_fill(value, self, l, additional) };
            }
            #[cfg(not(feature = "specialization"))]
            {
                // `fill_with` accepts 0 here.
                self.fill_with(additional.saturating_sub(1), |_| value.clone());
                // `additional >= 1` because `new_len > len`. Re-derive the base
                // pointer *after* the `fill_with` length writes: under Stacked
                // Borrows each write to `self.len` is a use of `self` that would
                // invalidate an earlier raw derivation.
                let l = self.raw_len();
                // Safety: `reserve(additional)` guaranteed capacity for `l + 1`
                // slots, and slot `l` is uninitialized. For a ZST the write
                // consumes `value` and transfers its conceptual destructor to
                // the count, keeping the drop total at `new_len`.
                unsafe {
                    self.base_ptr().add(l).write(value);
                }
                self.len = (self.len & HEAP_TAG) | (l + 1);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Resizes the vector to `new_len`, using `f` to build any new elements.
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.raw_len();
        if new_len > len {
            let additional = new_len - len;
            self.reserve(additional);
            self.fill_with(additional, |_| f());
        } else {
            self.truncate(new_len);
        }
    }

    /// Creates a vector with `n` copies of `elem` (like `vec![elem; n]`).
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
        out.reserve(n);
        #[cfg(feature = "specialization")]
        {
            // Move the owned `elem` into slot 0, then clone *that* slot for
            // the rest (exactly `n` values, hence `n` destructors —
            // `vec![elem; n]` semantics; cloning `elem` into every slot
            // would create `n + 1` destructors). Under `T: Copy` this is one
            // length publication plus the O(log n) doubling copies of
            // `spec::bulk` (nothing user-visible can run in between); the
            // stable path below is the same accounting, element-wise.
            // Safety: capacity `n` was reserved, the container is empty, and
            // slots `[0, n)` are uninitialized. For a ZST the write consumes
            // `elem` and transfers its conceptual destructor to the count
            // (ZST protocol site 1).
            unsafe { T::seed_fill(elem, &mut out, n) };
        }
        #[cfg(not(feature = "specialization"))]
        {
            // Move the owned `elem` into slot 0, then clone *that* slot for
            // the rest, so exactly `n` values (and hence `n` destructors)
            // result — `vec![elem; n]` semantics. (Cloning `elem` into every
            // slot would create `n` clones plus the original, i.e. `n + 1`
            // destructors.)
            //
            // `reserve` may have spilled, so the length writes below preserve
            // the heap tag via the usual masked form.
            let dst0 = out.base_ptr();
            // Safety: capacity `n` was reserved and slot 0 is uninitialized.
            // For a ZST this consumes `elem` and transfers its conceptual
            // destructor to the count (ZST protocol site 1).
            unsafe { dst0.write(elem) };
            out.len = (out.len & HEAP_TAG) | 1;
            for i in 1..n {
                // Safety: slot 0 is initialized (`base_ptr_const` handles
                // both states); re-derived each iteration because the
                // `out.len` write below would invalidate a hoisted pointer in
                // the inline state. (ZST protocol: for a ZST slot 0 was never
                // *written*, but reading it through a `&T` touches no bytes,
                // and the resulting clone is counted — one conceptual
                // destructor per counted element.)
                let v = unsafe { (&*out.base_ptr_const()).clone() };
                // Safety: slot [i] is uninitialized (capacity `n` reserved).
                unsafe { out.base_ptr().add(i).write(v) };
                out.len = (out.len & HEAP_TAG) | (i + 1);
            }
        }
        out
    }

    /// Inserts `value` at `index`, shifting later elements right.
    #[track_caller]
    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.raw_len();
        assert!(
            index <= len,
            "insertion index (is {index}) should be <= len (is {len})"
        );
        self.reserve(1);
        let len = self.raw_len();
        let p = self.base_ptr();
        // Safety: shift then write; `ptr::copy` handles overlap. For a ZST both
        // the shift and the write touch no bytes, but the write still *consumes*
        // `value`, transferring its one conceptual destructor to the count (ZST
        // protocol site 1), exactly as in `push`.
        unsafe {
            if index < len {
                ptr::copy(p.add(index), p.add(index + 1), len - index);
            }
            p.add(index).write(value);
        }
        self.len = (self.len & HEAP_TAG) | (len + 1);
    }

    /// Removes and returns the element at `index`, shifting later elements
    /// left.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T {
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
    /// element.
    #[track_caller]
    pub fn swap_remove(&mut self, index: usize) -> T {
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

    /// Retains only the elements specified by the predicate, passing a mutable
    /// reference to each element.
    ///
    /// Removes all elements `e` for which `f(&mut e)` returns `false`, visiting
    /// each element exactly once in the original order and preserving the
    /// relative order of the retained ones. The vector never reallocates: both
    /// the inline and the heap state are updated in place, and
    /// [`is_heap`](Self::is_heap) (and a heap block's capacity) is unchanged.
    ///
    /// # Panic safety
    ///
    /// If `f` panics, or the [`Drop`] of a removed element panics, a
    /// `BackshiftOnDrop`-style guard restores a consistent state: the vector
    /// ends up holding exactly the elements retained so far followed by the
    /// untouched tail (the element the closure was examining when it panicked
    /// is kept; an element whose *own destructor* panicked is leaked). No
    /// element is ever dropped twice and the length never lies.
    ///
    /// The in-place scan and its panic guard are shared with
    /// `CovariantCow::retain_mut` (see `retain_mut_impl`).
    pub fn retain_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        retain_mut_impl(self, f);
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements `e` for which `f(&e)` returns `false`. See
    /// [`Self::retain_mut`] for the in-place and panic-safety guarantees.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|x| f(x));
    }

    /// Removes the specified range from the vector, in bulk, returning all
    /// removed elements as an iterator.
    ///
    /// When the returned [`Drain`] is dropped (exhausted or not, normally or
    /// while unwinding), the tail is shifted back and the length restored: the
    /// vector then contains exactly the elements outside `range`, in their
    /// original order. The inline/heap state never changes — a heap-backed
    /// vector keeps its block and capacity, an inline vector stays inline.
    ///
    /// [`Drain`] moves the whole container out and back rather than borrowing
    /// it; see its type-level documentation for why.
    ///
    /// # Panics
    ///
    /// Panics if the range's start is greater than its end, or if its end is
    /// greater than the vector's length.
    ///
    /// # Leaking
    ///
    /// If the returned [`Drain`] is leaked (e.g. with `core::mem::forget`), the
    /// whole container — every element and, if heap-backed, the heap block — is
    /// leaked. The caller's vector keeps a valid empty placeholder. As with
    /// `Vec::drain`, leaking the iterator is a (safe) memory leak, never a
    /// double-drop hazard.
    ///
    /// Exception: an **empty** range (e.g. `drain(2..2)`) is a no-op that never
    /// takes the container, so leaking it leaks nothing and leaves the vector
    /// fully intact.
    #[track_caller]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, 'a, T, N>
    where
        R: RangeBounds<usize>,
    {
        let len = self.raw_len();
        let (begin, end) = drain_bounds(&range, len);
        if begin == end {
            // Empty range: there is nothing to drain, so skip taking the
            // container — `mem::take` would memcpy the entire struct (for a
            // large inline `N`, that is the whole inline array). The returned
            // `Drain` owns nothing: its `Drop` is a no-op and the caller's
            // vector is left untouched. (A pleasant side effect: forgetting an
            // empty-range `Drain` no longer leaks the container.)
            return Drain {
                vec: ManuallyDrop::new(CovariantVec::new()),
                slot: self,
                owns: false,
                begin,
                pos: begin,
                end,
                back: end,
                original_len: len,
            };
        }
        // Move the entire container into the drain and leave an empty inline
        // placeholder behind. The drain now owns the kept prefix, the drained
        // region and the tail; `Drain`'s `Drop` puts everything but the drained
        // elements back — even when unwinding.
        //
        // A heap-backed container is handed over as a *state swap* instead of a
        // `mem::take`: only `(ptr, cap)` and the tagged length move, and the
        // caller's slot keeps its (now inert) inline bytes. `mem::take` would
        // instead memcpy the whole `size_of::<Self>()` — for a heap-backed
        // container that is mostly the `N · size_of::<T>()` inline array, which
        // is dead padding in the heap state — and copy it back on rollback.
        // See `DrainRollback::drop` for the matching write-back. The inline
        // case still uses `mem::take`: there the array holds the live elements,
        // so they must move into the drain either way.
        let vec = if self.is_heap() {
            // Safety: heap state.
            let (ptr, cap) = unsafe { self.data.heap };
            let raw_len = self.len;
            // Leave the slot an empty inline vector by writing only `len`. The
            // stale `(ptr, cap)` bytes still in the union are inert: with
            // `raw_len == 0` and no tag, no element slot is ever read, and the
            // `data` field is only ever reached through `addr_of!`/`addr_of_mut!`
            // (place access, not a typed union-field read). This is the same
            // state `new()` produces, modulo bytes nothing looks at.
            self.len = 0;
            // Safety: `ptr`/`cap` describe the live block this vector owned and
            // `raw_len` is its tagged length (`HEAP_TAG | len`, `len <= cap`);
            // ownership moves into the returned vector, and `self` no longer
            // refers to the block.
            CovariantVec {
                data: Data { heap: (ptr, cap) },
                len: raw_len,
                marker: PhantomData,
            }
        } else {
            mem::take(self)
        };
        Drain {
            vec: ManuallyDrop::new(vec),
            slot: self,
            owns: true,
            begin,
            pos: begin,
            end,
            back: end,
            original_len: len,
        }
    }

    // ----------------------------------------------------------------------
    // Capacity management
    // ----------------------------------------------------------------------

    /// Reserves room for at least `additional` more elements.
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
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

    /// Grows capacity so that at least `needed` elements fit.
    #[track_caller]
    fn grow_to_at_least(&mut self, needed: usize) {
        if let Err(e) = self.try_grow(needed, false) {
            e.abort();
        }
    }

    /// Fallible capacity-growth core shared by `reserve`/`reserve_exact` and
    /// `try_reserve`/`try_reserve_exact`.
    ///
    /// `exact` requests `needed` verbatim instead of the amortized `grow_to`
    /// curve. Capacity overflow and allocation failure are both reported as
    /// `Err`, and the container is left unchanged in every error case.
    fn try_grow(&mut self, needed: usize, exact: bool) -> Result<(), TryReserveError> {
        if size_of::<T>() == 0 {
            // No storage, but the length field still bounds `needed`.
            return if needed > isize::MAX as usize {
                Err(TryReserveError::capacity_overflow())
            } else {
                Ok(())
            };
        }
        let cur = self.capacity();
        let new_cap = if exact {
            needed
        } else {
            let grown = Self::grow_to(cur);
            if grown < needed {
                needed
            } else {
                grown
            }
        };
        if new_cap > isize::MAX as usize {
            return Err(TryReserveError::capacity_overflow());
        }
        // May itself overflow for `size_of::<T>() > 1`.
        let new_layout = try_heap_layout::<T>(new_cap)?;

        if self.is_heap() {
            let old_layout = heap_layout::<T>(cur);
            // Safety: `old_layout` matches the live allocation.
            let new_ptr =
                unsafe { realloc(self.data.heap.0.as_ptr(), old_layout, new_layout.size()) };
            let nn = NonNull::new(new_ptr).ok_or(TryReserveError::alloc_error(new_layout))?;
            // Update the tag-consistent heap metadata (a union field write is
            // safe; the heap field has no drop glue).
            self.data.heap = (nn, new_cap);
        } else {
            // Safety: non-zero layout; the fresh block is distinct from the
            // inline array, so the migration below does not overlap.
            let new_ptr = unsafe { alloc(new_layout) };
            let nn = NonNull::new(new_ptr).ok_or(TryReserveError::alloc_error(new_layout))?;
            let len = self.raw_len();
            // Safety: migrate `len` initialized inline elements (a bitwise
            // move, valid for `!Copy`), then switch to the heap state.
            unsafe {
                let src = addr_of_mut!(self.data.inline).cast::<T>();
                ptr::copy_nonoverlapping(src, nn.cast::<T>().as_ptr(), len);
                self.data.heap = (nn, new_cap);
            }
            self.len = HEAP_TAG | len;
        }
        Ok(())
    }

    /// Reserves room for exactly `additional` more elements.
    ///
    /// Like [`reserve`](Self::reserve), but requests the exact capacity instead
    /// of the amortized growth curve; prefer [`reserve`](Self::reserve) when
    /// growing repeatedly.
    #[track_caller]
    pub fn reserve_exact(&mut self, additional: usize) {
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

    /// Tries to reserve room for at least `additional` more elements, reporting
    /// failure as an error instead of panicking (capacity overflow) or aborting
    /// (allocation failure).
    ///
    /// The capacity may grow beyond `len + additional` to avoid frequent
    /// reallocations; use [`try_reserve_exact`](Self::try_reserve_exact) to
    /// request exactly that much. For a zero-sized element nothing is
    /// allocated, but the `len + additional <= isize::MAX` bound is still
    /// enforced.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveErrorKind::CapacityOverflow`] if `len + additional`
    /// overflows `usize` or exceeds the representable capacity, and
    /// [`TryReserveErrorKind::AllocError`] if the allocator cannot satisfy the
    /// request.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if needed <= self.capacity() {
            return Ok(());
        }
        self.try_grow(needed, false)
    }

    /// Tries to reserve room for exactly `additional` more elements.
    ///
    /// Like [`try_reserve`](Self::try_reserve), but requests the exact capacity.
    ///
    /// # Errors
    ///
    /// Same as [`try_reserve`](Self::try_reserve).
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let len = self.raw_len();
        let needed = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        if needed <= self.capacity() {
            return Ok(());
        }
        self.try_grow(needed, true)
    }

    /// Moves the elements back into the inline buffer and frees the heap block,
    /// if the length currently fits in `N`.
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
            // ZST can never be in the heap state, so this is always a real
            // allocation.
            debug_assert!(size_of::<T>() != 0);
            dealloc(ptr.as_ptr(), heap_layout::<T>(cap));
        }
        self.len = len;
    }

    /// Directly sets the length (same contract as `Vec::set_len`).
    ///
    /// # Safety
    ///
    /// `new_len <= capacity`; if growing, the newly exposed slots must be
    /// initialized; if shrinking, the removed elements must already have been
    /// dropped.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.len = (self.len & HEAP_TAG) | new_len;
    }

    // ----------------------------------------------------------------------
    // Drop
    // ----------------------------------------------------------------------

    /// Drops all live elements and frees the heap block if present.
    ///
    /// # Safety
    ///
    /// Must only be invoked from `Drop` (or an equivalent terminal position).
    /// It **reads no element contents** (the `may_dangle` contract).
    unsafe fn drop_contents(&mut self) {
        let len = self.raw_len();
        if self.is_heap() {
            // Safety: heap state.
            let (ptr, cap) = unsafe { self.data.heap };
            if mem::needs_drop::<T>() {
                // Safety: first `len` heap slots are initialized.
                let p = ptr.cast::<T>().as_ptr();
                unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len)) };
            }
            // ZST never reaches the heap state, so the layout is valid.
            debug_assert!(size_of::<T>() != 0);
            // Safety: layout matches the allocation.
            unsafe {
                dealloc(ptr.as_ptr(), heap_layout::<T>(cap));
            }
        } else if mem::needs_drop::<T>() {
            // Safety: first `len` inline slots are initialized.
            let p = addr_of_mut!(self.data.inline).cast::<T>();
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, len)) };
        }
    }
}

/// Internal: an owned element buffer that can be backshifted in place by
/// [`retain_mut_impl`]. Implemented for both [`CovariantVec`] and (under the
/// `cow` feature) [`CovariantCow`], so the retain algorithm and its panic guard
/// live in exactly one place.
///
/// # Safety
///
/// `backshift_base` must only be called while the container is in an **owned**
/// state (inline or heap), and the returned pointer must stay valid until the
/// next call that reborrows the container.
pub(crate) trait BackshiftTarget {
    type Elem;

    /// Number of initialized elements (tag bits masked off).
    fn backshift_raw_len(&self) -> usize;

    /// Raw pointer to the first initialized element.
    ///
    /// # Safety
    ///
    /// The container must be owned (not borrowed).
    unsafe fn backshift_base(&mut self) -> *mut Self::Elem;

    /// Sets the raw length, preserving the state tag(s).
    ///
    /// # Safety
    ///
    /// The real requirement is `n <= capacity()` with slots `[0, n)`
    /// initialized — **not** `n <=` the current raw length. The guard's rollback
    /// legitimately calls this while the container's `len` is published as `0`
    /// (see [`BackshiftGuard`]), so a `n <= raw_len` precondition would be
    /// violated by the crate's own only caller. The setter must preserve the
    /// state tag(s), as the tag bits are carried across the write.
    unsafe fn backshift_set_len(&mut self, n: usize);
}

/// Panic guard shared by [`CovariantVec::retain_mut`] and
/// [`CovariantCow::retain_mut`] (named after `Vec`'s internal
/// `BackshiftOnDrop`).
///
/// While the guard is alive the buffer obeys the *backshift invariant*:
///
/// * `[0, processed_len - deleted_cnt)` — the retained prefix, already in its
///   final positions;
/// * `[processed_len - deleted_cnt, processed_len)` — dead holes (removed
///   elements already dropped, or kept elements already shifted down);
/// * `[processed_len, original_len)` — the untouched, initialized tail;
/// * the container's `len` field is published as `0` for the scan's duration
///   (std-parity hardening, see `retain_mut_impl`), so the stale geometry above
///   is never observable through a re-entrant call — and every unwind path
///   goes through [`Drop::drop`] below.
///
/// On drop the tail is shifted down by `deleted_cnt` and the length is
/// restored, yielding a consistent container. The guard never runs an element
/// destructor — everything left of `processed_len` is already dropped or
/// moved, and the tail must stay alive — so the rollback itself cannot panic
/// (a `ptr::copy` plus a length fixup).
struct BackshiftGuard<'r, V: BackshiftTarget> {
    v: &'r mut V,
    processed_len: usize,
    deleted_cnt: usize,
    original_len: usize,
}

impl<V: BackshiftTarget> Drop for BackshiftGuard<'_, V> {
    fn drop(&mut self) {
        if self.deleted_cnt > 0 {
            // Safety: the source `[processed_len, original_len)` lies inside
            // the initialized tail; the destination covers dead holes plus
            // freshly copied tail bytes only. Overlap is fine: `ptr::copy` is
            // `memmove`.
            unsafe {
                let base = self.v.backshift_base();
                ptr::copy(
                    base.add(self.processed_len),
                    base.add(self.processed_len - self.deleted_cnt),
                    self.original_len - self.processed_len,
                );
            }
        }
        // Safety: after the shift, exactly `original_len - deleted_cnt` leading
        // slots are initialized (retained prefix + moved tail). This only ever
        // shrinks, so the capacity bound holds, and the setter preserves the
        // state tag(s).
        unsafe {
            self.v
                .backshift_set_len(self.original_len - self.deleted_cnt)
        };
    }
}

/// The shared `retain`/`retain_mut` core: an in-place, never-reallocating
/// two-stage scan (stage 1 does no copying; stage 2 shifts survivors down).
///
/// The closure sees each element exactly once, in order. If `f` panics, or the
/// destructor of a removed element panics, [`BackshiftGuard`] restores a
/// consistent state on unwind: the container ends up holding the elements
/// retained so far followed by the untouched tail (the element `f` was
/// examining when it panicked is kept; an element whose own destructor panicked
/// is leaked). No element is ever dropped twice and the length never lies.
///
/// Re-entrancy: deliberately **stricter than std**, and still not *safe*. For
/// the scan's duration the length is published as `0`, so a closure that
/// reaches the *same* container through interior mutability (a `RefCell`, a
/// callback, a `Weak`) observes an empty container — a re-entrant `pop` yields
/// `None`, a re-entrant `truncate`/`clear` is a no-op, and indexing panics —
/// rather than the stale length. `Vec::retain_mut` publishes no such thing
/// (its guard leaves the length stale), so the zeroing is extra hardening, not
/// parity. Re-entrant *mutation* is still unsupported: a re-entrant `push`
/// sees `len == 0` and writes into slot 0 (leaking whatever the scan had
/// there), and any re-entrant `&mut` use of the container pops the scan's
/// hoisted `base` — borrow stacks are per-location, so a reborrow of the
/// container invalidates pointers derived from an earlier one.
pub(crate) fn retain_mut_impl<V, F>(v: &mut V, mut f: F)
where
    V: BackshiftTarget,
    F: FnMut(&mut V::Elem) -> bool,
{
    let original_len = v.backshift_raw_len();
    // Nothing to restore in an empty container: don't set the guard up.
    if original_len == 0 {
        return;
    }
    let mut g = BackshiftGuard {
        v,
        processed_len: 0,
        deleted_cnt: 0,
        original_len,
    };
    // Hardening, stricter than std: publish an empty length for the scan's
    // duration so a re-entrant call cannot observe the stale geometry (a
    // re-entrant `pop` would otherwise hand out an element the scan still owns
    // — a double drop — and indexing could read a moved or dead slot). std's
    // `PanicGuard` leaves the length stale instead. The guard restores the
    // true length on every exit path.
    //
    // Order matters: publish *before* deriving `base` below, because this
    // `backshift_set_len` reborrows `g.v` and would otherwise invalidate a
    // `base` derived earlier, under Stacked Borrows.
    unsafe { g.v.backshift_set_len(0) };
    // Compute the element base pointer **once, after the guard exists**.
    // `backshift_base` takes `&mut self`, and in the inline state the elements
    // live *inside* the container, so every call reborrows it. A later reborrow
    // (a second `backshift_base()` call, or any other `&mut g.v` use)
    // invalidates the pointers derived from an earlier one — a Stacked-Borrows
    // violation. (Moving a `&mut` reference into the guard does *not* retag;
    // only reborrows do. The hazard is the repeated `backshift_base()` calls,
    // not the move.) Every loop access therefore goes through this one raw
    // pointer, and `g.v` is not touched again until the final length fixup.
    let base = unsafe { g.v.backshift_base() };
    // Stage 1 — nothing deleted yet: the retained prefix still sits in its
    // final positions, so the scan needs no copying at all.
    while g.processed_len < g.original_len {
        // Safety: `processed_len < original_len`; slot `processed_len` belongs
        // to the untouched, initialized tail.
        let cur = unsafe { base.add(g.processed_len) };
        if !f(unsafe { &mut *cur }) {
            // Advance the counters *before* dropping: if the destructor panics,
            // the guard must already count this slot as a dead hole, or the
            // rollback would keep the half-dropped element alive (and drop it a
            // second time when the container drops).
            g.deleted_cnt += 1;
            g.processed_len += 1;
            // Safety: initialized, and removed by the predicate.
            unsafe { ptr::drop_in_place(cur) };
            break;
        }
        g.processed_len += 1;
    }
    // Stage 2 — at least one hole exists: every element that survives `f` is
    // shifted down into the hole region as it is examined.
    while g.processed_len < g.original_len {
        // Safety: as in stage 1.
        let cur = unsafe { base.add(g.processed_len) };
        if !f(unsafe { &mut *cur }) {
            g.deleted_cnt += 1;
            g.processed_len += 1;
            // Safety: initialized, and removed by the predicate.
            unsafe { ptr::drop_in_place(cur) };
            continue;
        }
        // Safety: distinct slots in the same buffer (`hole < cur`).
        unsafe {
            let hole = base.add(g.processed_len - g.deleted_cnt);
            ptr::copy_nonoverlapping(cur, hole, 1);
        }
        g.processed_len += 1;
    }
    // Publish the final length (tag-preserving) and retire the guard so its —
    // by now redundant — rollback does not run.
    unsafe { g.v.backshift_set_len(g.original_len - g.deleted_cnt) };
    mem::forget(g);
}

impl<'a, T, const N: usize> BackshiftTarget for CovariantVec<'a, T, N> {
    type Elem = T;

    #[inline]
    fn backshift_raw_len(&self) -> usize {
        self.raw_len()
    }

    #[inline]
    unsafe fn backshift_base(&mut self) -> *mut T {
        self.base_ptr()
    }

    #[inline]
    unsafe fn backshift_set_len(&mut self, n: usize) {
        self.len = (self.len & HEAP_TAG) | n;
    }
}

// --------------------------------------------------------------------------
// Drop
// --------------------------------------------------------------------------

#[cfg(feature = "dropck_eyepatch")]
// Safety: the body only reads the `usize` tag and drops / deallocates; it never
// reads any `T` nor anything reachable through `'a`. `#[may_dangle]` is
// therefore sound. Note that `'a` must be listed here as well as `T` — the
// strict drop-check applies to lifetime parameters too.
unsafe impl<#[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for CovariantVec<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position, `self` is never observed again.
        unsafe { self.drop_contents() }
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'a, T, const N: usize> Drop for CovariantVec<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_contents() }
    }
}

// --------------------------------------------------------------------------
// Auto traits
// --------------------------------------------------------------------------

// `NonNull<u8>` is `!Send`/`!Sync`; restore the intended `Vec`-like semantics.
// Safety: the only owned data are `T`s (inline or heap); `PhantomData<&'a ()>`
// is `Send + Sync` for all `'a`.
unsafe impl<'a, T: Send, const N: usize> Send for CovariantVec<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for CovariantVec<'a, T, N> {}

// --------------------------------------------------------------------------
// Clone
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> CloneSink for CovariantVec<'a, T, N> {
    type Elem = T;

    #[inline]
    unsafe fn raw_base(&mut self) -> *mut T {
        // Safe operation: the caller guarantees the owned state and capacity,
        // which is exactly what `base_ptr` (tag dispatch + place access) needs.
        self.base_ptr()
    }

    #[inline]
    unsafe fn publish_len(&mut self, n: usize) {
        // `set_len` preserves the tag; the fill never changes the state.
        // Safety: `n` slots were just initialized, and `n <= capacity`.
        unsafe { self.set_len(n) };
    }
}

impl<'a, T: Clone, const N: usize> Clone for CovariantVec<'a, T, N> {
    fn clone(&self) -> Self {
        let len = self.raw_len();
        let mut out = Self::new();
        // Normalize the state: a clone whose length fits inline is *always*
        // inline, even if the source is a spilled-and-popped heap vector. A
        // ZST clone is *always* inline too, because ZSTs never spill — so a
        // `len > N` ZST vector is still inline, **not** heap. (Taking the heap
        // branch for it would `alloc` a zero-size layout, which is UB, and
        // would set `HEAP_TAG` on a ZST vector, breaking the state encoding.)
        // This avoids allocating for the clone; the heap case allocates exactly
        // `len` (not the source's possibly-larger capacity), matching
        // `Vec::clone`'s "allocate only what is needed" behavior.
        //
        // The fill itself is `clone_fill::CloneFill`: the element-wise loop
        // (length published per element, so a panic inside `T::clone` leaves
        // `out` holding exactly the clones made so far, which its `Drop`
        // cleans up) or — under the nightly-only `specialization` feature — a
        // single `ptr::copy_nonoverlapping` for `T: Copy`. See that module for
        // the Stacked-Borrows and panic-safety rationale.
        //
        // The *source* pointer is hoisted: `self` is never mutated during the
        // clone, so its provenance is stable.
        let src = self.base_ptr_const();
        if size_of::<T>() == 0 || len <= N {
            // Safety: `src` is valid for `len` reads (`self` may be inline or
            // heap; `base_ptr_const` handles both); `out` is owned, empty and
            // has capacity `N >= len` (ZSTs included: the slots are no-ops).
            unsafe { T::clone_fill(src, &mut out, len) };
        } else {
            // Non-ZST with `len > N`, so `self` is heap-backed; allocate
            // exactly `len`.
            out.alloc_heap(len);
            // Safety: as above, with `out` freshly heap-allocated with
            // capacity exactly `len`.
            unsafe { T::clone_fill(src, &mut out, len) };
        }
        out
    }

    fn clone_from(&mut self, source: &Self) {
        let len = source.raw_len();
        // Reuse the destination's storage when it already fits the source
        // length — the point of `clone_from`, since it avoids a fresh
        // allocation. The `heap_cap` probe is short-circuited away for an
        // inline target (`len <= N` already answered), so it never reads the
        // heap variant of an inline container's union.
        let reuse = size_of::<T>() == 0 || len <= N || (self.is_heap() && self.heap_cap() >= len);
        if !reuse {
            *self = source.clone();
            return;
        }
        // Drop the old contents first: a panic inside `T::clone` then leaves
        // `self` holding exactly the clones made so far (the fill publishes the
        // length per element), which `Drop` cleans up — no double drops, no
        // leaked-mismatch. The published length stays `0` until then, so the
        // partially-filled prefix is invisible to destructors mid-panic.
        self.clear();
        // Safety: `src` is valid for `len` reads (`self` may be inline or
        // heap; `base_ptr_const` handles both; ZST offsets are no-ops), and
        // `self` is owned, empty after `clear`, with capacity >= `len` (either
        // inline `N >= len`, the never-spilling ZST case, or a heap block
        // checked above).
        //
        // The ZST case: `clone_fill`'s loop reads and writes no bytes, so the
        // provenance question does not arise; it does run `T::clone` `len`
        // times, which for a side-effectful ZST `Clone` is deliberate
        // (callers count those clones), and the sink adopts the count.
        // Contrast `CovariantCow::to_mut`, which *adopts* a borrowed ZST's
        // count without cloning; the asymmetry is documented there.
        let src = source.base_ptr_const();
        unsafe { T::clone_fill(src, self, len) };
    }
}

// --------------------------------------------------------------------------
// Conversions to/from Vec
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> From<Vec<T>> for CovariantVec<'a, T, N> {
    #[track_caller]
    fn from(vec: Vec<T>) -> Self {
        // The heap branch below bypasses `new()`, so repeat the precondition.
        const { assert!(N <= isize::MAX as usize, "N must fit in isize") };
        let len = vec.len();

        if size_of::<T>() == 0 {
            // ZSTs: storage is irrelevant; take the count only. A `Vec` of ZSTs
            // can have an arbitrarily large length (it consumes no memory), so
            // enforce the same bound as `push` — otherwise a length ≥ 2^63
            // would pollute `HEAP_TAG` and break the state encoding.
            assert!(len <= isize::MAX as usize, "capacity overflow");
            // Transferring ownership by forgetting `vec` and adopting its
            // length is sound: a ZST has no storage, so neither `vec`'s drop
            // glue nor ours performs any memory access for the elements. The
            // count moves from `vec` to `out`, keeping destructor bookkeeping
            // balanced (ZST protocol sites 1/2: one conceptual drop per counted
            // element — see the crate-level "ZST ownership protocol").
            mem::forget(vec);
            let mut out = Self::new();
            out.len = len;
            return out;
        }

        if vec.capacity() > N {
            // Take over the allocation (cap may be <= N here only
            // when `N == 0`, which cannot happen since cap > N).
            let mut v = ManuallyDrop::new(vec);
            let ptr = v.as_mut_ptr();
            CovariantVec {
                data: Data {
                    heap: (
                        unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) },
                        v.capacity(),
                    ),
                },
                len: HEAP_TAG | len,
                marker: PhantomData,
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

impl<'a, T, const N: usize, const M: usize> From<[T; M]> for CovariantVec<'a, T, N> {
    fn from(arr: [T; M]) -> Self {
        let mut out = Self::new();
        out.extend(arr);
        out
    }
}

impl<'a, T, const N: usize> From<CovariantVec<'a, T, N>> for Vec<T> {
    fn from(cv: CovariantVec<'a, T, N>) -> Self {
        let cv = ManuallyDrop::new(cv);
        if cv.is_heap() {
            let len = cv.raw_len();
            // Safety: heap state.
            let (ptr, cap) = unsafe { cv.data.heap };
            // Safety: the block was allocated by the global allocator with
            // exactly `Layout::array::<T>(cap)`, which is what `Vec` expects.
            unsafe { Vec::from_raw_parts(ptr.cast::<T>().as_ptr(), len, cap) }
        } else {
            let len = cv.raw_len();
            let mut v = Vec::with_capacity(len);
            // Safety: inline state, first `len` slots initialized.
            unsafe {
                ptr::copy_nonoverlapping(cv.base_ptr_const(), v.as_mut_ptr(), len);
                v.set_len(len);
            }
            v
        }
    }
}

// --------------------------------------------------------------------------
// Deref
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> Deref for CovariantVec<'a, T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, const N: usize> DerefMut for CovariantVec<'a, T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// --------------------------------------------------------------------------
// Slice-borrowing / slice-comparison impls (all via `as_slice`)
// --------------------------------------------------------------------------

impl<'a, T, const N: usize> AsRef<[T]> for CovariantVec<'a, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, const N: usize> AsMut<[T]> for CovariantVec<'a, T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, const N: usize> Borrow<[T]> for CovariantVec<'a, T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, const N: usize> BorrowMut<[T]> for CovariantVec<'a, T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, B, const N: usize, const M: usize> PartialEq<[B; M]> for CovariantVec<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &[B; M]) -> bool {
        self.as_slice() == other
    }
}

impl<'a, 'b, T, B, const N: usize> PartialEq<&'b [B]> for CovariantVec<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &&'b [B]) -> bool {
        self.as_slice() == *other
    }
}

impl<'a, T, B, const N: usize> PartialEq<Vec<B>> for CovariantVec<'a, T, N>
where
    T: PartialEq<B>,
{
    fn eq(&self, other: &Vec<B>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

// --------------------------------------------------------------------------
// Forwarding trait impls (all via `as_slice`, no new unsafe)
// --------------------------------------------------------------------------

impl<'a, T: fmt::Debug, const N: usize> fmt::Debug for CovariantVec<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<'a, T: PartialEq, const N: usize> PartialEq for CovariantVec<'a, T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T: Eq, const N: usize> Eq for CovariantVec<'a, T, N> {}

impl<'a, T: PartialOrd, const N: usize> PartialOrd for CovariantVec<'a, T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, T: Ord, const N: usize> Ord for CovariantVec<'a, T, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, T: Hash, const N: usize> Hash for CovariantVec<'a, T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<'a, T, const N: usize> Default for CovariantVec<'a, T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, const N: usize> Extend<T> for CovariantVec<'a, T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // `size_hint().0` is trusted as a reservation, exactly as `Vec::extend`
        // does: a lying lower bound is sized up front rather than growing
        // incrementally. A `usize::MAX`-scale hint panics `"capacity overflow"`
        // *before* anything is allocated (the element count exceeds what the
        // length field can represent); only a hint that is huge but still
        // layout-valid reaches the allocator, where failure aborts. This is
        // deliberate std parity.
        self.extend_from_iter(iter);
    }
}

impl<'a, T, const N: usize> core::iter::FromIterator<T> for CovariantVec<'a, T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        out.extend(iter);
        out
    }
}

// `T: Copy` by-reference variants, mirroring `Vec`'s own impls.
impl<'a, 'b, T: Copy, const N: usize> Extend<&'b T> for CovariantVec<'a, T, N> {
    fn extend<I: IntoIterator<Item = &'b T>>(&mut self, iter: I) {
        self.extend_from_iter(iter.into_iter().copied());
    }
}

impl<'a, 'b, T: Copy, const N: usize> core::iter::FromIterator<&'b T> for CovariantVec<'a, T, N> {
    fn from_iter<I: IntoIterator<Item = &'b T>>(iter: I) -> Self {
        let mut out = Self::new();
        out.extend(iter);
        out
    }
}

// --------------------------------------------------------------------------
// IntoIter
// --------------------------------------------------------------------------

/// Consuming iterator over a [`CovariantVec`].
pub struct IntoIter<'a, T, const N: usize> {
    vec: ManuallyDrop<CovariantVec<'a, T, N>>,
    head: usize,
    tail: usize,
}

impl<'a, T, const N: usize> IntoIter<'a, T, N> {
    fn new(vec: CovariantVec<'a, T, N>) -> Self {
        let tail = vec.len();
        IntoIter {
            vec: ManuallyDrop::new(vec),
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
        unsafe { slice::from_raw_parts(self.vec.base_ptr_const().add(self.head), n) }
    }

    /// The not-yet-consumed elements, mutably.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let n = self.tail - self.head;
        // Safety: [head, tail) are initialized; `&mut self` excludes aliases.
        unsafe { slice::from_raw_parts_mut(self.vec.base_ptr().add(self.head), n) }
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
            // Safety: [head, tail) are still initialized and unused.
            let p = unsafe { self.vec.base_ptr().add(self.head) };
            unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(p, remaining)) };
        }
        if self.vec.is_heap() {
            // Safety: heap state; layout matches the allocation.
            let (ptr, cap) = unsafe { self.vec.data.heap };
            debug_assert!(size_of::<T>() != 0);
            unsafe {
                dealloc(ptr.as_ptr(), heap_layout::<T>(cap));
            }
        }
    }
}

impl<'a, T, const N: usize> Iterator for IntoIter<'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        // Safety: slot [head] is initialized.
        let p = unsafe { self.vec.base_ptr().add(self.head) };
        self.head += 1;
        Some(unsafe { p.read() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.tail - self.head;
        (n, Some(n))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IntoIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.head >= self.tail {
            return None;
        }
        self.tail -= 1;
        // Safety: slot [tail] is initialized.
        let p = unsafe { self.vec.base_ptr().add(self.tail) };
        Some(unsafe { p.read() })
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IntoIter<'a, T, N> {}
impl<'a, T, const N: usize> FusedIterator for IntoIter<'a, T, N> {}

impl<'a, T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

// The `IntoIter` drop body has exactly the same obligations as the container's
// (it reads only the tag, drops elements, and deallocates), so it can carry the
// same `#[may_dangle]` annotations — otherwise dropping a partially consumed
// iterator after `'a` expires would be rejected, asymmetrically with the
// container itself.
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<#[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for IntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'a, T, const N: usize> Drop for IntoIter<'a, T, N> {
    fn drop(&mut self) {
        // Safety: terminal position.
        unsafe { self.drop_remaining() }
    }
}

impl<'a, T, const N: usize> IntoIterator for CovariantVec<'a, T, N> {
    type Item = T;
    type IntoIter = IntoIter<'a, T, N>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, 'r, T, const N: usize> IntoIterator for &'r CovariantVec<'a, T, N> {
    type Item = &'r T;
    type IntoIter = slice::Iter<'r, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, 'r, T, const N: usize> IntoIterator for &'r mut CovariantVec<'a, T, N> {
    type Item = &'r mut T;
    type IntoIter = slice::IterMut<'r, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// --------------------------------------------------------------------------
// Drain
// --------------------------------------------------------------------------

/// A draining iterator over a [`CovariantVec`].
///
/// Created by [`CovariantVec::drain`]. Yields the elements of the requested
/// range, in order, from either end. Whenever the `Drain` is dropped —
/// exhausted or not, normally or while unwinding — the tail of the vector is
/// shifted back over the drained region and the length is restored.
///
/// # Leaking differs from `std::vec::Drain`
///
/// Because this `Drain` **owns** the container (see below), `mem::forget`-ing
/// it leaks the **entire container** — every element, including those outside
/// the drained range, plus the heap block if heap-backed — and leaves the
/// caller's vector as a valid empty placeholder. By contrast
/// `mem::forget(std::vec::Drain)` leaves the source `Vec` fully intact (only
/// the drained elements are leaked). Both are safe memory leaks, never a
/// double-drop hazard, but if you are porting `Vec` intuitions, note that the
/// *rest* of the vector is not preserved here. This is the deliberate price of
/// the ownership design described next. (An **empty** range never takes the
/// container, so forgetting it is a true no-op that leaves the vector intact —
/// the one case that matches `std`.)
///
/// # Why this does *not* borrow the vector like `std::vec::Drain`
///
/// `std`'s `Drain` is ~32 bytes and does no work at construction: it borrows
/// the `Vec` through a `NonNull` and lets the elements sit in place. That is
/// possible *only* because a `Vec`'s elements always live in an independent,
/// address-stable heap allocation, so the iterator can hold a pointer to the
/// container and operate on the block without ever touching the container
/// itself.
///
/// An inline `CovariantVec` breaks that premise: while inline, the elements
/// live **inside the union**, in the same memory as the container. A borrowed
/// `Drain` would therefore have to
///
/// * derive element pointers from `base_ptr()` while the container is
///   simultaneously reachable through a raw `NonNull`,
/// * keep `as_slice(&self) -> &[T]` and `next(&mut self) -> Option<T>` from
///   aliasing each other through that pointer under Stacked/Tree Borrows,
/// * hand-write `unsafe impl Send/Sync` (the `NonNull` poisons them), and
/// * accept the same `mem::forget` semantics as `std` — i.e. the caller's
///   vector is left with a stale length and arbitrary contents.
///
/// None of that is unprovable; `std` proves it every day. But it is an entire
/// second, independent aliasing model whose only payoff is saving one
/// `O(1)` memcpy at construction — and it would *lose* the property that
/// `mem::forget(drain)` leaves the caller with a valid, empty, usable vector.
///
/// Moving the whole container into the `Drain` instead makes every aliasing
/// question trivial: the `Drain` *owns* the container, so `as_slice` is an
/// ordinary borrow, `Send`/`Sync` are inherited, and `forget` is a plain safe
/// leak with the caller holding an empty placeholder. The cost is one move of
/// the container at construction and one at drop — independent of the number
/// of elements drained, and invisible next to dropping a non-`Copy` element
/// even once.
///
/// A **heap-backed** container is moved as a state swap (only `(ptr, cap)` and
/// the tagged length), so its cost is `O(1)` — the `N · size_of::<T>()` inline
/// array is dead padding there and is never touched. An **inline** container
/// must move its `size_of::<CovariantVec>() ≈ N · size_of::<T>() + word` bytes
/// (twice: out at construction, back at drop), because the elements live in
/// the union itself and the `Drain` owns them. So `N` is the knob to watch
/// only for drains of vectors that stay inline — for the heap case it is
/// irrelevant. Concretely: draining a *single* element out of a full inline
/// vector still moves `2 · size_of::<Self>()` bytes, independent of how wide
/// the drained range is; the container-sized move is the price of the
/// aliasing and `mem::forget` guarantees above, not of the range width.
/// (`keep_rest` skips the shift-back when nothing was yielded, so that path is
/// a single move, not two.)
///
/// This is a deliberate trade, not an oversight. If a future change tries to
/// "optimize" this into a borrowed iterator, the aliasing argument above has
/// to be discharged first — and the `mem::forget` guarantee has to be
/// re-earned or explicitly given up in the docs.
pub struct Drain<'r, 'a, T, const N: usize> {
    /// The entire container, moved out of the caller's `&mut` and restored on
    /// drop. `ManuallyDrop` so the rollback can hand it back by value exactly
    /// once.
    vec: ManuallyDrop<CovariantVec<'a, T, N>>,
    /// Where `vec` goes when the drain finishes (the caller's slot, currently
    /// holding an empty placeholder).
    slot: &'r mut CovariantVec<'a, T, N>,
    /// Whether `vec` actually owns the caller's container. `false` only for an
    /// empty range, where the container was never taken: `vec` is then a
    /// throwaway empty vector and `Drop` must leave `slot` untouched.
    owns: bool,
    /// Start of the drained region; also where the tail is shifted back to.
    begin: usize,
    /// Front read cursor into the drained region.
    pos: usize,
    /// End of the drained region (fixed).
    end: usize,
    /// Back read cursor into the drained region.
    back: usize,
    /// Raw length of the container when `drain` was called. Read only on the
    /// `owns == true` paths (`drop_impl`, `keep_rest`); the empty-range
    /// placeholder never consults it (its rollback is skipped entirely).
    original_len: usize,
}

impl<'r, 'a, T, const N: usize> Drain<'r, 'a, T, N> {
    /// The not-yet-yielded elements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let n = self.back - self.pos;
        // Safety: `[pos, back)` is an initialized sub-range of the drained
        // region; the container's storage (hence `base_ptr_const`) is unchanged
        // for the drain's whole lifetime.
        unsafe { slice::from_raw_parts(self.vec.base_ptr_const().add(self.pos), n) }
    }

    /// The not-yet-yielded elements as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let n = self.back - self.pos;
        // Safety: `[pos, back)` is an initialized sub-range of the drained
        // region and `&mut self` excludes aliases. (`base_ptr` asserts the
        // non-borrowed state; a `Drain` never owns a borrowed container.)
        unsafe { slice::from_raw_parts_mut(self.vec.base_ptr().add(self.pos), n) }
    }

    /// Keeps every not-yet-yielded element in the container, so only the
    /// already-yielded ones are removed (std parity with the unstable
    /// `Vec::Drain::keep_rest`).
    ///
    /// This is the safe way to abandon a `drain` early. It is the opposite
    /// trade from `mem::forget(drain)`, which leaks the *whole* container
    /// (drop glue and heap block included): here the container is rolled back
    /// into the caller with exactly `begin + kept + tail` elements, where
    /// `kept` is the number of elements the iterator has not yielded yet.
    /// Already-yielded elements stay the caller's responsibility — they were
    /// moved out by `next`/`next_back` and this method does not drop them.
    pub fn keep_rest(&mut self) {
        if !self.owns {
            // Empty range: the container was never taken; nothing to keep.
            return;
        }
        // Shift the unyielded run to the start of the drained region, over the
        // slots of the elements already yielded (which the caller now owns).
        let kept = self.back - self.pos;
        // `pos == begin` (only `next_back` has been used) makes this a
        // self-move: `ptr::copy` would be a correct no-op, but for a wide
        // inline `N` the wasted memmove is O(N), so skip it. The rollback
        // below is unchanged — it shifts from `end` to `begin + kept == back`
        // either way.
        if self.pos != self.begin {
            // Safety: `[pos, back)` is initialized; `[begin, pos)` holds dead
            // slots (yielded). `ptr::copy` handles the overlap. `base` is
            // derived once and consumed by this single copy before `self.vec`
            // is reborrowed, matching the rule in `DrainRollback::drop`.
            unsafe {
                let base = self.vec.base_ptr();
                ptr::copy(base.add(self.pos), base.add(self.begin), kept);
            }
        }
        // Roll the untouched tail back behind the kept run and hand the
        // container to the slot — precisely what `Drop` would do for the range
        // `[begin + kept, end)`.
        let rb = DrainRollback {
            vec: &mut self.vec,
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

    /// Shared `Drop` body.
    ///
    /// Drops the not-yet-yielded drained elements, then rolls the container
    /// back into the caller's slot. If an element destructor panics, the
    /// rollback still runs during unwinding: the tail is shifted over the
    /// entire drained region, so any drained element whose destructor has not
    /// run yet has it *skipped* — its storage is reclaimed, but the destructor
    /// is never invoked (a deliberate leak-on-panic) — rather than being
    /// dropped a second time. The caller's vector is left in a valid state.
    fn drop_impl(&mut self) {
        if !self.owns {
            // Empty range: the caller's container was never taken, so there is
            // nothing to roll back. `vec` is a throwaway empty inline vector
            // with no resources; skipping its `Drop` (it is `ManuallyDrop`)
            // leaks nothing.
            return;
        }
        let mut rb = DrainRollback {
            vec: &mut self.vec,
            slot: &mut *self.slot,
            begin: self.begin,
            end: self.end,
            original_len: self.original_len,
            pos: self.pos,
            back: self.back,
        };
        while rb.pos < rb.back {
            // Safety: `pos < back <= end <= original_len`; slot `pos` holds an
            // element that has not been yielded yet. (`pos` is advanced before
            // the drop purely for this loop's own bookkeeping — the rollback
            // guard does *not* read `pos`; it shifts using `begin`/`end`/
            // `original_len` and never runs destructors, so the ordering is not
            // load-bearing for the unwind path.)
            //
            // `rb.vec.base_ptr()` is re-derived every iteration rather than
            // hoisted across the `rb.pos += 1` bookkeeping. This is required
            // for the inline state, where the elements share `rb.vec`'s own
            // allocation (so touching `rb.vec` invalidates a hoisted pointer
            // under Stacked Borrows); for the heap state a hoisted pointer would
            // be sound (its provenance is in the independent heap allocation).
            // Re-deriving is free, so it is done uniformly. See the longer note
            // on `retain_mut_impl` for the per-location rule.
            let p = unsafe { rb.vec.base_ptr().add(rb.pos) };
            rb.pos += 1;
            unsafe { ptr::drop_in_place(p) };
        }
        // `rb` drops here — or during unwinding — completing the rollback.
    }
}

/// Rollback guard for [`Drain`]: shifts the tail back over the drained region,
/// restores the length, and hands the container back to the caller's slot.
/// Runs on the normal *and* the unwinding path; every step (`ptr::copy`, the
/// length fixup, and either the move into the slot or the heap-words write-back
/// — neither of which can panic) is panic-free, so the rollback always
/// completes.
struct DrainRollback<'f, 'a, T, const N: usize> {
    vec: &'f mut ManuallyDrop<CovariantVec<'a, T, N>>,
    slot: &'f mut CovariantVec<'a, T, N>,
    begin: usize,
    end: usize,
    original_len: usize,
    pos: usize,
    back: usize,
}

impl<T, const N: usize> Drop for DrainRollback<'_, '_, T, N> {
    fn drop(&mut self) {
        // An empty drained range needs no rollback at all: the container is
        // untouched and the length is unchanged, so skip the (otherwise
        // full-tail) memmove and the redundant `set_len`. std's `Drain::drop`
        // has the analogous skip.
        if self.begin != self.end {
            // Safety: `[end, original_len)` is the untouched initialized tail;
            // `[begin, end)` is the drained region, all of whose slots are dead
            // from the container's point of view (yielded, or dropped/abandoned
            // above). `ptr::copy` handles the overlap.
            //
            // `base` is derived once and consumed by this single `ptr::copy`
            // before `self.vec` is reborrowed again (`&mut **self.vec` below),
            // so no Stacked-Borrows reborrow trap arises here.
            unsafe {
                let base = self.vec.base_ptr();
                ptr::copy(
                    base.add(self.end),
                    base.add(self.begin),
                    self.original_len - self.end,
                );
            }
            let v = &mut **self.vec;
            // Safety: exactly `original_len - (end - begin)` leading slots are
            // initialized (kept prefix + shifted tail); this only shrinks, and
            // `set_len` preserves the inline/heap tag.
            unsafe { v.set_len(self.original_len - (self.end - self.begin)) };
        }
        // Hand the container back.
        if self.vec.is_heap() {
            // Heap state: write back only `(ptr, cap)` + the tagged length,
            // mirroring the state swap in `drain`. The slot is an empty inline
            // vector, so its inline bytes are never read once the tag is set —
            // moving the whole struct back would memcpy `N · size_of::<T>()`
            // bytes of dead padding. (The placeholder is an empty vector with
            // no resources, so skipping its `Drop` — the assignment in the
            // other branch would run it — is unobservable: it has no elements
            // and no block to free.)
            //
            // Safety: `self.vec` still owns the block (it was moved, not
            // copied, out of the slot by `drain`); this transfers ownership
            // back. `self.vec` is `ManuallyDrop` and is never read again, so
            // the block is freed exactly once, by the slot.
            let (ptr, cap) = unsafe { self.vec.data.heap };
            self.slot.data.heap = (ptr, cap);
            self.slot.len = self.vec.len;
        } else {
            // The placeholder currently in the slot is an empty inline vector —
            // assigning over it drops that no-op first. After the take the
            // `ManuallyDrop` is logically empty, so nothing is ever dropped
            // twice.
            // Safety: `self.vec` is taken exactly once, here, at the end of the
            // guard's life; nothing reads it afterwards.
            *self.slot = unsafe { ManuallyDrop::take(self.vec) };
        }
    }
}

impl<'r, 'a, T, const N: usize> Iterator for Drain<'r, 'a, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.pos == self.back {
            return None;
        }
        // Safety: slot `pos` is initialized and not yet yielded. The value is
        // moved out; its slot becomes dead and will be overwritten by the
        // rollback without ever being dropped again.
        unsafe {
            let p = self.vec.base_ptr().add(self.pos);
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

impl<'r, 'a, T, const N: usize> DoubleEndedIterator for Drain<'r, 'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.pos == self.back {
            return None;
        }
        self.back -= 1;
        // Safety: as in `next` (slot `back` is initialized, not yet yielded).
        unsafe {
            let p = self.vec.base_ptr().add(self.back);
            Some(p.read())
        }
    }
}

impl<'r, 'a, T, const N: usize> ExactSizeIterator for Drain<'r, 'a, T, N> {}
impl<'r, 'a, T, const N: usize> FusedIterator for Drain<'r, 'a, T, N> {}

impl<'r, 'a, T: fmt::Debug, const N: usize> fmt::Debug for Drain<'r, 'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

// The `Drain` drop body has exactly the same obligations as the container's and
// the `IntoIter`'s (it reads only the tag/length bookkeeping, runs `T`'s drop
// glue, shifts raw bytes, and finally moves the container back into the
// caller's slot — never reading any `T` contents nor anything reachable through
// `'a`; the placeholder being overwritten is empty), so it carries the same
// `#[may_dangle]` annotations.
#[cfg(feature = "dropck_eyepatch")]
unsafe impl<'r, #[may_dangle] 'a, #[may_dangle] T, const N: usize> Drop for Drain<'r, 'a, T, N> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<'r, 'a, T, const N: usize> Drop for Drain<'r, 'a, T, N> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(test)]
mod error_impl_tests {
    use alloc::string::ToString;

    use super::*;

    // `core::error::Error` exists only on toolchains >= 1.81; the cfg is emitted
    // by `build.rs` from the building compiler's version.
    #[cfg(has_error_trait)]
    #[test]
    fn try_reserve_error_implements_core_error() {
        let e = TryReserveError::capacity_overflow();
        let dyn_err: &dyn core::error::Error = &e;
        assert_eq!(dyn_err.to_string(), "capacity overflow");
        // Neither kind wraps another error.
        assert!(dyn_err.source().is_none());
    }
}
