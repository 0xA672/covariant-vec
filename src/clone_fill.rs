//! The one clone loop used by `Clone` (both containers), `CovariantCow::to_mut`
//! and `CovariantCow::try_materialize`.
//!
//! Every clone in this crate has the same shape — walk `[0, len)`, clone the
//! source element, write it into the destination, publish the new length — and
//! the same two constraints:
//!
//! * **Stacked Borrows.** In the inline state the destination's elements live
//!   *inside* the destination container, so a slot pointer derived from the
//!   container is invalidated by the length publication that follows it. The
//!   element-wise loop therefore re-derives the slot pointer (via
//!   [`CloneSink::raw_base`]) after every [`CloneSink::publish_len`] call.
//!   [`CloneSink`] deliberately exposes only those two reborrowing operations
//!   as low-level primitives, so the discipline is expressed by construction:
//!   there is no way to hold a slot pointer across a publish. The range-fill
//!   convenience ([`CloneSink::fill_range`]) is built on them and never hands
//!   a pointer out.
//!   (In the heap state a hoisted pointer would be sound — it points into an
//!   independent block — and the re-derivation folds there anyway: `raw_base`
//!   is a constant offset in the inline state and a load of a field that
//!   `publish_len` never writes in the heap state. Verified in the release
//!   asm: neither loop body reloads the destination base, the only per-element
//!   memory traffic being the element store and the length publication.)
//! * **Panic safety.** The length is published *after* each element is
//!   written, so a panic inside `T::clone` leaves the destination holding
//!   exactly the elements cloned so far, which its `Drop` cleans up
//!   (leak-not-double-drop).
//!
//! # The `T: Copy` fast path
//!
//! Under the nightly-only `specialization` cargo feature, a `T: Copy` element
//! takes a single `ptr::copy_nonoverlapping` instead of the loop. The
//! specialization is keyed **exactly** on `T: Copy`, matching what `std`'s
//! `Vec` does: `Copy` is a contract that `clone` is a bitwise copy, so the
//! substitution is unobservable. It is deliberately *not* keyed on
//! `!needs_drop::<T>()`, which is a strict superset of `Copy` — a no-drop
//! `!Copy` type with a side-effectful `Clone` would silently stop running
//! `clone`, a semantic deviation `std` refuses to make.
//!
//! The same substitution covers the range form
//! ([`CloneFill::clone_fill_range`], reached through
//! [`CloneSink::fill_range`]): `T: Copy` copies the whole range in one go,
//! `T: Clone` runs the element-wise loop starting at `start`. That is what
//! lets [`CovariantCow::append`](crate::CovariantCow::append)'s
//! borrowed-source path bulk-copy into an already-populated destination.
//!
//! On stable (feature off) every element type takes the element-wise loop; the
//! two are the same code path with the same publication points, so the feature
//! changes speed only, never behaviour or panic-safety.
//!
//! ## Why the *full* `specialization` feature, and why it is sound here
//!
//! `min_specialization`, the sound subset, rejects this shape outright —
//! specializing on a trait bound rather than a concrete type is exactly what
//! it forbids (`error: cannot specialize on trait `Copy``). Keying on `T: Copy`
//! is the only std-parity option, so the full feature is required.
//!
//! `specialization` is incomplete, and its known unsoundness is about
//! *lifetime* specialization (and associated-item specialization). This crate
//! uses the narrow form that `std` itself relies on: two impls whose self type
//! is the *same* type parameter, differing only by a strictly stronger bound
//! (`T: Copy` implies `T: Clone`, its supertrait), specializing a *method*,
//! with no lifetimes in the trait at all. Nothing here can observe a lifetime
//! or an associated item through the specialization, so the unsound corner is
//! unreachable. The crate carries `#[allow(incomplete_features)]` next to the
//! feature gate; it is scoped to the feature, so stable builds never see it.

/// Destination for the clone loops: the two container operations they need,
/// plus the range-fill convenience built on them.
///
/// The two primitives reborrow the container, which is what makes the inline
/// state's Stacked-Borrows discipline expressible: a pointer derived from
/// `raw_base` must never be held across a `publish_len` call.
pub(crate) trait CloneSink {
    type Elem;

    /// Raw pointer to slot 0 of the destination's **owned** storage.
    ///
    /// # Safety
    ///
    /// The sink must be in an owned state (never borrowed) with capacity at
    /// least the number of slots about to be filled.
    unsafe fn raw_base(&mut self) -> *mut Self::Elem;

    /// Publishes `n` as the length, preserving the state tag(s).
    ///
    /// # Safety
    ///
    /// Slots `[0, n)` must be initialized.
    unsafe fn publish_len(&mut self, n: usize);

    /// Fills slots `[start, start + len)` by cloning from `src`, publishing
    /// the length after each element (a single bulk copy and publication for
    /// `T: Copy`).
    ///
    /// This is the range counterpart of [`CloneFill::clone_fill`], for sinks
    /// that already hold `start` elements. It is a provided method so the
    /// container impls stay down to the two reborrowing primitives above; the
    /// body forwards to the element-keyed [`CloneFill`], which is where the
    /// `T: Copy` specialization lives (a container-keyed trait cannot see the
    /// element's `Copy`-ness).
    ///
    /// # Safety
    ///
    /// * `src` must be valid for reads of `len` initialized `Self::Elem`
    ///   values, and its storage must not overlap the sink's.
    /// * The sink must be owned, with published length `start`, capacity
    ///   `>= start + len`, and slots `[start, start + len)` uninitialized.
    // Only the cow's borrowed-source `append` uses the range form (the
    // two-state `append` is a raw move), so it is dead code in a `cow`-less
    // build.
    #[cfg_attr(not(feature = "cow"), allow(dead_code))]
    #[inline]
    unsafe fn fill_range(&mut self, src: *const Self::Elem, start: usize, len: usize)
    where
        Self: Sized,
        Self::Elem: Clone,
    {
        // Safety: forwarded from the caller. `Self: CloneSink<Elem = ...>` is
        // exactly the bound `CloneFill` requires.
        unsafe { <Self::Elem as CloneFill<Self>>::clone_fill_range(src, self, start, len) }
    }
}

/// Element-fill strategy: the `T: Copy` `memcpy` path under the nightly-only
/// `specialization` feature, the element-wise loop otherwise.
///
/// Keyed on the *element* type (not the container) so one impl pair serves
/// every call site in the crate.
pub(crate) trait CloneFill<C: CloneSink> {
    /// Fills `dest`'s slots `[0, len)` by cloning from `src`.
    ///
    /// # Safety
    ///
    /// * `src` must be valid for reads of `len` initialized `Self` values.
    /// * `dest` must be owned with published length 0, capacity `>= len`, and
    ///   storage that does not overlap `src`.
    unsafe fn clone_fill(src: *const Self, dest: &mut C, len: usize);

    /// Fills `dest`'s slots `[start, start + len)` by cloning from `src`.
    ///
    /// The range-fill primitive: the destination already holds `start`
    /// elements (its published length is `start`). For `T: Copy` it is one
    /// bulk copy plus one publication — no per-element `raw_base`
    /// re-derivation — which is what lets [`CovariantCow::append`]'s
    /// borrowed-source path take the `memcpy` route instead of extending
    /// from `iter().cloned()`. For `!Copy` it is the same element-wise loop
    /// as [`Self::clone_fill`], with the same per-element publication points.
    ///
    /// [`CovariantCow::append`]: crate::CovariantCow::append
    ///
    /// # Safety
    ///
    /// * `src` must be valid for reads of `len` initialized `Self` values.
    /// * `dest` must be owned with published length `start`, capacity
    ///   `>= start + len`, and storage that does not overlap `src`.
    /// * Slots `[start, start + len)` must be uninitialized.
    // See `CloneSink::fill_range`: only the `cow` feature reaches this.
    #[cfg_attr(not(feature = "cow"), allow(dead_code))]
    unsafe fn clone_fill_range(src: *const Self, dest: &mut C, start: usize, len: usize);
}

/// The element-wise loop: clone, write, publish, per element, starting at
/// slot `start`.
#[inline]
unsafe fn elementwise<T: Clone, C: CloneSink<Elem = T>>(
    src: *const T,
    dest: &mut C,
    start: usize,
    len: usize,
) {
    for i in 0..len {
        // Safety: `src` is valid for `len` reads (caller contract).
        let v = unsafe { (&*src.add(i)).clone() };
        // Safety: slot `start + i` is uninitialized (caller contract).
        // `raw_base` is re-called every iteration on purpose: it reborrows
        // `dest`, and the `publish_len` below invalidates a pointer derived
        // earlier (in the inline state the elements live inside `dest`
        // itself).
        unsafe { dest.raw_base().add(start + i).write(v) };
        // Safety: slots `[0, start + i + 1)` are initialized.
        unsafe { dest.publish_len(start + i + 1) };
    }
}

#[cfg(not(feature = "specialization"))]
impl<T: Clone, C: CloneSink<Elem = T>> CloneFill<C> for T {
    unsafe fn clone_fill(src: *const T, dest: &mut C, len: usize) {
        // Safety: forwarded from the caller.
        unsafe { elementwise(src, dest, 0, len) }
    }

    unsafe fn clone_fill_range(src: *const T, dest: &mut C, start: usize, len: usize) {
        // Safety: forwarded from the caller.
        unsafe { elementwise(src, dest, start, len) }
    }
}

// The specialized pair lives in a child *file* on purpose: the
// `specialization` feature gate is checked on the pre-expansion AST, so a
// `default fn` written in this file would warn (`unstable_syntax_pre_expansion`,
// which cannot be `allow`ed) even when `#[cfg]`-stripped, and hiding it behind
// a macro would just evade the lint. A `#[cfg]`-gated `mod` declaration, by
// contrast, is stripped before the file is ever read, so the feature-off build
// never contains the syntax at all. See `spec.rs`.
#[cfg(feature = "specialization")]
mod spec;

#[cfg(feature = "specialization")]
pub(crate) use spec::CloneRepeat;
