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
//!   [`CloneSink`] deliberately exposes *exactly* those two reborrowing
//!   operations and nothing else, so the discipline is expressed by
//!   construction: there is no way to hold a slot pointer across a publish.
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

/// Destination for [`CloneFill::clone_fill`]: the two container operations the
/// clone loops need.
///
/// Both operations reborrow the container, which is what makes the inline
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
}

/// The element-wise loop: clone, write, publish, per element.
#[inline]
unsafe fn elementwise<T: Clone, C: CloneSink<Elem = T>>(src: *const T, dest: &mut C, len: usize) {
    for i in 0..len {
        // Safety: `src` is valid for `len` reads (caller contract).
        let v = unsafe { (&*src.add(i)).clone() };
        // Safety: slot `i` is uninitialized (caller contract). `raw_base` is
        // re-called every iteration on purpose: it reborrows `dest`, and the
        // `publish_len` below invalidates a pointer derived earlier (in the
        // inline state the elements live inside `dest` itself).
        unsafe { dest.raw_base().add(i).write(v) };
        // Safety: slots `[0, i + 1)` are initialized.
        unsafe { dest.publish_len(i + 1) };
    }
}

#[cfg(not(feature = "specialization"))]
impl<T: Clone, C: CloneSink<Elem = T>> CloneFill<C> for T {
    unsafe fn clone_fill(src: *const T, dest: &mut C, len: usize) {
        // Safety: forwarded from the caller.
        unsafe { elementwise(src, dest, len) }
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
