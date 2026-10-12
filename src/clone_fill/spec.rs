//! The `T: Copy` specialization pair — the only `default fn` in the crate.
//!
//! This file is compiled **only** when the nightly `specialization` cargo
//! feature is on (see the `#[cfg]`-gated `mod spec;` in `clone_fill.rs`).
//! That split is not cosmetic: rustc checks the `specialization` feature gate
//! on the *pre-expansion* AST, so a `default fn` written in `clone_fill.rs`
//! would trip the `unstable_syntax_pre_expansion` lint on stable builds even
//! though the `#[cfg]` around it is stripped moments later, and that lint
//! cannot be `allow`ed. Gating the *file* keeps the syntax out of the
//! feature-off build entirely, with no lint suppression.
//!
//! # Why this specialization is sound
//!
//! `specialization` is an incomplete feature whose known unsoundness concerns
//! specializing on *lifetimes* and on associated items. The pattern below is
//! the narrow one `std` itself uses: two impls whose self type is the *same*
//! type parameter `T`, differing only by a strictly stronger bound (`T: Copy`
//! implies `T: Clone`, its supertrait), specializing a *method*. The trait
//! [`CloneFill`] has no lifetime parameters, no associated types and no
//! associated consts, so none of the unsound corners are reachable: the only
//! thing the specialization can change is which body runs, and for `T: Copy`
//! the two bodies are observationally identical by the `Copy` contract.
//!
//! (`min_specialization`, the sound subset, is not an option: it rejects
//! trait-bound specialization outright with "cannot specialize on trait
//! `Copy`". Keying on `T: Copy` — rather than on `!needs_drop::<T>()`, which
//! is a strict superset — is what keeps the change observably behaviour-free.)

use core::ptr;

use super::{elementwise, CloneFill, CloneSink};

/// The `memcpy` path: one bulk copy, then a single length publication.
///
/// Neither `ptr::copy_nonoverlapping` nor `publish_len` can panic, so there is
/// no partially-filled window to recover from, and no need for the
/// per-element publication the element-wise loop performs.
#[inline]
unsafe fn memcpy_fill<T: Copy, C: CloneSink<Elem = T>>(src: *const T, dest: &mut C, len: usize) {
    // Safety: `dest` is owned (caller contract).
    let dst = unsafe { dest.raw_base() };
    // Safety: `src` is valid for `len` reads and `dst` for `len` writes, and
    // the two do not overlap (caller contract). `raw_base` is derived exactly
    // once: nothing touches `dest` between here and the single `publish_len`,
    // so there is nothing to re-derive under Stacked Borrows.
    unsafe { ptr::copy_nonoverlapping(src, dst, len) };
    // Safety: slots `[0, len)` are initialized.
    unsafe { dest.publish_len(len) };
}

impl<T: Clone, C: CloneSink<Elem = T>> CloneFill<C> for T {
    default unsafe fn clone_fill(src: *const T, dest: &mut C, len: usize) {
        // Safety: forwarded from the caller.
        unsafe { elementwise(src, dest, len) }
    }
}

impl<T: Copy, C: CloneSink<Elem = T>> CloneFill<C> for T {
    unsafe fn clone_fill(src: *const T, dest: &mut C, len: usize) {
        // Safety: forwarded from the caller.
        unsafe { memcpy_fill(src, dest, len) }
    }
}

/// Repeat-fill strategy for fills of **one repeated value** (`from_elem`,
/// `resize`-grow appends), as opposed to [`CloneFill`], which copies each
/// element from a distinct source slot.
///
/// Two shapes, because the two call sites pin different (equally
/// std-parity) accounting contracts:
///
/// * [`CloneRepeat::seed_fill`] — move the value into slot 0, clone *that*
///   slot for the rest: `from_elem`/`vec![elem; n]` semantics (exactly `n`
///   values, `n - 1` clones, `n` destructors).
/// * [`CloneRepeat::tail_fill`] — clone the value into the first
///   `len - 1` new slots and move it into the last: `Vec::resize` semantics
///   (also `len - 1` clones, no extra destructor).
///
/// Under `T: Copy` both collapse to one length publication plus `bulk`'s
/// O(log len) doubling `copy_nonoverlapping` calls (≈2× write amplification —
/// the only true single-memcpy path is `CloneFill`'s `memcpy_fill`) — sound
/// because no user code runs between the copies and the publication (nothing
/// can observe a partially-filled buffer or panic), and for a type with no
/// destructor "moved original" vs "last copy" is unobservable.
pub(crate) trait CloneRepeat<C: CloneSink> {
    /// Fills `dest[0, len)` with `len` copies of `elem`, moving `elem` into
    /// slot 0 (`from_elem` semantics).
    ///
    /// # Safety
    ///
    /// `dest` must be owned with published length 0, capacity `>= len`, and
    /// slots `[0, len)` uninitialized.
    unsafe fn seed_fill(elem: Self, dest: &mut C, len: usize);

    /// Fills `dest[start, start + len)` with `len` copies of `elem`, moving
    /// `elem` into slot `start + len - 1` (`Vec::resize` semantics).
    ///
    /// # Safety
    ///
    /// `dest` must be owned with published length `start`, capacity
    /// `>= start + len`, and slots `[start, start + len)` uninitialized.
    unsafe fn tail_fill(elem: Self, dest: &mut C, start: usize, len: usize);
}

/// The bulk path for `T: Copy`: seed the first slot of the run with the single
/// owned value, then double the initialized run (`[0, 1)` → `[1, 2)`,
/// `[0, 2)` → `[2, 4)`, …) until the whole run is filled, and publish the
/// length once. That is O(log len) `copy_nonoverlapping` calls with ≈2× write
/// amplification, but a single publication: nothing between the copies and the
/// publication can panic or run user code, so there is no partially-filled
/// window to recover from.
#[inline]
unsafe fn bulk<T: Copy, C: CloneSink<Elem = T>>(elem: &T, dest: &mut C, start: usize, len: usize) {
    if len == 0 {
        return;
    }
    // Safety: `dest` is owned with capacity `>= start + len` (caller contract).
    // `raw_base` is derived exactly once: nothing touches `dest` between here
    // and the single `publish_len`, so there is nothing to re-derive under
    // Stacked Borrows.
    let base = unsafe { dest.raw_base().add(start) };
    // Seed the run's first slot from the caller-owned value (not from `dest`,
    // so the two never overlap).
    // Safety: slot `start` is uninitialized (caller contract); `elem` is valid
    // for one read and does not point into `dest`.
    unsafe { ptr::write(base, *elem) };
    // Double: copy the initialized prefix `[start, start + filled)` into
    // `[start + filled, start + 2 * filled)`, clamped to the run end. Each copy
    // reads only already-initialized slots, and source and destination are
    // either disjoint or exactly abutting, never overlapping.
    let mut filled = 1;
    while filled < len {
        let n = filled.min(len - filled);
        // Safety: `base[0, filled)` is initialized and `base + filled` has room
        // for `n <= filled` more slots (uninitialized part of the run).
        unsafe { ptr::copy_nonoverlapping(base, base.add(filled), n) };
        filled += n;
    }
    // Safety: slots `[start, start + len)` are initialized.
    unsafe { dest.publish_len(start + len) };
}

impl<T: Clone, C: CloneSink<Elem = T>> CloneRepeat<C> for T {
    default unsafe fn seed_fill(elem: T, dest: &mut C, len: usize) {
        if len == 0 {
            return;
        }
        // Move into slot 0.
        // Safety: slot 0 is uninitialized (caller contract).
        unsafe { dest.raw_base().write(elem) };
        // Safety: slot 0 is initialized.
        unsafe { dest.publish_len(1) };
        for i in 1..len {
            // Clone the seed slot. The base is re-derived each iteration on
            // purpose: the publication below is a use of `dest` that
            // invalidates a hoisted pointer in the inline state (Stacked
            // Borrows, same discipline as `elementwise`).
            let v = {
                // Safety: slot 0 is initialized.
                let seed = unsafe { dest.raw_base() };
                unsafe { (&*seed).clone() }
            };
            // Safety: slot `i` is uninitialized (caller contract).
            unsafe { dest.raw_base().add(i).write(v) };
            // Safety: slots `[0, i + 1)` are initialized.
            unsafe { dest.publish_len(i + 1) };
        }
    }

    default unsafe fn tail_fill(elem: T, dest: &mut C, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let last = start + len - 1;
        for i in start..last {
            let v = elem.clone();
            // Safety: slot `i` is uninitialized (caller contract). The base is
            // re-derived each iteration (Stacked Borrows, as above).
            unsafe { dest.raw_base().add(i).write(v) };
            // Safety: slots `[start, i + 1)` are initialized.
            unsafe { dest.publish_len(i + 1) };
        }
        // Move the original into the last new slot: no extra destructor.
        // Safety: slot `last` is uninitialized (caller contract).
        unsafe { dest.raw_base().add(last).write(elem) };
        // Safety: slots `[start, last + 1)` are initialized.
        unsafe { dest.publish_len(last + 1) };
    }
}

impl<T: Copy, C: CloneSink<Elem = T>> CloneRepeat<C> for T {
    unsafe fn seed_fill(elem: T, dest: &mut C, len: usize) {
        if len == 0 {
            return;
        }
        // Safety: forwarded from the caller.
        unsafe { bulk(&elem, dest, 0, len) };
    }

    unsafe fn tail_fill(elem: T, dest: &mut C, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        // Safety: forwarded from the caller.
        unsafe { bulk(&elem, dest, start, len) };
    }
}
