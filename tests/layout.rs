//! Compile-time layout assertions.
//!
//! These numbers assume `W = size_of::<usize>() = 8`, so the whole file is
//! gated to 64-bit targets.

#![cfg(target_pointer_width = "64")]

use covariant_vec::CovariantVec;
use std::mem::{align_of, size_of};

// --- T = u64 (S = 8, A = 8), N = 4: N*S = 32 >= 2W = 16 ---
// union = 32, size = round_up(32 + 8, 8) = 40 = N*S + W
const _: () = assert!(size_of::<CovariantVec<'static, u64, 4>>() == 40);
const _: () = assert!(align_of::<CovariantVec<'static, u64, 4>>() == 8);

// --- T = u8 (S = 1, A = 1), N = 4: N*S = 4 < 2W = 16 ---
// union = 16, size = round_up(16 + 8, 8) = 24 = 3W
const _: () = assert!(size_of::<CovariantVec<'static, u8, 4>>() == 24);
const _: () = assert!(align_of::<CovariantVec<'static, u8, 4>>() == 8);

// --- N = 0: the heap variant dominates ---
const _: () = assert!(size_of::<CovariantVec<'static, u64, 0>>() == 24);

// --- ZST element: same 3W layout, heap variant never used ---
const _: () = assert!(size_of::<CovariantVec<'static, (), 4>>() == 24);

// --- Alignment follows max(A_T, W) ---
const _: () = assert!(align_of::<CovariantVec<'static, u64, 4>>() == 8);
// A 16-byte-aligned element raises the struct alignment.
#[repr(align(16))]
struct Align16(#[allow(dead_code)] u64);
const _: () = assert!(align_of::<CovariantVec<'static, Align16, 2>>() == 16);
// N*S = 32, union = 32, + W = 40, round up to 16 => 48
const _: () = assert!(size_of::<CovariantVec<'static, Align16, 2>>() == 48);

// --- No niche: `Option` costs a full word (known limitation) ---
//
// The two-state design leaves bit 62 of the `len` word unused, but it cannot be
// turned into an `Option` niche: the tag lives inside a plain `usize`, whose
// every bit pattern is a valid `usize`, and `rustc` has no way to see the
// semantic constraint "bit 63 set ⇒ heap, bit 62 never set". Niche optimization
// requires a genuinely invalid *value* of a field's type (as `NonNull` or `&T`
// provide), which a raw `usize` never has. So `Option<CovariantVec>` pays the
// full extra discriminant word.
const _: () = assert!(size_of::<Option<CovariantVec<'static, u64, 4>>>() == 48);

// --- The cow has the same union + len layout (no extra marker field) ---
#[cfg(feature = "cow")]
mod cow_layout {
    use covariant_vec::CovariantCow;
    use std::mem::{align_of, size_of};

    const _: () = assert!(size_of::<CovariantCow<'static, u64, 4>>() == 40);
    const _: () = assert!(size_of::<CovariantCow<'static, u8, 4>>() == 24);
    const _: () = assert!(size_of::<CovariantCow<'static, u64, 0>>() == 24);
    const _: () = assert!(align_of::<CovariantCow<'static, u64, 4>>() == 8);
}

#[test]
fn layout_is_as_documented() {
    // The `const _` items above already enforce this at compile time.
}
