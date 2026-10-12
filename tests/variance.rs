//! Compile-time variance assertions.
//!
//! If `CovariantVec` were invariant or contravariant in `'a` or `T`, the
//! function bodies below would fail with `E0308` when the file is compiled.
//!
//! To make the assertion *self-documenting* (rather than relying only on
//! definition-time body checking), each function item is additionally coerced
//! to an `fn` pointer via a `const _` item. A bare `let _ = f::<..>;` merely
//! *names* the item and checks nothing — the same trap that `autotraits.rs`
//! warns about.

use covariant_vec::CovariantVec;

/// Covariance in the lifetime parameter `'a`.
#[allow(dead_code)]
fn assert_covariant_in_lifetime<'short, 'long: 'short>(
    v: CovariantVec<'long, &'static str, 4>,
) -> CovariantVec<'short, &'static str, 4> {
    v
}

/// Covariance in the element type parameter `T` (here carried by `&'x str`).
#[allow(dead_code)]
fn assert_covariant_in_ty<'short, 'long: 'short>(
    v: CovariantVec<'static, &'long str, 4>,
) -> CovariantVec<'static, &'short str, 4> {
    v
}

// Force the signature check by coercing the monomorphized item to an `fn`
// pointer (the coercion validates the exact signature).
const _: fn(CovariantVec<'static, &'static str, 4>) -> CovariantVec<'static, &'static str, 4> =
    assert_covariant_in_lifetime::<'static, 'static>;
const _: fn(CovariantVec<'static, &'static str, 4>) -> CovariantVec<'static, &'static str, 4> =
    assert_covariant_in_ty::<'static, 'static>;

#[test]
fn variance_assertions_type_check() {
    // Nothing to run: the `const _` items above enforce the property at compile
    // time (including plain `cargo check`).
}

/// The three-state cow is covariant in both `'a` and `T` (its `&'a [T]`
/// union field is covariant).
#[cfg(feature = "cow")]
mod cow_variance {
    use covariant_vec::CovariantCow;

    #[allow(dead_code)]
    fn assert_covariant_in_lifetime<'short, 'long: 'short>(
        v: CovariantCow<'long, &'static str, 4>,
    ) -> CovariantCow<'short, &'static str, 4> {
        v
    }

    #[allow(dead_code)]
    fn assert_covariant_in_ty<'short, 'long: 'short>(
        v: CovariantCow<'static, &'long str, 4>,
    ) -> CovariantCow<'static, &'short str, 4> {
        v
    }

    const _: fn(CovariantCow<'static, &'static str, 4>) -> CovariantCow<'static, &'static str, 4> =
        assert_covariant_in_lifetime::<'static, 'static>;
    const _: fn(CovariantCow<'static, &'static str, 4>) -> CovariantCow<'static, &'static str, 4> =
        assert_covariant_in_ty::<'static, 'static>;

    #[test]
    fn cow_variance_type_checks() {}
}
