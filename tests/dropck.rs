//! Dropck leniency test (nightly `dropck_eyepatch` only).
//!
//! The motivating scenario: a `CovariantVec<'src, Label<'src>, N>` whose `'src`
//! (and the `Label<'src>` inside it) expires at the same scope boundary as the
//! container itself.
//!
//! Note: `'a` needs `#[may_dangle]` just as much as `T` does — the strict
//! drop-check applies to every generic parameter of a type with a `Drop` impl,
//! lifetime parameters included. Hence the crate marks **both** `'a` and `T`.

#![cfg(feature = "dropck_eyepatch")]

use covariant_vec::{CovariantVec, IntoIter};

#[derive(Clone)]
struct Label<'a>(&'a str);

/// Forces the container's `'a` and `Label`'s lifetime to be the same region.
fn push_label<'a>(v: &mut CovariantVec<'a, Label<'a>, 4>, s: &'a str) {
    v.push(Label(s));
}

#[test]
fn dropck_is_lenient_for_a_and_t() {
    // `'a` is inferred as the inner block's region; the container outlives it.
    let mut v: CovariantVec<'_, Label<'_>, 4> = CovariantVec::new();
    {
        let s = String::from("hello");
        push_label(&mut v, &s);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0, "hello"); // `s` is still alive here
    }
    // `s` is gone here. We must not touch `v` again — even `v.len()` would make
    // NLL require the borrow to be live (a region error, not a dropck one). The
    // point of this test is that `v` is *dropped* here, after `'a` has expired,
    // which is only accepted because of `#[may_dangle]` on `'a` and `T`.
}

fn build_iter<'a>(s: &'a str) -> IntoIter<'a, Label<'a>, 4> {
    let mut v: CovariantVec<'a, Label<'a>, 4> = CovariantVec::new();
    v.push(Label(s));
    v.push(Label(s));
    v.into_iter()
}

/// The partially-consumed iterator must be droppable after `'a` expires,
/// just like the container itself.
#[test]
fn into_iter_dropck_is_lenient() {
    let _it;
    {
        let s = String::from("x");
        let mut it = build_iter(&s);
        // Consume one element, leaving one for `Drop::drop_remaining` to drop.
        let first = it.next();
        assert!(first.is_some());
        _it = it;
    }
    // `_it` is dropped here, with one element already consumed, after `'a` expired.
}

#[cfg(feature = "cow")]
#[test]
fn cow_dropck_is_lenient() {
    use covariant_vec::CovariantCow;

    fn build<'a>(s: &'a str) -> CovariantCow<'a, Label<'a>, 4> {
        let mut c = CovariantCow::new();
        c.push(Label(s));
        c
    }

    let _c;
    {
        let s = String::from("x");
        _c = build(&s);
    }
    // `_c` is dropped here, after `'a` expired.
}
