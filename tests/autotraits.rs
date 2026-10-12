//! Auto-trait assertions.
//!
//! A `fn` item only enforces its generic bounds when it is **called** (or
//! coerced to a fn pointer), so each assertion below is written as a real call
//! rather than a bare `let _ = f::<T>;` (which checks nothing). The calls are
//! no-ops at runtime; the value is in the compile-time bound check.

use std::cell::Cell;

use covariant_vec::CovariantVec;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn send_sync_match_vec_semantics() {
    // Plain data: `Send` and `Sync`.
    assert_send::<CovariantVec<'static, i32, 4>>();
    assert_sync::<CovariantVec<'static, i32, 4>>();

    // `Cell<i32>: Send + !Sync`, so the container stays `Send`. The `!Sync`
    // half is a *negative* property and cannot be asserted positively here.
    assert_send::<CovariantVec<'static, Cell<i32>, 4>>();
}

#[cfg(feature = "cow")]
#[test]
fn cow_send_sync() {
    use covariant_vec::CovariantCow;
    assert_send::<CovariantCow<'static, i32, 4>>();
    assert_sync::<CovariantCow<'static, i32, 4>>();
}

#[cfg(feature = "cow")]
#[test]
fn cow_into_iter_is_send_for_non_sync_elem() {
    // Review #3: `CowIntoIter` always holds an OWNED cow, so the borrowed
    // state's `T: Sync` requirement does not apply. `Cell<i32>` is
    // `Send + !Sync`, so this iteration can cross a thread boundary even
    // though `CovariantCow<'_, Cell<i32>, _>` itself cannot.
    use std::thread;

    use covariant_vec::CovariantCow;

    // A scoped thread, not `Box::leak`: the borrow only has to outlive the
    // thread, and an intentional leak would trip Miri's leak checker (the
    // same reason the `mem::forget` drain tests are `cfg(not(miri))`).
    let data = vec![Cell::new(1), Cell::new(2), Cell::new(3)].into_boxed_slice();
    let mut c = CovariantCow::<Cell<i32>, 4>::from_borrowed(&data);
    c.to_mut()[0].set(10); // materialize: owned from here on
    let it = c.into_iter();
    thread::scope(|s| {
        let handle = s.spawn(move || it.map(|x| x.get()).sum::<i32>());
        assert_eq!(handle.join().unwrap(), 10 + 2 + 3);
    });
}
