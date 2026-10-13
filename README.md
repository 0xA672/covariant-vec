# covariant-vec

A `no_std`, `alloc`-backed small-vector whose inline/heap state is encoded in the **highest bit of the length field**, plus an optional three-state copy-on-write variant.

```rust
use covariant_vec::CovariantVec;

let mut v = CovariantVec::<i32, 4>::new();
v.extend(1..=4);
assert!(!v.is_heap());     // still inline

v.push(5);                 // spills to the heap on the 5th element
assert!(v.is_heap());
assert_eq!(&*v, &[1, 2, 3, 4, 5]);
```

## Highlights

- **Length-encoded state tag.** No data-embedded sentinel, no extra discriminant word. `len`'s top bit is `HEAP_TAG`; `CovariantVec<u64, 4>` is exactly 40 bytes (`N·S + W`).
- **Covariant in both `'a` and `T`.** The `'a` parameter is a variance anchor (`PhantomData<&'a ()>`), so `CovariantVec<'src, Label<'src>, N>` is the intended shape; variance is pinned at compile time in `tests/variance.rs`.
- **Lenient dropck (nightly).** With the `dropck_eyepatch` feature, `#[may_dangle]` is applied to **both** `'a` and `T`, so a container can be dropped after its lifetime has expired.
- **Fallible allocation everywhere.** `try_reserve` / `try_reserve_exact` return `TryReserveError` on capacity overflow or `alloc` failure, without unstable APIs.
- **Optional copy-on-write (`cow`).** `CovariantCow<'a, T, N>` adds a zero-copy `Borrowed` state by spending bit 62 of the length word, with owned-only `_owned` methods so `!Clone` types remain usable when the cow is known to be owned.
- **Nightly `T: Copy` fast paths (`specialization`).** `Clone`, `to_mut`, and `try_materialize` switch to a single `ptr::copy_nonoverlapping` for `T: Copy` — same code path, same panic safety, same observable behavior. Stable builds always take the element-wise loop.

## Feature flags

| Feature | Nightly? | Effect |
|---|---|---|
| *(default)* | — | Inline + heap two-state storage, element-wise clone loops. |
| `cow` | No | Enables the three-state `CovariantCow`. |
| `dropck_eyepatch` | Yes | `#[may_dangle]` on `Drop` impls (RFC 1238). |
| `specialization` | Yes | `T: Copy` bulk-copy fast paths. Minimum known-good nightly: `nightly-2026-01-16`. |

## MSRV

Rust **1.80**. The floor is set by bare `size_of`/`align_of` in the prelude (1.80) and inline `const {}` blocks (1.79). The `core::error::Error` impl for `TryReserveError` is gated on a `build.rs`-emitted cfg for rustc ≥ 1.81; older compilers simply don't get the impl.

## The ZST ownership protocol

For a zero-sized `T`, no bytes are ever stored; the **count is the ownership**. Three conventions must hold together:

1. `push`/`push_owned` `mem::forget` the value and increment the length — the count now owns its one conceptual destructor.
2. `drop_contents`/`truncate`/`retain_mut`/`Drain::drop` run `drop_in_place` over `raw_len` *unwritten* slots — one destructor per counted element.
3. `pop`/`IntoIter::next`/`Drain::next` `ptr::read` a value out of such an unwritten slot and decrement the count — the element is not dropped twice.

Every site is annotated in the source; a change that breaks the balance is a change that either double-counts or double-drops.

## Deliberately not provided

- `append` / `split_off` / `splice` / `extract_if` — omitted to keep the unsafe surface small. `sort*`/`dedup*` are reachable through `DerefMut`, and a `Vec` round-trip covers the rest.
- A stable `T: Copy` bulk-copy fast path — specialization is the only mechanism that keys on `T: Copy` exactly, and it is nightly-only.
- An `Option` niche — the tag lives inside a plain `usize`, whose every bit pattern is valid; rustc cannot see "bit 63 set ⇒ heap". `Option<CovariantVec>` pays the full discriminant word.

## Copy-on-write

With the `cow` feature, `CovariantCow<'a, T, N>` adds a zero-copy borrowed state:

```rust
use covariant_vec::CovariantCow;

let data = [1, 2, 3, 4, 5];
let mut c = CovariantCow::<i32, 4>::from_borrowed(&data);
assert!(c.is_borrowed());          // zero-copy
c.push(6);                         // copy-on-write: materializes, then appends
assert_eq!(&*c, &[1, 2, 3, 4, 5, 6]);
assert_eq!(data, [1, 2, 3, 4, 5]); // borrower untouched
```

Borrowed-aware methods carry a `T: Clone` bound (materialization requires it); each has an `_owned` twin that instead panics on a borrowed cow, so an owned `!Clone` cow can still mutate in every way that does not clone.

## Testing

```sh
cargo test --all-features     # nightly: dropck_eyepatch + specialization + cow
cargo test --features cow     # stable: cow only
cargo test --no-default-features
cargo miri test --all-features
```

CI additionally runs Miri under both Stacked Borrows and Tree Borrows, AddressSanitizer + LeakSanitizer, clippy, and rustdoc.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
