//! Shared test helpers.
//!
//! This is a directory module (`tests/common/mod.rs`), not a top-level
//! `tests/*.rs` file, so Cargo does not compile it as its own test binary.
//! That matters for the allocator-based suites: a second binary would need a
//! second `#[global_allocator]` (and its own single-test discipline). See the
//! header of `alloc_count.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A drop-accounting element type: every destructor increments `counter`, and
/// `val` identifies the element in assertions.
#[derive(Clone, Debug)]
pub(crate) struct Tracked {
    counter: Arc<AtomicUsize>,
    pub(crate) val: i32,
}

impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}
impl Eq for Tracked {}

impl Tracked {
    pub(crate) fn new(counter: &Arc<AtomicUsize>, val: i32) -> Self {
        Tracked {
            counter: Arc::clone(counter),
            val,
        }
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}
