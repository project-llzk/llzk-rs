//! `llzk` dialect.

mod attrs;
mod ops;

pub use attrs::LoopBoundsAttribute;
pub use attrs::PublicAttribute;

pub use ops::{is_nondet, nondet};

/// Exports the common types of the llzk dialect.
pub mod prelude {
    pub use super::attrs::LoopBoundsAttribute;
    pub use super::attrs::PublicAttribute;
}
