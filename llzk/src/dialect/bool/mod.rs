//! `bool` dialect.

mod attrs;
mod ops;

pub use attrs::{CmpPredicate, CmpPredicateAttribute};
pub use ops::{and, assert, eq, exists, forall, ge, gt, le, lt, ne, not, or, xor, r#yield};
pub use ops::{
    is_bool_and, is_bool_assert, is_bool_cmp, is_bool_exists, is_bool_forall, is_bool_not,
    is_bool_or, is_bool_xor, is_bool_yield,
};

/// Exports the common types of the felt dialect.
pub mod prelude {
    pub use super::attrs::{CmpPredicate, CmpPredicateAttribute};
}
