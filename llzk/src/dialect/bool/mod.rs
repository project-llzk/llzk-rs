//! `bool` dialect.

mod attrs;
mod ops;

pub use attrs::{CmpPredicate, CmpPredicateAttribute};
use melior::ir::{Type, TypeLike};
pub use ops::{and, assert, eq, exists, forall, ge, gt, le, lt, ne, not, or, xor, r#yield};
pub use ops::{
    is_bool_and, is_bool_assert, is_bool_cmp, is_bool_exists, is_bool_forall, is_bool_not,
    is_bool_or, is_bool_xor, is_bool_yield,
};

use crate::error::Error;

/// Exports the common types of the felt dialect.
pub mod prelude {
    pub use super::attrs::{CmpPredicate, CmpPredicateAttribute};
}

/// Returns the type used in the body of a quantifier op (`bool.forall` and `bool.exists`)
/// based on the given type.
///
/// The type must be a valid type for the domain of those operations.
pub fn quantifier_iter_type(r#type: Type) -> Result<Type, Error> {
    unsafe {
        Type::from_option_raw(llzk_sys::llzkBool_QuantifierOpGetDomainIterType(
            r#type.to_raw(),
        ))
    }
    .ok_or_else(|| Error::GeneralError("expected valid quantifier sort type"))
}
