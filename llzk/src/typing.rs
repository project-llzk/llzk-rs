//! Utilities related to type unification.

use std::ptr::null;

use melior::{
    StringRef,
    ir::{TypeLike, r#type::Type},
};
use mlir_sys::MlirStringRef;

/// Return `true` iff the two types are equivalent or could be equivalent after full
/// instantiation of struct parameters.
///
/// `rhs_reverse_prefix` describes the symbol path prefix to prepend to reverse references found on
/// the right-hand side type before attempting unification.
pub fn types_unify<'c>(
    lhs: impl TypeLike<'c>,
    rhs: impl TypeLike<'c>,
    rhs_reverse_prefix: &[StringRef<'_>],
) -> bool {
    let rhs_reverse_prefix: Vec<MlirStringRef> = rhs_reverse_prefix
        .iter()
        .map(|name| name.to_raw())
        .collect();
    let rhs_reverse_prefix_ptr = if rhs_reverse_prefix.is_empty() {
        null()
    } else {
        rhs_reverse_prefix.as_ptr()
    };

    unsafe {
        llzk_sys::llzkTypesUnify(
            lhs.to_raw(),
            rhs.to_raw(),
            rhs_reverse_prefix.len() as isize,
            rhs_reverse_prefix_ptr,
        )
    }
}

/// Return `true` iff the two types are equivalent without any symbol prefix adjustment.
#[inline]
pub fn types_equal_or_unifiable<'c>(lhs: impl TypeLike<'c>, rhs: impl TypeLike<'c>) -> bool {
    types_unify(lhs, rhs, &[])
}

/// Return `true` iff the given [Type] is equivalent to the other type without any symbol prefix
/// adjustment.
#[inline]
pub fn is_unifiable_with(lhs: Type<'_>, rhs: Type<'_>) -> bool {
    types_equal_or_unifiable(lhs, rhs)
}
