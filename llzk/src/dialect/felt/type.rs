//! Implementation of `!felt.type` type.

use crate::utils::IsA;
use llzk_sys::{llzkFelt_FeltTypeGet, llzkFelt_FeltTypeGetUnspecified, llzkTypeIsA_Felt_FeltType};
use melior::{
    Context,
    ir::{Identifier, Type, TypeLike},
};
use mlir_sys::MlirType;

/// Represents the `!felt.type` type.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct FeltType<'c> {
    r#type: Type<'c>,
}

impl<'c> FeltType<'c> {
    pub(super) unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(raw) },
        }
    }

    /// Creates a new felt type.
    pub fn new(ctx: &'c Context) -> Self {
        unsafe { Self::from_raw(llzkFelt_FeltTypeGetUnspecified(ctx.to_raw())) }
    }

    /// Creates a new felt type with the given field name.
    ///
    /// # Safety
    ///
    /// The process will abort at the C++ level if the given field name is not valid.
    pub fn with_field(ctx: &'c Context, name: &str) -> Self {
        let ident = Identifier::new(ctx, name);
        unsafe { Self::from_raw(llzkFelt_FeltTypeGet(ctx.to_raw(), ident.to_raw())) }
    }
}

impl<'c> TypeLike<'c> for FeltType<'c> {
    fn to_raw(&self) -> MlirType {
        self.r#type.to_raw()
    }
}

impl<'c> TryFrom<Type<'c>> for FeltType<'c> {
    type Error = melior::Error;

    fn try_from(t: Type<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkTypeIsA_Felt_FeltType(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::TypeExpected("llzk felt", t.to_string()))
        }
    }
}

impl<'c> std::fmt::Display for FeltType<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.r#type, formatter)
    }
}

impl<'c> From<FeltType<'c>> for Type<'c> {
    fn from(t: FeltType<'c>) -> Type<'c> {
        t.r#type
    }
}

/// Return `true` iff the given [Type] is a [FeltType].
#[inline]
pub fn is_felt_type(t: Type) -> bool {
    t.isa::<FeltType>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::LlzkContext;
    use melior::ir::{Location, operation::OperationLike};
    use rstest::rstest;

    #[rstest]
    #[case("mersenne31")]
    fn test_ctor_with_field(#[case] field: &str) {
        let ctx = LlzkContext::new();
        let t = FeltType::with_field(&ctx, field);
        // Test by printing.
        assert_eq!(t.to_string(), format!("!felt.type<\"{field}\">"));
        // And by using some op that we validate.
        let op = crate::dialect::llzk::nondet(Location::unknown(&ctx), t.into());
        assert!(op.verify());
    }
}
