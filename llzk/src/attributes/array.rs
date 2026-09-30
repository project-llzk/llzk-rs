//! Array attributes and extensions for affine map attributes.

pub use melior::ir::attribute::ArrayAttribute;

use melior::{
    Context,
    ir::{AffineMap, Attribute, AttributeLike},
};
use mlir_sys::{MlirAttribute, mlirAffineMapAttrGet};

use crate::error::Error;

/// Represents an affine map attribute in MLIR.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AffineMapAttribute<'ctx> {
    /// Inner attribute.
    inner: Attribute<'ctx>,
}

impl<'ctx> AffineMapAttribute<'ctx> {
    /// Creates an identity map with the given number of dimensions
    /// (i.e. for 1 creates `(d0)[] -> (d0)`.)
    pub fn identity(context: &'ctx Context, dims: usize) -> Self {
        AffineMap::multi_dim_identity(context, dims).into()
    }

    /// Create an [AffineMapAttribute] from a string definition.
    pub fn parse(context: &'ctx Context, definition: &str) -> Result<Self, Error> {
        let Some(a) = Attribute::parse(context, definition) else {
            return Err(Error::GeneralError(
                "could not parse attribute from definition",
            ));
        };
        Self::try_from(a)
    }
}

impl<'ctx> From<AffineMap<'ctx>> for AffineMapAttribute<'ctx> {
    fn from(map: AffineMap<'ctx>) -> Self {
        Self {
            inner: unsafe { Attribute::from_raw(mlirAffineMapAttrGet(map.to_raw())) },
        }
    }
}

impl<'ctx> AttributeLike<'ctx> for AffineMapAttribute<'ctx> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'ctx> TryFrom<Attribute<'ctx>> for AffineMapAttribute<'ctx> {
    type Error = Error;

    fn try_from(inner: Attribute<'ctx>) -> Result<Self, Self::Error> {
        if inner.is_affine_map() {
            Ok(Self { inner })
        } else {
            Err(Error::AttributeExpected("affine_map", inner.to_string()))
        }
    }
}

impl<'ctx> From<AffineMapAttribute<'ctx>> for Attribute<'ctx> {
    fn from(value: AffineMapAttribute<'ctx>) -> Self {
        value.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn melior_affine_map_converts_to_attribute() {
        use crate::affine::AffineExprExt;
        use melior::ir::AffineExpr;

        let context = Context::new();
        let dim = AffineExpr::dim(&context, 0);
        let symbol = AffineExpr::symbol(&context, 0);
        let map = AffineMap::new(&context, 1, 1, &[dim.sub(symbol)]);
        let attr: Attribute = AffineMapAttribute::from(map).into();
        let expected = Attribute::parse(&context, "affine_map<(d0)[s0] -> (d0 - s0)>").unwrap();
        assert_eq!(attr, expected);
    }

    #[test]
    fn parse_affine_map_attribute() {
        let context = Context::new();
        let attr = AffineMapAttribute::parse(&context, "affine_map<(d0) -> (d0)>").unwrap();
        assert!(attr.is_affine_map());
    }

    #[test]
    fn parse_non_affine_map_attribute_returns_error() {
        let context = Context::new();
        let err = AffineMapAttribute::parse(&context, "unit").unwrap_err();
        assert_eq!(
            err,
            Error::AttributeExpected("affine_map", "unit".to_string())
        );
    }
}
