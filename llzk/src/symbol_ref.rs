//! Types related to the SymbolRef attribute.

use std::fmt;

use melior::{
    Context, StringRef,
    ir::{Attribute, AttributeLike, attribute::FlatSymbolRefAttribute},
};
use mlir_sys::{
    MlirAttribute, mlirAttributeIsASymbolRef, mlirSymbolRefAttrGet,
    mlirSymbolRefAttrGetLeafReference, mlirSymbolRefAttrGetNestedReference,
    mlirSymbolRefAttrGetNumNestedReferences, mlirSymbolRefAttrGetRootReference,
};

/// A `SymbolRef` attribute.
///
/// The difference between this attribute and [`FlatSymbolRefAttribute`] is that this attribute
/// contains a path of symbol names, which allows referencing operations defined inside modules,
/// structs, etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolRefAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> SymbolRefAttribute<'c> {
    /// Creates a new symbol attribute with the given root and nested path symbols.
    ///
    /// For symbols defined at the _top level_ the `refs` slice should be empty.
    pub fn new(ctx: &'c Context, root: StringRef, nested: &[impl SymbolRefAttrLike<'c>]) -> Self {
        let raw_refs: Vec<_> = nested.iter().map(|r| r.to_raw()).collect();
        Self {
            inner: unsafe {
                Attribute::from_raw(mlirSymbolRefAttrGet(
                    ctx.to_raw(),
                    root.to_raw(),
                    raw_refs.len() as isize,
                    raw_refs.as_ptr(),
                ))
            },
        }
    }

    /// Creates a new symbol attribute with the given root and nested path symbols.
    ///
    /// For symbols defined at the _top level_ the `refs` slice should be empty.
    pub fn new_from_str(ctx: &'c Context, root: &str, nested: &[&str]) -> Self {
        let refs: Vec<_> = nested
            .iter()
            .map(|r| FlatSymbolRefAttribute::new(ctx, r))
            .collect();
        Self::new(ctx, StringRef::new(root), &refs)
    }

    /// Returns the root of the symbol's path.
    pub fn root(&self) -> StringRef<'c> {
        unsafe { StringRef::from_raw(mlirSymbolRefAttrGetRootReference(self.to_raw())) }
    }

    /// Returns the leaf of the symbol's path. This corresponds with the symbol name.
    pub fn leaf(&self) -> StringRef<'c> {
        unsafe { StringRef::from_raw(mlirSymbolRefAttrGetLeafReference(self.to_raw())) }
    }

    /// Returns the symbol path, excluding the root, as a vector of independent attributes.
    pub fn nested(&self) -> Vec<FlatSymbolRefAttribute<'c>> {
        let nested_count = unsafe { mlirSymbolRefAttrGetNumNestedReferences(self.to_raw()) };
        (0..nested_count)
            .map(|i| {
                unsafe {
                    // TODO: return as FlatSymREfAttribute instead of generic Attribute,
                    Attribute::from_raw(mlirSymbolRefAttrGetNestedReference(self.to_raw(), i))
                }
                .try_into()
                .expect("expected FlatSymbolRefAttribute")
            })
            .collect()
    }
}

impl<'c> AttributeLike<'c> for SymbolRefAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for SymbolRefAttribute<'c> {
    type Error = melior::Error;

    fn try_from(value: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { mlirAttributeIsASymbolRef(value.to_raw()) } {
            Ok(Self { inner: value })
        } else {
            Err(Self::Error::AttributeExpected(
                "symbol ref attr",
                value.to_string(),
            ))
        }
    }
}

impl<'c> From<SymbolRefAttribute<'c>> for Attribute<'c> {
    fn from(sym: SymbolRefAttribute<'c>) -> Attribute<'c> {
        sym.inner
    }
}

impl<'c> fmt::Display for SymbolRefAttribute<'c> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Equivalent to `SymbolRefAttr` in MLIR, providing a common trait for
/// both flat and non-flat symbol reference attributes.
pub trait SymbolRefAttrLike<'c>: AttributeLike<'c> + fmt::Display + private::Sealed {}

impl<'c> SymbolRefAttrLike<'c> for SymbolRefAttribute<'c> {}
impl<'c> SymbolRefAttrLike<'c> for FlatSymbolRefAttribute<'c> {}

/// Sealed trait pattern to prevent external implementations of `SymbolRefAttrLike`.
mod private {
    use crate::symbol_ref::SymbolRefAttribute;
    use melior::ir::attribute::FlatSymbolRefAttribute;

    pub trait Sealed {}

    // Implement for the same types as above, but no others.
    impl<'c> Sealed for SymbolRefAttribute<'c> {}
    impl<'c> Sealed for FlatSymbolRefAttribute<'c> {}
}
