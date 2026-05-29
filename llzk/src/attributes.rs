//! Utilities related to MLIR attributes.

use melior::{
    Context,
    ir::{Attribute, AttributeLike, Identifier},
};
use mlir_sys::MlirAttribute;

pub mod array;

/// An attribute associated to a name.
pub type NamedAttribute<'c> = (Identifier<'c>, Attribute<'c>);

/// Returns a null MLIR attribute handle.
pub(crate) fn null_attr() -> MlirAttribute {
    MlirAttribute {
        ptr: std::ptr::null_mut(),
    }
}

/// Rebuilds an ArrayAttribute from a generic Attribute.
pub(crate) fn rebuild_array_attr<'c>(
    context: &'c Context,
    attr: Attribute<'c>,
) -> array::ArrayAttribute<'c> {
    let elements = (0..unsafe { mlir_sys::mlirArrayAttrGetNumElements(attr.to_raw()) })
        .map(|idx| unsafe {
            Attribute::from_raw(mlir_sys::mlirArrayAttrGetElement(attr.to_raw(), idx))
        })
        .collect::<Vec<_>>();
    array::ArrayAttribute::new(context, &elements)
}
