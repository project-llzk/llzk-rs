//! Utilities related to MLIR attributes.

use melior::{
    Context, StringRef,
    ir::{Attribute, AttributeLike, Identifier},
};
use mlir_sys::{
    MlirAttribute, mlirDictionaryAttrGet, mlirDictionaryAttrGetElement,
    mlirDictionaryAttrGetElementByName, mlirDictionaryAttrGetNumElements, mlirNamedAttributeGet,
};

pub mod array;

/// An attribute associated to a name.
pub type NamedAttribute<'c> = (Identifier<'c>, Attribute<'c>);

/// Returns a null MLIR attribute handle. Used for CAPI function calls that
/// expect an optional attribute, with a null attribute used for an unspecified
/// attribute.
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

/// Creates an empty dictionary attribute in the provided context.
pub(crate) fn empty_dictionary_attr(context: &Context) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirDictionaryAttrGet(context.to_raw(), 0, std::ptr::null())) }
}

/// Converts a Rust named-attribute tuple to the raw C API representation.
pub(crate) fn tuple_to_raw_named_attr((name, attr): &NamedAttribute) -> mlir_sys::MlirNamedAttribute {
    unsafe { mlirNamedAttributeGet(name.to_raw(), attr.to_raw()) }
}

/// Converts a slice of named attributes into a dictionary attribute.
pub(crate) fn named_attributes_to_dictionary_attr<'c>(
    context: &'c Context,
    attrs: &[NamedAttribute<'c>],
) -> Attribute<'c> {
    let named_attrs: Vec<_> = attrs.iter().map(tuple_to_raw_named_attr).collect();
    unsafe {
        Attribute::from_raw(mlirDictionaryAttrGet(
            context.to_raw(),
            isize::try_from(named_attrs.len()).expect("named_attrs too large"),
            named_attrs.as_ptr(),
        ))
    }
}

/// Expands a dictionary attribute back into its `(Identifier, Attribute)` pairs.
pub(crate) fn dictionary_attr_entries<'c>(attr: Attribute<'c>) -> Vec<NamedAttribute<'c>> {
    (0..unsafe { mlirDictionaryAttrGetNumElements(attr.to_raw()) })
        .map(|idx| unsafe { mlirDictionaryAttrGetElement(attr.to_raw(), idx) })
        .map(|attr| unsafe {
            (
                Identifier::from_raw(attr.name),
                Attribute::from_raw(attr.attribute),
            )
        })
        .collect()
}

/// Returns the named attribute with `name` from `dict`, if present.
pub(crate) fn dictionary_attr_get_named<'c>(
    dict: Attribute<'c>,
    name: &str,
) -> Option<Attribute<'c>> {
    let raw = unsafe {
        mlirDictionaryAttrGetElementByName(dict.to_raw(), StringRef::new(name).to_raw())
    };
    unsafe { Attribute::from_option_raw(raw) }
}

/// Replaces or inserts one named attribute inside a dictionary attribute.
pub(crate) fn dictionary_attr_set_named<'c>(
    context: &'c Context,
    dict: Attribute<'c>,
    name: Identifier<'c>,
    attr: Attribute<'c>,
) -> Attribute<'c> {
    let mut entries = dictionary_attr_entries(dict);
    if let Some(existing) = entries.iter_mut().find(|(existing_name, _)| *existing_name == name) {
        existing.1 = attr;
    } else {
        entries.push((name, attr));
    }
    named_attributes_to_dictionary_attr(context, &entries)
}

/// Extends or rewrites the dictionary attribute element at `idx` inside an array
/// of dictionary attributes.
pub(crate) fn set_named_attr_in_dict_array<'c>(
    context: &'c Context,
    count: usize,
    current_attrs: Option<array::ArrayAttribute<'c>>,
    idx: usize,
    name: Identifier<'c>,
    attr: Attribute<'c>,
) -> array::ArrayAttribute<'c> {
    let mut dicts: Vec<_> = current_attrs
        .map(|attrs| attrs.into_iter().collect())
        .unwrap_or_else(|| vec![empty_dictionary_attr(context); count]);
    if dicts.len() < count {
        dicts.resize(count, empty_dictionary_attr(context));
    }

    dicts[idx] = dictionary_attr_set_named(context, dicts[idx], name, attr);
    array::ArrayAttribute::new(context, &dicts)
}
