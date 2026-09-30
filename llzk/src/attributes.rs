//! Utilities related to MLIR attributes.

use melior::{
    Context,
    ir::{
        Attribute, AttributeLike, Identifier,
        attribute::{ArrayAttribute, DictionaryAttribute},
    },
};
use mlir_sys::{MlirAttribute, mlirNamedAttributeGet};

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

/// Converts a Rust named-attribute tuple to the raw C API representation.
pub(crate) fn tuple_to_raw_named_attr(
    (name, attr): &NamedAttribute,
) -> mlir_sys::MlirNamedAttribute {
    unsafe { mlirNamedAttributeGet(name.to_raw(), attr.to_raw()) }
}

/// Replaces or inserts one named attribute inside a dictionary attribute.
pub fn dictionary_attr_set_named<'c>(
    context: &'c Context,
    dict: Attribute<'c>,
    name: Identifier<'c>,
    attr: Attribute<'c>,
) -> Attribute<'c> {
    let dict = DictionaryAttribute::try_from(dict).expect("expected a dictionary attribute");
    let mut entries: Vec<_> = (0..dict.len())
        .map(|idx| dict.element(idx).unwrap())
        .collect();
    if let Some(existing) = entries
        .iter_mut()
        .find(|(existing_name, _)| *existing_name == name)
    {
        existing.1 = attr;
    } else {
        entries.push((name, attr));
    }
    DictionaryAttribute::new(context, &entries).into()
}

/// Extends or rewrites the dictionary attribute element at `idx` inside an array
/// of dictionary attributes.
pub fn set_named_attr_in_dict_array<'c>(
    context: &'c Context,
    count: usize,
    current_attrs: Option<ArrayAttribute<'c>>,
    idx: usize,
    name: Identifier<'c>,
    attr: Attribute<'c>,
) -> ArrayAttribute<'c> {
    let mut dicts: Vec<_> = current_attrs
        .map(|attrs| {
            (0..attrs.len())
                .map(|idx| attrs.element(idx).unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![DictionaryAttribute::new(context, &[]).into(); count]);
    if dicts.len() < count {
        dicts.resize(count, DictionaryAttribute::new(context, &[]).into());
    }

    dicts[idx] = dictionary_attr_set_named(context, dicts[idx], name, attr);
    ArrayAttribute::new(context, &dicts)
}
