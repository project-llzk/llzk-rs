//! High-level representation of an array attribute with any kind of attributes.
//!
//! [Melior's version](melior::ir::attribute::array::ArrayAttribute) only wraps
//! dense arrays of i64. The version in this file wraps a type erased attribute.

use std::ops::Index;

use melior::{
    Context,
    ir::{Attribute, AttributeLike},
};
use mlir_sys::MlirAttribute;

use crate::error::Error;

/// An attribute that contains an array of other attributes. These attributes can be on any type.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ArrayAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> ArrayAttribute<'c> {
    /// Creates a new array attribute.
    pub fn new(context: &'c Context, attrs: &[Attribute<'c>]) -> Self {
        let raw_attrs: Vec<_> = attrs.iter().map(|a| a.to_raw()).collect();
        Self::try_from(unsafe {
            Attribute::from_raw(mlir_sys::mlirArrayAttrGet(
                context.to_raw(),
                attrs.len() as isize,
                raw_attrs.as_ptr(),
            ))
        })
        .expect("newly created atribute must be an array attribute")
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        unsafe { mlir_sys::mlirArrayAttrGetNumElements(self.to_raw()) as usize }
    }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the idx-th element of the array.
    ///
    /// Returns None if the index is out of bounds.
    pub fn get(&self, idx: usize) -> Option<Attribute<'c>> {
        (idx < self.len()).then(|| unsafe {
            Attribute::from_raw(mlir_sys::mlirArrayAttrGetElement(
                self.to_raw(),
                idx as isize,
            ))
        })
    }
}

impl std::fmt::Debug for ArrayAttribute<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl std::fmt::Display for ArrayAttribute<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl<'c> PartialEq<Attribute<'c>> for ArrayAttribute<'c> {
    fn eq(&self, other: &Attribute<'c>) -> bool {
        self.inner == *other
    }
}

impl<'c> PartialEq<ArrayAttribute<'c>> for Attribute<'c> {
    fn eq(&self, other: &ArrayAttribute<'c>) -> bool {
        *self == other.inner
    }
}

impl<'c> AttributeLike<'c> for ArrayAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for ArrayAttribute<'c> {
    type Error = Error;

    fn try_from(inner: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { mlir_sys::mlirAttributeIsAArray(inner.to_raw()) } {
            Ok(ArrayAttribute { inner })
        } else {
            Err(Error::AttributeExpected("array", format!("{inner}")))
        }
    }
}

impl<'c> From<ArrayAttribute<'c>> for Attribute<'c> {
    fn from(value: ArrayAttribute<'c>) -> Self {
        value.inner
    }
}

impl<'c> IntoIterator for ArrayAttribute<'c> {
    type Item = Attribute<'c>;

    type IntoIter = ArrayAttributeIter<'c>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayAttributeIter {
            array: self,
            current: 0,
        }
    }
}

impl<'c> IntoIterator for &ArrayAttribute<'c> {
    type Item = Attribute<'c>;

    type IntoIter = ArrayAttributeIter<'c>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayAttributeIter {
            array: *self,
            current: 0,
        }
    }
}

/// Iterator of an [`ArrayAttribute`].
#[derive(Debug)]
pub struct ArrayAttributeIter<'c> {
    array: ArrayAttribute<'c>,
    current: usize,
}

impl<'c> Iterator for ArrayAttributeIter<'c> {
    type Item = Attribute<'c>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current;
        self.current += 1;
        self.array.get(idx)
    }
}
