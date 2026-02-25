use llzk_sys::{
    LlzkBoolFeltCmpPredicate, llzkAttributeIsA_Bool_FeltCmpPredicateAttr,
    llzkBool_FeltCmpPredicateAttrGet,
};
use melior::{
    Context,
    ir::{Attribute, AttributeLike},
};
use mlir_sys::MlirAttribute;

/// Possible options for creating [`CmpPredicateAttribute`].
#[derive(Debug)]
#[repr(u32)]
pub enum CmpPredicate {
    /// Equal to.
    Eq = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_EQ,
    /// Not equal to.
    Ne = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_NE,
    /// Less than.
    Lt = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_LT,
    /// Less than or equal to.
    Le = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_LE,
    /// Greater than.
    Gt = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_GT,
    /// Greater than or equal to.
    Ge = llzk_sys::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_GE,
}

/// Attribute representing a comparison predicate.
#[derive(Clone, Copy, Debug)]
pub struct CmpPredicateAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> CmpPredicateAttribute<'c> {
    /// Creates a new attribute from its raw representation.
    ///
    /// # Safety
    ///
    /// The MLIR attribute must contain a valid pointer of type `CmpPredicateAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates a new attribute.
    pub fn new(ctx: &'c Context, predicate: CmpPredicate) -> Self {
        unsafe {
            Self::from_raw(llzkBool_FeltCmpPredicateAttrGet(
                ctx.to_raw(),
                predicate as LlzkBoolFeltCmpPredicate,
            ))
        }
    }
}

impl<'c> AttributeLike<'c> for CmpPredicateAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for CmpPredicateAttribute<'c> {
    type Error = melior::Error;

    fn try_from(t: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Bool_FeltCmpPredicateAttr(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected(
                "llzk cmp attr",
                t.to_string(),
            ))
        }
    }
}

impl<'c> std::fmt::Display for CmpPredicateAttribute<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<'c> From<CmpPredicateAttribute<'c>> for Attribute<'c> {
    fn from(attr: CmpPredicateAttribute<'c>) -> Attribute<'c> {
        attr.inner
    }
}
