use crate::attributes::NamedAttribute;
use llzk_sys::{
    llzkAttributeIsA_Llzk_LoopBoundsAttr, llzkAttributeIsA_Llzk_PublicAttr,
    llzkLlzk_LoopBoundsAttrGet, llzkLlzk_PublicAttrGet,
};
use melior::{
    Context, StringRef,
    ir::{Attribute, AttributeLike, Identifier},
};
use mlir_sys::MlirAttribute;

/// Represents the `llzk.pub` attribute.
#[derive(Clone, Copy, Debug)]
pub struct PublicAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> PublicAttribute<'c> {
    /// Creates a new attribute from its raw representation.
    ///
    /// # Safety
    ///
    /// The MLIR attribute must contain a valid pointer of type `PublicAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates a new attribute.
    pub fn new(ctx: &'c Context) -> Self {
        unsafe { Self::from_raw(llzkLlzk_PublicAttrGet(ctx.to_raw())) }
    }

    /// Creates a new `llzk.pub` NamedAttribute.
    pub fn new_named_attr(ctx: &'c Context) -> NamedAttribute<'c> {
        (Identifier::new(ctx, "llzk.pub"), Attribute::unit(ctx))
    }
}

impl<'c> AttributeLike<'c> for PublicAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for PublicAttribute<'c> {
    type Error = melior::Error;

    fn try_from(t: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Llzk_PublicAttr(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected("llzk pub", t.to_string()))
        }
    }
}

impl<'c> std::fmt::Display for PublicAttribute<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<'c> From<PublicAttribute<'c>> for Attribute<'c> {
    fn from(attr: PublicAttribute<'c>) -> Attribute<'c> {
        attr.inner
    }
}

/// Represents the `llzk.loopbounds` attribute.
///
/// A trusted annotation of a loop variable's lower bound (inclusive), upper bound
/// (exclusive), and nonzero step. Positive and negative steps are allowed; a zero
/// step is a verification error.
///
/// The producer must ensure these values agree with the loop's behavior, since
/// the bounds are trusted unconditionally.
#[derive(Clone, Copy, Debug)]
pub struct LoopBoundsAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> LoopBoundsAttribute<'c> {
    /// Creates a new attribute from its raw representation.
    ///
    /// # Safety
    ///
    /// The MLIR attribute must contain a valid pointer of type `LoopBoundsAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates loop bounds with an inclusive `begin` and exclusive `end`.
    ///
    /// `step` must be nonzero and may be positive or negative.
    pub fn new(ctx: &'c Context, begin: i64, end: i64, step: i64) -> Self {
        Self::parse(ctx, &begin.to_string(), &end.to_string(), &step.to_string())
    }

    /// Creates loop bounds from signed base-10 integers of arbitrary precision.
    ///
    /// `begin` is inclusive and `end` is exclusive. `step` must represent a
    /// nonzero integer and may be positive or negative.
    ///
    /// # Panics
    ///
    /// If any bound or the step is not a valid decimal integer.
    pub fn parse(ctx: &'c Context, begin: &str, end: &str, step: &str) -> Self {
        let attr = unsafe {
            llzkLlzk_LoopBoundsAttrGet(
                ctx.to_raw(),
                StringRef::new(begin).to_raw(),
                StringRef::new(end).to_raw(),
                StringRef::new(step).to_raw(),
            )
        };
        assert!(!attr.ptr.is_null(), "invalid decimal loop bounds");
        unsafe { Self::from_raw(attr) }
    }
}

impl<'c> AttributeLike<'c> for LoopBoundsAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for LoopBoundsAttribute<'c> {
    type Error = melior::Error;

    fn try_from(t: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Llzk_LoopBoundsAttr(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected(
                "llzk loopbounds",
                t.to_string(),
            ))
        }
    }
}

impl<'c> std::fmt::Display for LoopBoundsAttribute<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<'c> From<LoopBoundsAttribute<'c>> for Attribute<'c> {
    fn from(attr: LoopBoundsAttribute<'c>) -> Attribute<'c> {
        attr.inner
    }
}
