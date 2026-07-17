//! `cast` dialect.

use crate::builder::OpBuilderLike;
use crate::prelude::FeltType;
use llzk_sys::{
    LlzkCastOverflowSemantics, llzkAttributeIsA_Cast_OverflowSemanticsAttr,
    llzkCast_FeltToIndexOpBuild, llzkCast_IntToFeltOpBuildWithType,
    llzkCast_OverflowSemanticsAttrGet, llzkCast_OverflowSemanticsAttrGetValue,
    mlirGetDialectHandle__llzk__cast__,
};
use melior::{
    Context,
    dialect::DialectHandle,
    ir::{
        Attribute, AttributeLike, Location, OperationRef, TypeLike, Value, ValueLike,
        operation::OperationLike,
    },
};
use mlir_sys::MlirAttribute;
use std::ptr::null_mut;

/// Overflow behavior for cast operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum OverflowSemantics {
    /// Assert that the input fits.
    Assert = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_ASSERT,
    /// Saturate overflowing values.
    Saturate = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_SATURATE,
    /// Wrap overflowing values.
    Wrap = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_WRAP,
    /// Truncate overflowing values.
    Truncate = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_TRUNCATE,
}

impl From<LlzkCastOverflowSemantics> for OverflowSemantics {
    fn from(value: LlzkCastOverflowSemantics) -> Self {
        match value {
            llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_ASSERT => Self::Assert,
            llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_SATURATE => {
                Self::Saturate
            }
            llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_WRAP => Self::Wrap,
            llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_TRUNCATE => {
                Self::Truncate
            }
            _ => panic!("unknown cast overflow semantics value {value}"),
        }
    }
}

/// Attribute representing cast overflow behavior.
#[derive(Clone, Copy, Debug)]
pub struct OverflowSemanticsAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> OverflowSemanticsAttribute<'c> {
    /// # Safety
    /// The MLIR attribute must contain a valid pointer of type `OverflowSemanticsAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates a new overflow semantics attribute.
    pub fn new(ctx: &'c Context, semantics: OverflowSemantics) -> Self {
        unsafe {
            Self::from_raw(llzkCast_OverflowSemanticsAttrGet(
                ctx.to_raw(),
                semantics as LlzkCastOverflowSemantics,
            ))
        }
    }

    /// Returns the represented overflow semantics.
    pub fn value(&self) -> OverflowSemantics {
        unsafe { llzkCast_OverflowSemanticsAttrGetValue(self.to_raw()) }.into()
    }
}

impl<'c> AttributeLike<'c> for OverflowSemanticsAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for OverflowSemanticsAttribute<'c> {
    type Error = melior::Error;

    fn try_from(attr: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Cast_OverflowSemanticsAttr(attr.to_raw()) } {
            Ok(unsafe { Self::from_raw(attr.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected(
                "llzk cast overflow semantics attr",
                attr.to_string(),
            ))
        }
    }
}

impl<'c> From<OverflowSemanticsAttribute<'c>> for Attribute<'c> {
    fn from(attr: OverflowSemanticsAttribute<'c>) -> Self {
        attr.inner
    }
}

/// Returns a handle to the `cast` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__cast__()) }
}

/// Creates a 'cast.tofelt' operation with the given target `FeltType` or the
/// default "unspecified prime" `FeltType` if `None` is provided.
pub fn tofelt<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    val: Value<'c, '_>,
    out_type: Option<FeltType<'c>>,
) -> OperationRef<'c, 'a> {
    let ctx = location.context();
    let out_type = out_type.unwrap_or_else(|| FeltType::new(unsafe { ctx.to_ref() }));
    unsafe {
        OperationRef::from_raw(llzkCast_IntToFeltOpBuildWithType(
            builder.to_raw(),
            location.to_raw(),
            out_type.to_raw(),
            val.to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `cast.tofelt`.
#[inline]
pub fn is_cast_tofelt<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "cast.tofelt")
}

/// Creates a 'cast.toindex' operation.
pub fn toindex<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    val: Value<'c, '_>,
    overflow: Option<OverflowSemantics>,
) -> OperationRef<'c, 'a> {
    let ctx = location.context();
    let overflow = overflow
        .map(|overflow| OverflowSemanticsAttribute::new(unsafe { ctx.to_ref() }, overflow).to_raw())
        .unwrap_or(MlirAttribute { ptr: null_mut() });
    unsafe {
        OperationRef::from_raw(llzkCast_FeltToIndexOpBuild(
            builder.to_raw(),
            location.to_raw(),
            val.to_raw(),
            overflow,
        ))
    }
}

/// Return `true` iff the given op is `cast.toindex`.
#[inline]
pub fn is_cast_toindex<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "cast.toindex")
}
