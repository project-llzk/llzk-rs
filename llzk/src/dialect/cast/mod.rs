//! `cast` dialect.

use crate::{
    builder::{OpBuilder, OpBuilderLike},
    operation::detach_owned_operation,
    prelude::FeltType,
};
use llzk_sys::{
    LlzkCastOverflowSemantics, llzkAttributeIsA_Cast_OverflowSemanticsAttr,
    llzkCast_FeltToIndexOpBuild, llzkCast_IntToFeltOpBuildWithType,
    llzkCast_IntToFeltOpSetOverflow, llzkCast_OverflowSemanticsAttrGet,
    llzkCast_OverflowSemanticsAttrGetValue, llzkOperationIsA_Cast_FeltToIndexOp,
    llzkOperationIsA_Cast_IntToFeltOp, mlirGetDialectHandle__llzk__cast__,
};
use melior::{
    Context,
    dialect::DialectHandle,
    ir::{Attribute, AttributeLike, Block, Location, OperationRef, TypeLike, Value, ValueLike},
};
use mlir_sys::MlirAttribute;
use std::ptr::null_mut;

/// Overflow behavior for cast operations.
///
/// The safe range is `[0, p-1]` for a felt target with prime `p`, and
/// `[0, 2^(N-1)-1]` for an index target with bit width `N`. Boolean (`i1`)
/// inputs represent zero or one; index inputs are signed. Felt inputs are
/// reduced to their canonical field representatives before conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum OverflowSemantics {
    /// Fail if the input is outside the safe range.
    /// Potentially failing assertions remain observable even if the result is unused.
    Assert = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_ASSERT,
    /// Clamp the input to the safe range.
    Saturate = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_SATURATE,
    /// Reduce modulo `p` for a felt target, or modulo `2^(N-1)` for an index target,
    /// using a nonnegative remainder.
    Wrap = llzk_sys::LlzkCastOverflowSemantics_LlzkCastOverflowSemantics_WRAP,
    /// Keep the low `bitwidth(p)` bits, then reduce modulo `p` for a felt target.
    /// For an index target, keep the low `N-1` bits, clearing the sign bit.
    ///
    /// This differs from wrapping for felt targets: with `p = 7`, truncating `-1`
    /// yields `0`, while wrapping yields `6`.
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
///
/// Uses the upstream LLZK default overflow semantics. Use [`tofelt_with_overflow`]
/// to select a mode explicitly. Boolean (`i1`) inputs map to zero or one; index inputs
/// are signed. Boolean casts are always infallible.
/// Constant folding defers until the target field is specified.
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

/// Creates a 'cast.tofelt' operation with explicit overflow semantics.
///
/// As with [`tofelt`], `None` selects an unspecified target field. See
/// [`OverflowSemantics`] for the safe ranges and behavior of each mode.
/// Non-assert modes are side-effect-free and speculatable. Constant folding
/// defers when the target field is unspecified or an assertion would fail.
pub fn tofelt_with_overflow<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    val: Value<'c, '_>,
    out_type: Option<FeltType<'c>>,
    overflow: OverflowSemantics,
) -> OperationRef<'c, 'a> {
    builder.insert(location, |ctx, location| {
        // The upstream typed builder does not accept overflow semantics. Configure the
        // operation in a scratch block so insertion listeners see the final attributes.
        let block = Block::new(&[]);
        let scratch = OpBuilder::at_block_end(ctx, &block);
        let op = tofelt(&scratch, location, val, out_type);
        unsafe {
            llzkCast_IntToFeltOpSetOverflow(
                op.to_raw(),
                OverflowSemanticsAttribute::new(ctx, overflow).to_raw(),
            );
        }
        detach_owned_operation(op)
    })
}

crate::macros::isa_fn!(cast, tofelt, llzkOperationIsA_Cast_IntToFeltOp);

/// Creates a 'cast.toindex' operation.
///
/// Uses the upstream LLZK default overflow semantics. Use [`toindex_with_overflow`]
/// to select a mode explicitly. The input is reduced in its field before applying the
/// overflow mode to the index safe range `[0, 2^(N-1)-1]`.
/// Constant folding defers if the field is unspecified or an assertion would fail.
/// An assertion remains observable even if the result is unused, unless the source
/// field's entire canonical range fits in the index safe range.
pub fn toindex<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    val: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkCast_FeltToIndexOpBuild(
            builder.to_raw(),
            location.to_raw(),
            val.to_raw(),
            MlirAttribute { ptr: null_mut() },
        ))
    }
}

/// Creates a 'cast.toindex' operation with explicit overflow semantics.
///
/// As with [`toindex`], the input is reduced in its field before conversion. See
/// [`OverflowSemantics`] for the safe ranges and behavior of each mode.
/// Non-assert modes are side-effect-free and speculatable. Constant folding
/// defers when the source field is unspecified or an assertion would fail.
pub fn toindex_with_overflow<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    val: Value<'c, '_>,
    overflow: OverflowSemantics,
) -> OperationRef<'c, 'a> {
    let ctx = location.context();
    let overflow = OverflowSemanticsAttribute::new(unsafe { ctx.to_ref() }, overflow);
    unsafe {
        OperationRef::from_raw(llzkCast_FeltToIndexOpBuild(
            builder.to_raw(),
            location.to_raw(),
            val.to_raw(),
            overflow.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(cast, toindex, llzkOperationIsA_Cast_FeltToIndexOp);
