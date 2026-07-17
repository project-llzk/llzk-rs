use crate::{
    builder::OpBuilderLike,
    dialect::bool::{CmpPredicate, CmpPredicateAttribute},
    error::Error,
    macros::isa_fn,
};
use llzk_sys::{
    llzkBool_AndBoolOpBuild, llzkBool_AssertOpBuild, llzkBool_CmpOpBuild, llzkBool_NotBoolOpBuild,
    llzkBool_OrBoolOpBuild, llzkBool_XorBoolOpBuild,
};
use melior::ir::{
    AttributeLike as _, Block, Identifier, Location, OperationRef, RegionLike as _, Value,
    ValueLike as _, operation::OperationLike as _,
};
use mlir_sys::MlirIdentifier;
use std::ptr::null_mut;

fn build_cmp_op<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    pred: CmpPredicate,
    location: Location<'c>,
    operands: &[Value<'c, '_>],
) -> Result<OperationRef<'c, 'a>, Error> {
    let [lhs, rhs] = operands else {
        return Err(Error::BuildMethodFailed("bool.cmp"));
    };
    let ctx = location.context();
    Ok(unsafe {
        OperationRef::from_raw(llzkBool_CmpOpBuild(
            builder.to_raw(),
            location.to_raw(),
            lhs.to_raw(),
            rhs.to_raw(),
            CmpPredicateAttribute::new(ctx.to_ref(), pred).to_raw(),
        ))
    })
}

macro_rules! cmp_binop {
    ($name:ident, $pred:expr) => {
        #[doc = concat!("Creates a `bool.cmp ", stringify!($name) ,"` operation.")]
        pub fn $name<'c, 'a>(
            builder: &impl OpBuilderLike<'c>,
            location: Location<'c>,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
        ) -> Result<OperationRef<'c, 'a>, Error> {
            build_cmp_op(builder, $pred, location, &[lhs, rhs])
        }
    };
}

cmp_binop!(eq, CmpPredicate::Eq);
cmp_binop!(ge, CmpPredicate::Ge);
cmp_binop!(gt, CmpPredicate::Gt);
cmp_binop!(le, CmpPredicate::Le);
cmp_binop!(lt, CmpPredicate::Lt);
cmp_binop!(ne, CmpPredicate::Ne);

isa_fn!(prefixed bool, cmp);

macro_rules! op {
    ($arity:ident, $($args:tt)*) => {
        crate::macros::dialect_op!($arity untyped bool, $($args)*);
    };
}

op!(binop, and);
op!(binop, or);
op!(binop, xor);
op!(unop, not);

/// Creates a `bool.assert` operation.
pub fn assert<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    cond: Value<'c, '_>,
    msg: Option<&str>,
) -> Result<OperationRef<'c, 'a>, Error> {
    let ctx = location.context();
    let msg = msg
        .map(|msg| Identifier::new(unsafe { ctx.to_ref() }, msg).to_raw())
        .unwrap_or(MlirIdentifier { ptr: null_mut() });
    Ok(unsafe {
        OperationRef::from_raw(llzkBool_AssertOpBuild(
            builder.to_raw(),
            location.to_raw(),
            cond.to_raw(),
            msg,
        ))
    })
}

isa_fn!(prefixed bool, assert);

/// Helper for creating a quantifier op.
fn create_quantifier_body<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    domain: Value<'c, '_>,
    op_build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> Result<OperationRef<'c, 'a>, Error> {
    let op = unsafe {
        OperationRef::from_raw(op_build(
            builder.to_raw(),
            location.to_raw(),
            domain.to_raw(),
        ))
    };

    let region = op.region(0).map_err(Error::Melior)?;
    let iter_type = super::quantifier_iter_type(domain.r#type())?;
    region.append_block(Block::new(&[(iter_type, location)]));
    Ok(op)
}

/// Creates a `bool.forall` operation.
///
/// Adds an empty block with the correct iteration type based on the domain's type.
pub fn forall<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    domain: Value<'c, '_>,
) -> Result<OperationRef<'c, 'a>, Error> {
    create_quantifier_body(builder, location, domain, llzk_sys::llzkBool_ForAllOpBuild)
}

isa_fn!(prefixed bool, forall);

/// Creates a `bool.exists` operation.
///
/// Adds an empty block with the correct iteration type based on the domain's type.
pub fn exists<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    domain: Value<'c, '_>,
) -> Result<OperationRef<'c, 'a>, Error> {
    create_quantifier_body(builder, location, domain, llzk_sys::llzkBool_ExistsOpBuild)
}

isa_fn!(prefixed bool, exists);

/// Creates a `bool.yield` operation.
pub fn r#yield<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzk_sys::llzkBool_YieldOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(prefixed bool, r#yield);
