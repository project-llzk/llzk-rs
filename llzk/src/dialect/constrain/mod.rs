//! `constrain` dialect.

use crate::builder::OpBuilderLike;
use llzk_sys::{
    llzkConstrain_EmitContainmentOpBuild, llzkConstrain_EmitEqualityOpBuild,
    llzkOperationIsA_Constrain_EmitContainmentOp, llzkOperationIsA_Constrain_EmitEqualityOp,
    mlirGetDialectHandle__llzk__constrain__,
};
use melior::{
    dialect::DialectHandle,
    ir::{Location, OperationRef, Value, ValueLike},
};

/// Returns a handle to the `constrain` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__constrain__()) }
}

/// Creates a `constrain.eq` operation.
pub fn eq<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkConstrain_EmitEqualityOpBuild(
            builder.to_raw(),
            location.to_raw(),
            lhs.to_raw(),
            rhs.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(constrain, eq, llzkOperationIsA_Constrain_EmitEqualityOp);

/// Creates a `constrain.in` operation.
pub fn r#in<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkConstrain_EmitContainmentOpBuild(
            builder.to_raw(),
            location.to_raw(),
            lhs.to_raw(),
            rhs.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(constrain, in, llzkOperationIsA_Constrain_EmitContainmentOp);
