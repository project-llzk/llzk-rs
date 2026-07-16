//! `constrain` dialect.

use crate::builder::OpBuilderLike;
use llzk_sys::{
    llzkConstrain_EmitContainmentOpBuild, llzkConstrain_EmitEqualityOpBuild,
    mlirGetDialectHandle__llzk__constrain__,
};
use melior::{
    dialect::DialectHandle,
    ir::{Location, OperationRef, Value, ValueLike, operation::OperationLike},
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

/// Return `true` iff the given op is `constrain.eq`.
#[inline]
pub fn is_constrain_eq<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "constrain.eq")
}

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

/// Return `true` iff the given op is `constrain.in`.
#[inline]
pub fn is_constrain_in<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "constrain.in")
}
