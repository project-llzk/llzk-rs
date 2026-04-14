//! `cast` dialect.

use llzk_sys::mlirGetDialectHandle__llzk__cast__;
use melior::dialect::DialectHandle;
use melior::ir::{
    Location, Operation, Type, Value,
    operation::{OperationBuilder, OperationLike},
};

use crate::prelude::FeltType;

/// Returns a handle to the `cast` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__cast__()) }
}

/// Creates a 'cast.tofelt' operation with the given target `FeltType` or the
/// default "unspecified prime" `FeltType` if `None` is provided.
pub fn tofelt<'c>(
    location: Location<'c>,
    val: Value<'c, '_>,
    out_type: Option<FeltType<'c>>,
) -> Operation<'c> {
    let ctx = location.context();
    let out_type = out_type.unwrap_or_else(|| FeltType::new(unsafe { ctx.to_ref() }));
    OperationBuilder::new("cast.tofelt", location)
        .add_results(&[out_type.into()])
        .add_operands(&[val])
        .build()
        .expect("valid operation")
}

/// Return `true` iff the given op is `cast.tofelt`.
#[inline]
pub fn is_cast_tofelt<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "cast.tofelt")
}

/// Creates a 'cast.toindex' operation.
pub fn toindex<'c>(location: Location<'c>, val: Value<'c, '_>) -> Operation<'c> {
    let ctx = location.context();
    OperationBuilder::new("cast.toindex", location)
        .add_results(&[Type::index(unsafe { ctx.to_ref() })])
        .add_operands(&[val])
        .build()
        .expect("valid operation")
}

/// Return `true` iff the given op is `cast.toindex`.
#[inline]
pub fn is_cast_toindex<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "cast.toindex")
}
