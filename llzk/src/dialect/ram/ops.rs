//! `ram` dialect operations.

use melior::ir::{
    Location, Operation, Value,
    operation::{OperationBuilder, OperationLike},
};

use crate::dialect::felt::FeltType;

/// Creates a `ram.load` operation with the given target `FeltType` or the
/// default "unspecified prime" `FeltType` if `None` is provided.
pub fn load<'c>(
    location: Location<'c>,
    addr: Value<'c, '_>,
    out_type: Option<FeltType<'c>>,
) -> Operation<'c> {
    let ctx = location.context();
    let out_type = out_type.unwrap_or_else(|| FeltType::new(unsafe { ctx.to_ref() }));
    OperationBuilder::new("ram.load", location)
        .add_operands(&[addr])
        .add_results(&[out_type.into()])
        .build()
        .expect("valid operation")
}

/// Returns `true` iff the given op is `ram.load`.
#[inline]
pub fn is_ram_load<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "ram.load")
}

/// Creates a `ram.store` operation.
///
/// Writes a value to the flat memory region at the given address.
pub fn store<'c>(location: Location<'c>, addr: Value<'c, '_>, val: Value<'c, '_>) -> Operation<'c> {
    OperationBuilder::new("ram.store", location)
        .add_operands(&[addr, val])
        .build()
        .expect("valid operation")
}

/// Returns `true` iff the given op is `ram.store`.
#[inline]
pub fn is_ram_store<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "ram.store")
}
