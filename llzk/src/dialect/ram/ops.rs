//! `ram` dialect operations.

use melior::ir::{
    Location, Operation, Type, Value,
    operation::{OperationBuilder, OperationLike},
};

/// Creates a `ram.alloc` operation.
///
/// Allocates a memory region of the given size and returns a handle.
pub fn alloc<'c>(location: Location<'c>, size: Value<'c, '_>) -> Operation<'c> {
    let ctx = location.context();
    OperationBuilder::new("ram.alloc", location)
        .add_operands(&[size])
        .add_results(&[Type::index(unsafe { ctx.to_ref() })])
        .build()
        .expect("valid operation")
}

/// Returns `true` iff the given op is `ram.alloc`.
#[inline]
pub fn is_ram_alloc<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "ram.alloc")
}

/// Creates a `ram.load` operation.
///
/// Reads a value from the memory region at the given address.
pub fn load<'c>(
    location: Location<'c>,
    result_type: Type<'c>,
    mem: Value<'c, '_>,
    addr: Value<'c, '_>,
) -> Operation<'c> {
    OperationBuilder::new("ram.load", location)
        .add_operands(&[mem, addr])
        .add_results(&[result_type])
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
/// Writes a value to the memory region at the given address.
pub fn store<'c>(
    location: Location<'c>,
    mem: Value<'c, '_>,
    addr: Value<'c, '_>,
    val: Value<'c, '_>,
) -> Operation<'c> {
    OperationBuilder::new("ram.store", location)
        .add_operands(&[mem, addr, val])
        .build()
        .expect("valid operation")
}

/// Returns `true` iff the given op is `ram.store`.
#[inline]
pub fn is_ram_store<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "ram.store")
}
