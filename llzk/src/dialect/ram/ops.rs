//! `ram` dialect operations.

use crate::builder::OpBuilderLike;
use crate::dialect::felt::FeltType;
use llzk_sys::{
    llzkOperationIsA_Ram_LoadOp, llzkOperationIsA_Ram_StoreOp, llzkRam_LoadOpBuild,
    llzkRam_StoreOpBuild,
};
use melior::ir::{Location, OperationRef, TypeLike, Value, ValueLike};

/// Creates a `ram.load` operation with the given target `FeltType` or the
/// default "unspecified prime" `FeltType` if `None` is provided.
pub fn load<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    addr: Value<'c, '_>,
    out_type: Option<FeltType<'c>>,
) -> OperationRef<'c, 'a> {
    let ctx = location.context();
    let out_type = out_type.unwrap_or_else(|| FeltType::new(unsafe { ctx.to_ref() }));
    unsafe {
        OperationRef::from_raw(llzkRam_LoadOpBuild(
            builder.to_raw(),
            location.to_raw(),
            out_type.to_raw(),
            addr.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(ram, load, llzkOperationIsA_Ram_LoadOp);

/// Creates a `ram.store` operation.
///
/// Writes a value to the flat memory region at the given address.
pub fn store<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    addr: Value<'c, '_>,
    val: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkRam_StoreOpBuild(
            builder.to_raw(),
            location.to_raw(),
            addr.to_raw(),
            val.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(ram, store, llzkOperationIsA_Ram_StoreOp);
