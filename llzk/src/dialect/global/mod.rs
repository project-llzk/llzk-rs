//! `global` dialect.

use crate::{builder::OpBuilderLike, symbol_ref::SymbolRefAttribute};
use llzk_sys::{
    llzkGlobal_GlobalDefOpBuild, llzkGlobal_GlobalReadOpBuild, llzkGlobal_GlobalWriteOpBuild,
    llzkOperationIsA_Global_GlobalDefOp, llzkOperationIsA_Global_GlobalReadOp,
    llzkOperationIsA_Global_GlobalWriteOp, mlirGetDialectHandle__llzk__global__,
};
use melior::{
    dialect::DialectHandle,
    ir::{
        Attribute, AttributeLike, Identifier, Location, OperationRef, Type, TypeLike, Value,
        ValueLike, attribute::TypeAttribute,
    },
};
use mlir_sys::MlirAttribute;
use std::ptr::null_mut;

/// Returns a handle to the `global` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__global__()) }
}

/// Constructs a 'global.def' operation.
pub fn def<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: &str,
    r#type: Type<'c>,
    constant: bool,
    initial_value: Option<Attribute<'c>>,
) -> OperationRef<'c, 'a> {
    let ctx = location.context();
    let null_attr = MlirAttribute { ptr: null_mut() };
    let constant = if constant {
        Attribute::unit(unsafe { ctx.to_ref() }).to_raw()
    } else {
        null_attr
    };
    unsafe {
        OperationRef::from_raw(llzkGlobal_GlobalDefOpBuild(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(ctx.to_ref(), name).to_raw(),
            constant,
            TypeAttribute::new(r#type).to_raw(),
            initial_value.map_or(null_attr, |attr| attr.to_raw()),
        ))
    }
}

crate::macros::isa_fn!(global, def, llzkOperationIsA_Global_GlobalDefOp);

/// Constructs a 'global.read' operation.
pub fn read<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: SymbolRefAttribute<'c>,
    result: Type<'c>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkGlobal_GlobalReadOpBuild(
            builder.to_raw(),
            location.to_raw(),
            result.to_raw(),
            name.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(global, read, llzkOperationIsA_Global_GlobalReadOp);

/// Constructs a 'global.write' operation.
pub fn write<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: SymbolRefAttribute<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkGlobal_GlobalWriteOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
            name.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(global, write, llzkOperationIsA_Global_GlobalWriteOp);
