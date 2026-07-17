#![allow(unused_crate_dependencies)]
//! Integration tests for symbol table behavior.

use llzk::builder::{OpBuilder, OpBuilderLike};
use llzk::dialect::module::llzk_module;
use llzk::prelude::{BlockLike, FuncDefOpLike as _, FuncDefOpRef, FunctionType, LlzkContext};
use llzk::symbol_table;
use llzk_sys::llzkFunction_FuncDefOpBuildWithAttrsAndArgAttrs;
use melior::{
    StringRef,
    ir::{Location, Operation, TypeLike as _},
};

mod common;

#[inline]
fn make_empty_func<'c>(builder: &impl OpBuilderLike<'c>, name: &str) -> Operation<'c> {
    let context = unsafe { builder.context().to_ref() };
    unsafe {
        Operation::from_raw(llzkFunction_FuncDefOpBuildWithAttrsAndArgAttrs(
            builder.to_raw(),
            Location::unknown(context).to_raw(),
            StringRef::new(name).to_raw(),
            FunctionType::new(context, &[], &[]).to_raw(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
        ))
    }
}

#[test]
fn insert_renames_symbols_on_collision() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let builder = unsafe { OpBuilder::from_raw(llzk_sys::mlirOpBuilderCreate(context.to_raw())) };

    let first = FuncDefOpRef::try_from(
        module
            .body()
            .append_operation(make_empty_func(&builder, "foo")),
    )
    .unwrap();

    let module_op = module.as_operation();
    let inserted = symbol_table::insert(&module_op, make_empty_func(&builder, "foo"));
    let second = FuncDefOpRef::try_from(inserted).unwrap();

    assert_eq!(format!("{}", first.fully_qualified_name()), "@foo");
    assert_ne!(format!("{}", second.fully_qualified_name()), "@foo");
}
