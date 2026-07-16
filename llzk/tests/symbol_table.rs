#![allow(unused_crate_dependencies)]
//! Integration tests for symbol table behavior.

use llzk::dialect::module::llzk_module;
use llzk::prelude::{BlockLike, FuncDefOpLike as _, FuncDefOpRef, FunctionType, LlzkContext};
use llzk::symbol_table;
use llzk_sys::llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs;
use melior::{
    StringRef,
    ir::{Location, Operation, TypeLike as _},
};

mod common;

#[inline]
fn make_empty_func<'c>(context: &'c LlzkContext, name: &str) -> Operation<'c> {
    unsafe {
        Operation::from_raw(llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
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

    let first = FuncDefOpRef::try_from(
        module
            .body()
            .append_operation(make_empty_func(&context, "foo")),
    )
    .unwrap();

    let module_op = module.as_operation();
    let inserted = symbol_table::insert(&module_op, make_empty_func(&context, "foo"));
    let second = FuncDefOpRef::try_from(inserted).unwrap();

    assert_eq!(format!("{}", first.fully_qualified_name()), "@foo");
    assert_ne!(format!("{}", second.fully_qualified_name()), "@foo");
}
