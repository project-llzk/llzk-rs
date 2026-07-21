#![allow(unused_crate_dependencies)]
//! Integration tests for symbol table behavior.

use llzk::builder::{OpBuilder, OpBuilderLike};
use llzk::dialect::module::llzk_module;
use llzk::prelude::{
    FuncDefOpLike as _, FuncDefOpRef, FunctionType, LlzkContext, Location, Operation, dialect,
};
use llzk::symbol_table;

mod common;

#[inline]
fn make_empty_func<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    context: &'c LlzkContext,
    location: Location<'c>,
    name: &str,
) -> FuncDefOpRef<'c, 'a> {
    dialect::function::def(
        builder,
        location,
        name,
        FunctionType::new(context, &[], &[]),
        &[],
        None,
        llzk::dialect::empty_region,
    )
    .unwrap()
}

#[test]
fn insert_renames_symbols_on_collision() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let first = make_empty_func(&builder, &context, loc, "foo");
    let duplicate = unsafe { Operation::from_raw(llzk_sys::mlirOperationClone(first.to_raw())) };

    let module_op = module.as_operation();
    let inserted = symbol_table::insert(module_op, duplicate);
    let second = FuncDefOpRef::try_from(inserted).unwrap();

    assert_eq!(format!("{}", first.fully_qualified_name()), "@foo");
    assert_ne!(format!("{}", second.fully_qualified_name()), "@foo");
}
