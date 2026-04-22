use llzk::dialect;
use llzk::dialect::module::llzk_module;
use llzk::prelude::{BlockLike, FuncDefOp, FuncDefOpLike, FuncDefOpRef, FunctionType, LlzkContext};
use llzk::symbol_table;
use melior::ir::Location;

mod common;

#[inline]
fn make_empty_func<'c>(context: &'c LlzkContext, name: &str) -> FuncDefOp<'c> {
    dialect::function::def(
        Location::unknown(context),
        name,
        FunctionType::new(context, &[], &[]),
        &[],
        None,
    )
    .unwrap()
}

#[test]
fn insert_renames_symbols_on_collision() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));

    let first = FuncDefOpRef::try_from(
        module
            .body()
            .append_operation(make_empty_func(&context, "foo").into()),
    )
    .unwrap();

    let module_op = module.as_operation();
    let inserted = symbol_table::insert(&module_op, make_empty_func(&context, "foo").into());
    let second = FuncDefOpRef::try_from(inserted).unwrap();

    assert_eq!(format!("{}", first.fully_qualified_name()), "@foo");
    assert_ne!(format!("{}", second.fully_qualified_name()), "@foo");
}
