use melior::{
    Context,
    ir::{
        Block, BlockLike, Location, Module, RegionLike, Type, Value,
        attribute::IntegerAttribute,
        operation::{OperationLike, OperationRef},
        r#type::FunctionType,
    },
};
use rstest::rstest;

use crate::{
    dialect::{
        function::FuncDefOpLike,
        module::llzk_module,
    },
    test::ctx,
};

use super::ops;

/// Helper: wraps operations inside a `function.def` with `allow_witness` so
/// that RAM ops pass verification. Asserts verification succeeds.
fn witness_fn<'c>(
    module: &Module<'c>,
    ctx: &'c Context,
    location: Location<'c>,
    build: impl FnOnce(&Block<'c>),
) {
    let f = build_fn(module, ctx, location, build, |f| {
        f.set_allow_witness_attr(true);
    });
    assert!(f.verify());
}

/// Helper: wraps operations inside a `function.def` with `allow_constraint`.
/// Asserts that verification *fails* — RAM ops are only permitted inside
/// witness functions.
fn constraint_fn<'c>(
    module: &Module<'c>,
    ctx: &'c Context,
    location: Location<'c>,
    build: impl FnOnce(&Block<'c>),
) {
    let f = build_fn(module, ctx, location, build, |f| {
        f.set_allow_constraint_attr(true);
    });
    assert!(!f.verify());
}

fn build_fn<'c, 'm>(
    module: &'m Module<'c>,
    ctx: &'c Context,
    location: Location<'c>,
    build: impl FnOnce(&Block<'c>),
    configure: impl FnOnce(&crate::dialect::function::FuncDefOp<'c>),
) -> OperationRef<'c, 'm> {
    let f = crate::dialect::function::def(
        location,
        "test_fn",
        FunctionType::new(ctx, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    configure(&f);

    let block = Block::new(&[]);
    build(&block);
    block.append_operation(crate::dialect::function::r#return(location, &[]));
    f.region(0)
        .expect("function.def must have a region")
        .append_block(block);

    module.body().append_operation(f.into())
}

fn build_alloc<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'b> {
    let size_op = block.append_operation(melior::dialect::arith::constant(
        ctx,
        IntegerAttribute::new(Type::index(ctx), 1024).into(),
        location,
    ));
    block.append_operation(ops::alloc(location, size_op.result(0).unwrap().into()))
}

fn build_load<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'b> {
    let index_type = Type::index(ctx);
    let alloc_op = build_alloc(block, ctx, location);
    let mem: Value = alloc_op.result(0).unwrap().into();

    let addr_op = block.append_operation(melior::dialect::arith::constant(
        ctx,
        IntegerAttribute::new(index_type, 42).into(),
        location,
    ));
    let addr: Value = addr_op.result(0).unwrap().into();

    block.append_operation(ops::load(location, index_type, mem, addr))
}

fn build_store<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'b> {
    let index_type = Type::index(ctx);
    let alloc_op = build_alloc(block, ctx, location);
    let mem: Value = alloc_op.result(0).unwrap().into();

    let addr_op = block.append_operation(melior::dialect::arith::constant(
        ctx,
        IntegerAttribute::new(index_type, 0).into(),
        location,
    ));
    let addr: Value = addr_op.result(0).unwrap().into();

    let val_op = block.append_operation(melior::dialect::arith::constant(
        ctx,
        IntegerAttribute::new(index_type, 99).into(),
        location,
    ));
    let val: Value = val_op.result(0).unwrap().into();

    block.append_operation(ops::store(location, mem, addr, val))
}

#[rstest]
fn op_alloc(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    witness_fn(&module, &ctx, location, |block| {
        let alloc = build_alloc(block, &ctx, location);
        assert!(ops::is_ram_alloc(&alloc));
    });
}

#[rstest]
fn op_load(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    witness_fn(&module, &ctx, location, |block| {
        let load = build_load(block, &ctx, location);
        assert!(ops::is_ram_load(&load));
    });
}

#[rstest]
fn op_store(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    witness_fn(&module, &ctx, location, |block| {
        let store = build_store(block, &ctx, location);
        assert!(ops::is_ram_store(&store));
    });
}

#[rstest]
fn op_alloc_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    constraint_fn(&module, &ctx, location, |block| {
        build_alloc(block, &ctx, location);
    });
}

#[rstest]
fn op_load_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    constraint_fn(&module, &ctx, location, |block| {
        build_load(block, &ctx, location);
    });
}

#[rstest]
fn op_store_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    constraint_fn(&module, &ctx, location, |block| {
        build_store(block, &ctx, location);
    });
}
