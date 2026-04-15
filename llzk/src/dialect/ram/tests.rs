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
        felt::{FeltConstAttribute, FeltType},
        function::FuncDefOpLike,
        module::llzk_module,
    },
    test::ctx,
};

use super::ops;

/// Helper: wraps operations inside a `function.def` with `allow_witness` so
/// that RAM ops pass verification. Asserts verification succeeds.
fn witness_fn_passes<'c>(
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
fn constraint_fn_rejected<'c>(
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

fn build_addr<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
    value: i64,
) -> Value<'c, 'b> {
    let addr_op = block.append_operation(melior::dialect::arith::constant(
        ctx,
        IntegerAttribute::new(Type::index(ctx), value).into(),
        location,
    ));
    addr_op.result(0).unwrap().into()
}

fn build_load<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'b> {
    let element_type: Type = FeltType::new(ctx).into();
    let addr = build_addr(block, ctx, location, 42);
    block.append_operation(ops::load(location, element_type, addr))
}

fn build_store<'c, 'b>(
    block: &'b Block<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'b> {
    let addr = build_addr(block, ctx, location, 0);
    let val_op = block.append_operation(
        crate::dialect::felt::constant(location, FeltConstAttribute::new(ctx, 99, None))
            .expect("valid felt.const"),
    );
    let val: Value = val_op.result(0).unwrap().into();

    block.append_operation(ops::store(location, addr, val))
}

#[rstest]
fn op_load(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    witness_fn_passes(&module, &ctx, location, |block| {
        let load = build_load(block, &ctx, location);
        assert!(ops::is_ram_load(&load));
    });
}

#[rstest]
fn op_store(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    witness_fn_passes(&module, &ctx, location, |block| {
        let store = build_store(block, &ctx, location);
        assert!(ops::is_ram_store(&store));
    });
}

#[rstest]
fn op_load_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    constraint_fn_rejected(&module, &ctx, location, |block| {
        build_load(block, &ctx, location);
    });
}

#[rstest]
fn op_store_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location);

    constraint_fn_rejected(&module, &ctx, location, |block| {
        build_store(block, &ctx, location);
    });
}
