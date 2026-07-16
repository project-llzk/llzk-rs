use super::ops;
use crate::{
    builder::{OpBuilder, OpBuilderLike},
    dialect::{
        felt::{FeltConstAttribute, FeltType},
        function::FuncDefOpLike,
        module::llzk_module,
    },
    test::ctx,
};
use melior::{
    Context,
    ir::{
        Location, Module, RegionLike as _, Type, Value,
        attribute::IntegerAttribute,
        operation::{OperationLike as _, OperationRef},
        r#type::FunctionType,
    },
};
use rstest::rstest;

/// Helper: wraps operations inside a `function.def` with `allow_witness` so
/// that RAM ops pass verification. Asserts verification succeeds.
fn witness_fn_passes<'c>(
    module: &Module<'c>,
    ctx: &'c Context,
    location: Location<'c>,
    build: impl FnOnce(&OpBuilder<'c, '_>),
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
    build: impl FnOnce(&OpBuilder<'c, '_>),
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
    build: impl FnOnce(&OpBuilder<'c, '_>),
    configure: impl FnOnce(&crate::dialect::function::FuncDefOpRef<'c, 'm>),
) -> OperationRef<'c, 'm> {
    let builder = OpBuilder::at_block_end(ctx, module.body());
    let f = crate::dialect::function::def(
        &builder,
        location,
        "test_fn",
        FunctionType::new(ctx, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    configure(&f);

    let block = f
        .body()
        .expect("function.def must have a body")
        .append_block(melior::ir::Block::new(&[]));
    builder.set_insertion_point_at_start(block);
    build(&builder);
    crate::dialect::function::r#return(&builder, location, &[]);

    f.into()
}

fn build_addr<'c: 'a, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: i64,
) -> Value<'c, 'a> {
    let addr_op = builder.insert(location, |ctx, location| {
        melior::dialect::arith::constant(
            ctx,
            IntegerAttribute::new(Type::index(ctx), value).into(),
            location,
        )
    });
    addr_op.result(0).unwrap().into()
}

fn build_load<'c: 'a, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
) -> OperationRef<'c, 'a> {
    let addr = build_addr(builder, location, 42);
    ops::load(builder, location, addr, None)
}

fn build_store<'c: 'a, 'a>(
    builder: &impl OpBuilderLike<'c>,
    ctx: &'c Context,
    location: Location<'c>,
) -> OperationRef<'c, 'a> {
    let addr = build_addr(builder, location, 0);
    let val_op =
        crate::dialect::felt::constant(builder, location, FeltConstAttribute::new(ctx, 99, None))
            .expect("valid felt.const");
    let val: Value = val_op.result(0).unwrap().into();

    ops::store(builder, location, addr, val)
}

#[rstest]
fn op_load(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location, None);

    witness_fn_passes(&module, &ctx, location, |builder| {
        let load = build_load(builder, location);
        assert!(ops::is_ram_load(&load));
    });
}

#[rstest]
fn op_store(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location, None);

    witness_fn_passes(&module, &ctx, location, |builder| {
        let store = build_store(builder, &ctx, location);
        assert!(ops::is_ram_store(&store));
    });
}

#[rstest]
fn op_load_with_specified_field(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location, None);

    witness_fn_passes(&module, &ctx, location, |builder| {
        let addr = build_addr(builder, location, 42);
        let felt_ty = FeltType::with_field(&ctx, "bn254");
        let load = ops::load(builder, location, addr, Some(felt_ty));
        assert!(ops::is_ram_load(&load));
    });
}

#[rstest]
fn op_load_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location, None);

    constraint_fn_rejected(&module, &ctx, location, |builder| {
        build_load(builder, location);
    });
}

#[rstest]
fn op_store_rejected_in_constraint_fn(ctx: Context) {
    let location = Location::unknown(&ctx);
    let module = llzk_module(location, None);

    constraint_fn_rejected(&module, &ctx, location, |builder| {
        build_store(builder, &ctx, location);
    });
}
