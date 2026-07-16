use melior::{
    Context,
    dialect::arith,
    ir::{Location, Module, Type, Value, attribute::IntegerAttribute, operation::OperationLike},
};
use rstest::rstest;

use crate::{
    builder::{OpBuilder, OpBuilderLike as _},
    dialect::array::{ArrayCtor, new},
    test::ctx,
};

use super::ArrayType;

#[rstest]
fn type_new_with_dims(ctx: Context) {
    let idx_typ = Type::index(&ctx);
    let arr_typ = ArrayType::new_with_dims(idx_typ, &[2]);

    assert_eq!(arr_typ.element_type(), idx_typ);
    assert_eq!(arr_typ.num_dims(), 1);
    assert_eq!(
        arr_typ.dim(0),
        IntegerAttribute::new(Type::index(&ctx), 2).into()
    );
}

#[rstest]
fn op_new_with_values(ctx: Context) {
    let arr_typ = ArrayType::new_with_dims(Type::index(&ctx), &[2]);
    let module = Module::new(Location::unknown(&ctx));
    let op_builder = OpBuilder::at_block_begin(&ctx, module.body());
    assert_eq!(ctx, module.context());
    let loc = Location::unknown(&ctx);
    let op1 = op_builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(Type::index(ctx), 1).into(), loc)
    });

    let op2 = op_builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(Type::index(ctx), 1).into(), loc)
    });

    let vals: [Value; 2] = [op1.result(0).unwrap().into(), op2.result(0).unwrap().into()];
    let op = new(&op_builder, loc, arr_typ, ArrayCtor::Values(&vals));
    assert!(op.verify());
}
