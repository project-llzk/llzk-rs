#![allow(unused_crate_dependencies)]
//! Integration tests for the felt dialect.

use llzk::builder::{OpBuilder, OpBuilderLike as _};
use llzk::prelude::*;

mod common;

#[test]
fn f_constant() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_constant",
        FunctionType::new(&context, &[], &[FeltType::new(&context).into()]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[]));
        builder.set_insertion_point_at_start(block);
        let felt =
            dialect::felt::constant(&builder, loc, FeltConstAttribute::new(&context, 42, None))
                .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_constant() -> !felt.type {
  %felt_const_42 = felt.const  42
  function.return %felt_const_42 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_add() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_add",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::add(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_add(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type {
  %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_sub() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_sub",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::sub(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_sub(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type {
  %0 = felt.sub %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_mul() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_mul",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::mul(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_mul(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type {
  %0 = felt.mul %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_div() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_div",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::div(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_div(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type {
  %0 = felt.div %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_uintdiv() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_uintdiv",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::uintdiv(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_uintdiv(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.uintdiv %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_sintdiv() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_sintdiv",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::sintdiv(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_sintdiv(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.sintdiv %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_umod() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_umod",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::umod(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_umod(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.umod %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_smod() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_smod",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::smod(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_smod(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.smod %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_neg() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_neg",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::neg(&builder, loc, block.argument(0).unwrap().into()).unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_neg(%arg0: !felt.type) -> !felt.type {
  %0 = felt.neg %arg0 : !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_inv() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_inv",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::inv(&builder, loc, block.argument(0).unwrap().into()).unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_inv(%arg0: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.inv %arg0 : !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_bit_not() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_bit_not",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt =
            dialect::felt::bit_not(&builder, loc, block.argument(0).unwrap().into()).unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_bit_not(%arg0: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.bit_not %arg0 : !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_shl() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_shl",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::shl(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_shl(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.shl %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_shr() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_shr",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::shr(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_shr(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.shr %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_bit_and() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_bit_and",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::bit_and(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_bit_and(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.bit_and %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_bit_or() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_bit_or",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::bit_or(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_bit_or(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.bit_or %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_bit_xor() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_bit_xor",
        FunctionType::new(&context, &[felt_type, felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_non_native_field_ops_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::felt::bit_xor(
            &builder,
            loc,
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        )
        .unwrap();
        dialect::function::r#return(&builder, loc, &[felt.result(0).unwrap().into()]);
    }

    assert_eq!(f.region_count(), 1);
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @f_bit_xor(%arg0: !felt.type, %arg1: !felt.type) -> !felt.type attributes {function.allow_non_native_field_ops} {
  %0 = felt.bit_xor %arg0, %arg1 : !felt.type, !felt.type
  function.return %0 : !felt.type
}";
    assert_eq!(ir, expected);
}
