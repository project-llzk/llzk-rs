#![allow(unused_crate_dependencies)]
//! Integration tests for the bool dialect.

use llzk::builder::{OpBuilder, OpBuilderLike as _};
use llzk::prelude::*;
use melior::dialect::arith;

mod common;

#[test]
fn f_eq() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_eq",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::eq(
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
    let expected = r"function.def @f_eq(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp eq(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_ne() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_ne",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::ne(
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
    let expected = r"function.def @f_ne(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp ne(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_lt() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_lt",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::lt(
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
    let expected = r"function.def @f_lt(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp lt(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_le() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_le",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::le(
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
    let expected = r"function.def @f_le(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp le(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_gt() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_gt",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::gt(
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
    let expected = r"function.def @f_gt(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp gt(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_ge() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let bool_type: Type = IntegerType::new(&context, 1).into();
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let f = dialect::function::def(
        &builder,
        loc,
        "f_ge",
        FunctionType::new(&context, &[felt_type, felt_type], &[bool_type]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = f
            .body()
            .expect("function.def must have body region")
            .append_block(Block::new(&[(felt_type, loc), (felt_type, loc)]));
        builder.set_insertion_point_at_start(block);
        let felt = dialect::bool::ge(
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
    let expected = r"function.def @f_ge(%arg0: !felt.type, %arg1: !felt.type) -> i1 attributes {function.allow_witness} {
  %0 = bool.cmp ge(%arg0, %arg1) : !felt.type, !felt.type
  function.return %0 : i1
}";
    assert_eq!(ir, expected);
}

#[test]
fn f_assert() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let cond = arith::constant(&context, BoolAttribute::new(&context, false).into(), loc);
    let cond = cond.result(0).unwrap().into();
    let op = llzk::dialect::bool::assert(&builder, loc, cond, Some("assertion failed"))
        .expect("failed to build assert op");

    assert!(op.verify());
    log::info!("Op passed verification");
}
