#![allow(unused_crate_dependencies)]
//! Integration tests for the cast dialect.

use llzk::builder::{OpBuilder, OpBuilderLike as _};
use llzk::dialect::cast::*;
use llzk::prelude::melior_dialects::arith;
use llzk::prelude::*;

mod common;

#[test]
fn tofelt_unspecified() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&ctx, module.body());
    let index_ty = Type::index(&ctx);

    let c = builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(index_ty, 0).into(), loc)
    });
    let a = tofelt(&builder, loc, c.result(0).unwrap().into(), None);

    let ir = format!("{a}");
    let expected = "%0 = cast.tofelt %c0 : index, !felt.type";
    assert_eq!(ir, expected);
}

#[test]
fn tofelt_specified() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&ctx, module.body());
    let index_ty = Type::index(&ctx);
    let felt_ty = FeltType::with_field(&ctx, "babybear");

    let c = builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(index_ty, 0).into(), loc)
    });
    let a = tofelt(&builder, loc, c.result(0).unwrap().into(), Some(felt_ty));

    let ir = format!("{a}");
    let expected = "%0 = cast.tofelt %c0 : index, !felt.type<\"babybear\">";
    assert_eq!(ir, expected);
}

#[test]
fn toindex_unspecified_overflow() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&ctx, module.body());

    let felt = dialect::felt::constant(&builder, loc, FeltConstAttribute::new(&ctx, 0, None))
        .expect("valid felt const");
    let index = toindex(&builder, loc, felt.result(0).unwrap().into(), None);

    assert!(index.verify());
    assert!(!format!("{index}").contains("overflow"));
}

#[test]
fn overflow_semantics_attr_round_trip() {
    common::setup();
    let ctx = LlzkContext::new();
    let attr = OverflowSemanticsAttribute::new(&ctx, OverflowSemantics::Wrap);

    assert_eq!(attr.value(), OverflowSemantics::Wrap);
    let attr: Attribute = attr.into();
    let attr = OverflowSemanticsAttribute::try_from(attr).unwrap();
    assert_eq!(attr.value(), OverflowSemantics::Wrap);
}
