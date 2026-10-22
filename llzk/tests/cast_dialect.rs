#![allow(unused_crate_dependencies)]
//! Integration tests for the cast dialect.

use llzk::{
    builder::{EntryPoint, InsertPoint, OpBuilder, OpBuilderLike as _, SimpleOpBuilderListener},
    prelude::{dialect::cast::*, melior_dialects::arith, *},
};
use melior::pass::transform::{create_canonicalizer_pass, create_cse_pass};
use rstest::rstest;
use std::cell::RefCell;

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

    assert!(a.verify());
    assert!(is_tofelt_op(&a));
    assert_eq!(a.result(0).unwrap().r#type(), FeltType::new(&ctx).into());
    assert_eq!(a.operand(0).unwrap(), c.result(0).unwrap().into());
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

    assert!(a.verify());
    assert!(is_tofelt_op(&a));
    assert_eq!(a.result(0).unwrap().r#type(), felt_ty.into());
    assert_eq!(a.operand(0).unwrap(), c.result(0).unwrap().into());
}

#[test]
fn toindex_default_overflow() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&ctx, module.body());

    let felt = dialect::felt::constant(&builder, loc, FeltConstAttribute::new(&ctx, 0, None))
        .expect("valid felt const");
    let index = toindex(&builder, loc, felt.result(0).unwrap().into());

    assert!(index.verify());
    assert!(is_toindex_op(&index));
    assert_eq!(index.result(0).unwrap().r#type(), Type::index(&ctx));
    assert_eq!(index.operand(0).unwrap(), felt.result(0).unwrap().into());
}

#[rstest]
fn overflow_semantics_attr_round_trip(
    #[values(
        OverflowSemantics::Assert,
        OverflowSemantics::Saturate,
        OverflowSemantics::Wrap,
        OverflowSemantics::Truncate
    )]
    semantics: OverflowSemantics,
) {
    common::setup();
    let ctx = LlzkContext::new();
    let attr = OverflowSemanticsAttribute::new(&ctx, semantics);

    assert_eq!(attr.value(), semantics);
    let attr: Attribute = attr.into();
    let attr = OverflowSemanticsAttribute::try_from(attr).unwrap();
    assert_eq!(attr.value(), semantics);
}

fn overflow(op: OperationRef<'_, '_>) -> OverflowSemantics {
    OverflowSemanticsAttribute::try_from(op.attribute("overflow").unwrap())
        .unwrap()
        .value()
}

#[rstest]
#[case(OverflowSemantics::Assert, "")]
#[case(OverflowSemantics::Saturate, " sat")]
#[case(OverflowSemantics::Wrap, " wrap")]
#[case(OverflowSemantics::Truncate, " trunc")]
fn cast_overflow_builders(
    #[case] semantics: OverflowSemantics,
    #[case] keyword: &str,
    #[values(None, Some("babybear"))] field: Option<&str>,
) {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&ctx, module.body());
    let c = builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(Type::index(ctx), 1).into(), loc)
    });
    let inserted = RefCell::new(Vec::new());
    let builder = OpBuilder::new_with_listener(
        &ctx,
        EntryPoint::End(module.body()),
        SimpleOpBuilderListener::new(
            |op: OperationRef<'_, '_>, _: InsertPoint<'_, '_>| {
                inserted.borrow_mut().push(op.to_string());
            },
            |_: BlockRef<'_, '_>, _: Option<RegionRef<'_, '_>>, _: Option<BlockRef<'_, '_>>| {},
        ),
    );
    let felt_ty = field.map(|field| FeltType::with_field(&ctx, field));
    let felt = tofelt_with_overflow(
        &builder,
        loc,
        c.result(0).unwrap().into(),
        felt_ty,
        semantics,
    );
    let index = toindex_with_overflow(&builder, loc, felt.result(0).unwrap().into(), semantics);

    assert!(felt.verify());
    assert!(index.verify());
    assert!(is_tofelt_op(&felt));
    assert!(is_toindex_op(&index));
    assert_eq!(overflow(felt), semantics);
    assert_eq!(overflow(index), semantics);
    assert_eq!(
        felt.result(0).unwrap().r#type(),
        felt_ty.unwrap_or_else(|| FeltType::new(&ctx)).into()
    );
    assert_eq!(index.result(0).unwrap().r#type(), Type::index(&ctx));
    assert_eq!(felt.operand(0).unwrap(), c.result(0).unwrap().into());
    assert_eq!(index.operand(0).unwrap(), felt.result(0).unwrap().into());
    assert!(
        felt.to_string()
            .contains(&format!("cast.tofelt{keyword} %"))
    );
    assert!(
        index
            .to_string()
            .contains(&format!("cast.toindex{keyword} %"))
    );

    // Listeners must see the requested field and overflow mode at insertion time.
    assert_eq!(
        *inserted.borrow(),
        vec![felt.to_string(), index.to_string()]
    );

    let ir = module.as_operation().to_string();
    let parsed = Module::parse(&ctx, &ir).unwrap();
    assert!(parsed.as_operation().verify());
    assert_eq!(parsed.as_operation().to_string(), ir);
}

fn cast_function<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    inputs: &[Type<'c>],
    results: &[Type<'c>],
) -> BlockRef<'c, 'c> {
    let loc = Location::unknown(ctx);
    let builder = OpBuilder::at_block_begin(ctx, module.body());
    let function = dialect::function::def(
        &builder,
        loc,
        "cast",
        FunctionType::new(ctx, inputs, results),
        &[],
        None,
        llzk::dialect::empty_region,
    )
    .unwrap();
    function.set_allow_non_native_field_ops_attr(true);
    function.body().unwrap().first_block().unwrap()
}

fn canonicalize(ctx: &Context, module: &mut Module<'_>) {
    assert!(module.as_operation().verify());
    let pm = PassManager::new(ctx);
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.run(module).unwrap();
    assert!(module.as_operation().verify());
}

fn assert_folded_return(block: BlockRef<'_, '_>, expected: Attribute<'_>) {
    let constant = block.first_operation().unwrap();
    assert_eq!(constant.attribute("value").unwrap(), expected);
    let ret = constant.next_in_block().unwrap();
    assert_eq!(
        ret.name().as_string_ref().as_str().unwrap(),
        "function.return"
    );
    assert_eq!(ret.operand(0).unwrap(), constant.result(0).unwrap().into());
}

#[rstest]
#[case(1, false, Some("babybear"), OverflowSemantics::Assert, Some(1))]
#[case(
    2013265920,
    false,
    Some("babybear"),
    OverflowSemantics::Assert,
    Some(2013265920)
)]
#[case(-1, false, Some("babybear"), OverflowSemantics::Assert, None)]
#[case(2013265921, false, Some("babybear"), OverflowSemantics::Assert, None)]
#[case(-1, false, Some("babybear"), OverflowSemantics::Saturate, Some(0))]
#[case(
    2013265921,
    false,
    Some("babybear"),
    OverflowSemantics::Saturate,
    Some(2013265920)
)]
#[case(-1, false, Some("babybear"), OverflowSemantics::Wrap, Some(2013265920))]
#[case(2013265921, false, Some("babybear"), OverflowSemantics::Wrap, Some(0))]
#[case(-1, false, Some("babybear"), OverflowSemantics::Truncate, Some(134217726))]
#[case(
    2147483648,
    false,
    Some("babybear"),
    OverflowSemantics::Truncate,
    Some(0)
)]
#[case(1, true, Some("babybear"), OverflowSemantics::Assert, Some(1))]
#[case(1, true, Some("babybear"), OverflowSemantics::Saturate, Some(1))]
#[case(1, true, Some("babybear"), OverflowSemantics::Wrap, Some(1))]
#[case(1, true, Some("babybear"), OverflowSemantics::Truncate, Some(1))]
#[case(0, false, None, OverflowSemantics::Assert, None)]
#[case(0, false, None, OverflowSemantics::Saturate, None)]
#[case(0, false, None, OverflowSemantics::Wrap, None)]
#[case(0, false, None, OverflowSemantics::Truncate, None)]
fn tofelt_constant_folding(
    #[case] value: i64,
    #[case] boolean: bool,
    #[case] field: Option<&str>,
    #[case] semantics: OverflowSemantics,
    #[case] expected: Option<u64>,
) {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let mut module = LlzkModuleBuilder::create(loc, None);
    let felt_ty = field.map(|field| FeltType::with_field(&ctx, field));
    let result_ty = felt_ty.unwrap_or_else(|| FeltType::new(&ctx));
    let block = cast_function(&ctx, &module, &[], &[result_ty.into()]);
    let builder = OpBuilder::at_block_begin(&ctx, block);
    let input_ty = if boolean {
        IntegerType::new(&ctx, 1).into()
    } else {
        Type::index(&ctx)
    };
    let c = builder.insert(loc, |ctx, loc| {
        arith::constant(ctx, IntegerAttribute::new(input_ty, value).into(), loc)
    });
    let cast = tofelt_with_overflow(
        &builder,
        loc,
        c.result(0).unwrap().into(),
        felt_ty,
        semantics,
    );
    dialect::function::r#return(&builder, loc, &[cast.result(0).unwrap().into()]);

    canonicalize(&ctx, &mut module);
    if let Some(expected) = expected {
        assert_folded_return(block, FeltConstAttribute::new(&ctx, expected, field).into());
    } else {
        let cast = block.first_operation().unwrap().next_in_block().unwrap();
        assert!(is_tofelt_op(&cast));
        assert_eq!(overflow(cast), semantics);
    }
}

#[rstest]
#[case("-1", Some("babybear"), OverflowSemantics::Assert, Some(2013265920))]
#[case("2013265922", Some("babybear"), OverflowSemantics::Assert, Some(1))]
#[case(
    "9223372036854775808",
    Some("babybear"),
    OverflowSemantics::Assert,
    Some(1592717042)
)]
#[case(
    "9223372036854775807",
    Some("bn254"),
    OverflowSemantics::Assert,
    Some(i64::MAX)
)]
#[case("9223372036854775808", Some("bn254"), OverflowSemantics::Assert, None)]
#[case(
    "9223372036854775820",
    Some("bn254"),
    OverflowSemantics::Saturate,
    Some(i64::MAX)
)]
#[case(
    "9223372036854775820",
    Some("bn254"),
    OverflowSemantics::Wrap,
    Some(12)
)]
#[case(
    "9223372036854775820",
    Some("bn254"),
    OverflowSemantics::Truncate,
    Some(12)
)]
#[case("0", None, OverflowSemantics::Assert, None)]
#[case("0", None, OverflowSemantics::Saturate, None)]
#[case("0", None, OverflowSemantics::Wrap, None)]
#[case("0", None, OverflowSemantics::Truncate, None)]
fn toindex_constant_folding(
    #[case] value: &str,
    #[case] field: Option<&str>,
    #[case] semantics: OverflowSemantics,
    #[case] expected: Option<i64>,
) {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let mut module = LlzkModuleBuilder::create(loc, None);
    let block = cast_function(&ctx, &module, &[], &[Type::index(&ctx)]);
    let builder = OpBuilder::at_block_begin(&ctx, block);
    let c = dialect::felt::constant(&builder, loc, FeltConstAttribute::parse(&ctx, value, field))
        .unwrap();
    let cast = toindex_with_overflow(&builder, loc, c.result(0).unwrap().into(), semantics);
    dialect::function::r#return(&builder, loc, &[cast.result(0).unwrap().into()]);

    canonicalize(&ctx, &mut module);
    if let Some(expected) = expected {
        assert_folded_return(
            block,
            IntegerAttribute::new(Type::index(&ctx), expected).into(),
        );
    } else {
        let cast = block.first_operation().unwrap().next_in_block().unwrap();
        assert!(is_toindex_op(&cast));
        assert_eq!(overflow(cast), semantics);
    }
}

#[rstest]
#[case("index", true)]
#[case("i1", false)]
#[case("!felt.type", true)]
#[case("!felt.type<\"babybear\">", false)]
#[case("!felt.type<\"bn254\">", true)]
fn unused_cast_effects(
    #[case] input_ty: &str,
    #[case] assert_can_fail: bool,
    #[values(
        None,
        Some(OverflowSemantics::Assert),
        Some(OverflowSemantics::Saturate),
        Some(OverflowSemantics::Wrap),
        Some(OverflowSemantics::Truncate)
    )]
    semantics: Option<OverflowSemantics>,
) {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let mut module = LlzkModuleBuilder::create(loc, None);
    let input_ty = Type::parse(&ctx, input_ty).unwrap();
    let block = cast_function(&ctx, &module, &[input_ty], &[]);
    let builder = OpBuilder::at_block_begin(&ctx, block);
    let arg = block.argument(0).unwrap().into();
    let cast = match (FeltType::try_from(input_ty).is_ok(), semantics) {
        (true, Some(semantics)) => toindex_with_overflow(&builder, loc, arg, semantics),
        (true, None) => toindex(&builder, loc, arg),
        (false, Some(semantics)) => tofelt_with_overflow(&builder, loc, arg, None, semantics),
        (false, None) => tofelt(&builder, loc, arg, None),
    };
    let actual_semantics = overflow(cast);
    dialect::function::r#return(&builder, loc, &[]);

    canonicalize(&ctx, &mut module);
    let first = block.first_operation().unwrap();
    let should_remain = assert_can_fail && actual_semantics == OverflowSemantics::Assert;
    assert_eq!(is_tofelt_op(&first) || is_toindex_op(&first), should_remain);
    if !should_remain {
        assert_eq!(
            first.name().as_string_ref().as_str().unwrap(),
            "function.return"
        );
    }
}
