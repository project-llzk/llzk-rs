use llzk::{
    builder::OpBuilder,
    dialect::poly::{
        TemplateExprOpLike, TemplateOpLike, TemplateParamOpLike, TemplateSymbolBindingOp,
        TemplateSymbolBindingOpRef, applymap, expr, is_applymap_op, is_expr_op, is_param_op,
        is_template_op, is_unifiable_cast_op, is_yield_op, param, template, unifiable_cast,
        r#yield,
    },
    prelude::*,
};
use melior::{dialect::arith, ir::Location};
use rstest::rstest;

mod common;

#[test]
fn get_type() {
    common::setup();
    let context = LlzkContext::new();
    let t = TVarType::new(&context, StringRef::new("A"));

    let ir = format!("{t}");
    let expected = "!poly.tvar<@A>";
    assert_eq!(ir, expected);
}

#[test]
fn get_type_name_ref() {
    common::setup();
    let context = LlzkContext::new();
    let t = TVarType::new(&context, StringRef::new("A"));

    let ir = format!("{:?}", t.name().as_str().unwrap());
    let expected = "\"A\"";
    assert_eq!(ir, expected);
}

#[test]
fn create_read_const() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::poly::read_const(loc, "A", FeltType::new(&context).into());

    let ir = format!("{op}");
    let expected = "%0 = poly.read_const @A : !felt.type\n";
    assert_eq!(ir, expected);
    assert!(op.verify());
}

#[test]
fn is_read_const() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::poly::read_const(loc, "C", IntegerType::new(&context, 64).into());

    let op_ref = unsafe { OperationRef::from_raw(op.to_raw()) };
    assert!(dialect::poly::is_read_const_op(&op_ref));
}

#[test]
fn create_param() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc);
    let op = param(
        loc,
        "T",
        Some(TVarType::new(&context, StringRef::new("T")).into()),
    )
    .unwrap();

    let ir = format!("{op}");
    assert!(ir.contains("\"poly.param\""));
    assert!(ir.contains("sym_name = \"T\""));
    assert!(ir.contains("type_opt = !poly.tvar<@T>"));
    assert!(op.type_opt().is_some());
    assert!(is_param_op(&op));

    let tmpl = template(loc, "tmpl", [Ok(op.into())]).unwrap();
    let tmpl = module.body().append_operation(tmpl.into());
    assert!(tmpl.verify());
}

#[test]
fn create_template_with_param_and_expr() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
    let loc = Location::unknown(&context);
    let c1 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 1).into(),
        loc,
    );
    let c1_res = c1.result(0).unwrap();

    let tmpl = template(
        loc,
        "tmpl",
        [
            param(loc, "T", None).map(Into::into),
            expr(
                loc,
                "N",
                [Ok(c1), r#yield(loc, c1_res.into()).map(Into::into)],
            )
            .map(Into::into),
        ],
    )
    .unwrap();

    assert!(tmpl.has_const_param_ops());
    assert!(tmpl.has_const_expr_ops());
    assert!(tmpl.has_const_param_named("T"));
    assert!(tmpl.has_const_expr_named("N"));
    assert_eq!(tmpl.const_param_names().len(), 1);
    assert_eq!(tmpl.const_expr_names().len(), 1);
    assert!(is_template_op(&tmpl));

    let tmpl = module.body().append_operation(tmpl.into());
    let ir = format!("{}", module.as_operation());
    let expected = r#"module attributes {llzk.lang} {
  poly.template @tmpl {
    poly.param @T
    poly.expr @N {
      %c1 = arith.constant 1 : index
      poly.yield %c1 : index
    }
  }
}
"#;
    assert_eq!(ir, expected);
    assert!(tmpl.verify());
}

#[test]
fn template_const_ops() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let c1 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 1).into(),
        loc,
    );
    let c1_res = c1.result(0).unwrap();

    let tmpl = template(
        loc,
        "tmpl",
        [
            param(
                loc,
                "T",
                Some(TVarType::new(&context, StringRef::new("T")).into()),
            )
            .map(Into::into),
            expr(
                loc,
                "N",
                [Ok(c1), r#yield(loc, c1_res.into()).map(Into::into)],
            )
            .map(Into::into),
            param(loc, "U", None).map(Into::into),
        ],
    )
    .unwrap();

    let ops = tmpl.const_binding_ops();
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0], TemplateSymbolBindingOpRef::Param(_)));
    assert!(matches!(ops[1], TemplateSymbolBindingOpRef::Expr(_)));
    assert!(matches!(ops[2], TemplateSymbolBindingOpRef::Param(_)));
    assert_eq!(
        ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
        ["T", "N", "U"]
    );
    assert_eq!(
        ops[0].type_opt().map(|ty| ty.to_string()),
        Some(String::from("!poly.tvar<@T>"))
    );
    assert_eq!(
        ops[1].type_opt().map(|ty| ty.to_string()),
        Some(String::from("index"))
    );
    assert!(ops[2].type_opt().is_none());
}

#[test]
fn empty_struct_with_one_param() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
    let loc = Location::unknown(&context);
    let typ = StructType::new(
        SymbolRefAttribute::new_from_str(&context, "tmpl", &["empty"]),
        &[FlatSymbolRefAttribute::new(&context, "T").into()],
    );

    let s = dialect::r#struct::def(
        loc,
        "empty",
        [
            dialect::r#struct::helpers::compute_fn(loc, typ, &[], None).map(Into::into),
            dialect::r#struct::helpers::constrain_fn(loc, typ, &[], None).map(Into::into),
        ],
    )
    .unwrap();

    let tmpl = template(
        loc,
        "tmpl",
        [param(loc, "T", None).map(Into::into), Ok(s.into())],
    )
    .unwrap();
    let tmpl = module.body().append_operation(tmpl.into());

    assert_test!(tmpl, module, @file "expected/empty_struct_with_one_param.mlir");
}

#[test]
fn create_expr_and_get_type() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
    let loc = Location::unknown(&context);
    let c2 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 2).into(),
        loc,
    );
    let c2_res = c2.result(0).unwrap();

    let op = expr(
        loc,
        "Two",
        [Ok(c2), r#yield(loc, c2_res.into()).map(Into::into)],
    )
    .unwrap();

    assert!(is_expr_op(&op));
    assert_eq!(format!("{}", op.expr_type()), "index");
    assert_eq!(
        op.initializer_region()
            .first_block()
            .unwrap()
            .argument_count(),
        0
    );

    let tmpl = template(loc, "tmpl", [Ok(op.into())]).unwrap();
    let tmpl = module.body().append_operation(tmpl.into());
    assert!(tmpl.verify());
}

#[test]
fn create_yield() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = Module::new(loc);
    let block = module.body();
    let c3 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 3).into(),
        loc,
    );
    let c3 = block.append_operation(c3);
    let y = r#yield(loc, c3.result(0).unwrap().into()).unwrap();
    let y = block.append_operation(y.into());

    assert!(is_yield_op(&y));
    let ir = format!("{block}");
    assert!(ir.contains("\"poly.yield\"(%0)"));
    assert!(ir.contains("value = 3 : index"));
}

fn create_index_constant<'c>(
    ctx: &'c Context,
    block: &Block<'c>,
    location: Location<'c>,
    i: i64,
) -> Value<'c, 'c> {
    let int_attr = IntegerAttribute::new(Type::index(ctx), i);
    let op = arith::constant(ctx, int_attr.into(), location);
    let op_ref = block.append_operation(op);
    assert_eq!(1, op_ref.result_count());
    op_ref.result(0).unwrap().into()
}

#[rstest]
#[case("affine_map<()[] -> (2)>", &[],
r"^bb0:
  %0 = poly.applymap () affine_map<() -> (2)>
")]
#[case("affine_map<(i)[] -> (i)>", &[1],
r"^bb0:
  %c1 = arith.constant 1 : index
  %0 = poly.applymap (%c1) affine_map<(d0) -> (d0)>
")]
#[case("affine_map<()[s0, s1] -> (s0 + s1)>", &[7, 9],
r"^bb0:
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %0 = poly.applymap ()[%c7, %c9] affine_map<()[s0, s1] -> (s0 + s1)>
")]
#[case("affine_map<(i, j) -> (i + j)>", &[2, 4],
r"^bb0:
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %0 = poly.applymap (%c2, %c4) affine_map<(d0, d1) -> (d0 + d1)>
")]
fn create_applymap(#[case] affine_map: &str, #[case] ops: &[i64], #[case] expected: &str) {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);

    let affine_map =
        Attribute::parse(&context, affine_map).expect("could not parse affine_map attribute");
    let module = Module::new(location);
    let block = module.body();
    let operands = ops
        .iter()
        .map(|i| create_index_constant(&context, &block, location, *i))
        .collect::<Vec<_>>();

    let applymap_op = applymap(location, affine_map, &operands);
    assert!(applymap_op.verify(), "op {applymap_op} failed to verify");
    assert!(is_applymap_op(&applymap_op));
    block.append_operation(applymap_op);
    let ir = format!("{block}");
    assert_eq!(ir, expected);
}

#[test]
fn create_unifiable_cast() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = Module::new(location);
    let block = module.body();

    let affine_map_str = "affine_map<()[s0, s1] -> (s0 + s1)>";
    let affine_map =
        Attribute::parse(&context, affine_map_str).expect("could not parse affine_map attribute");
    let array_ty = ArrayType::new(
        FeltType::new(&context).into(),
        &[FlatSymbolRefAttribute::new(&context, "N").into()],
    );
    let array_op = dialect::array::new(
        &OpBuilder::new(&context),
        location,
        array_ty,
        llzk::dialect::array::ArrayCtor::Values(&[]),
    );
    let array_op = block.append_operation(array_op);

    let new_array_ty = ArrayType::new(FeltType::new(&context).into(), &[affine_map]);
    let cast = unifiable_cast(
        location,
        array_op.result(0).unwrap().into(),
        new_array_ty.into(),
    );
    let cast = block.append_operation(cast);
    assert!(cast.verify(), "op {cast} failed to verify");
    assert!(is_unifiable_cast_op(&cast));

    let expected = r#"^bb0:
  %0 = "array.new"() <{mapOpGroupSizes = array<i32>, numDimsPerMap = array<i32>, operandSegmentSizes = array<i32: 0, 0>}> : () -> !array.type<@N x !felt.type>
  %1 = "poly.unifiable_cast"(%0) : (!array.type<@N x !felt.type>) -> !array.type<affine_map<()[s0, s1] -> (s0 + s1)> x !felt.type>
"#;
    let ir = format!("{block}");
    assert_eq!(ir, expected);
}

#[test]
fn owned_binding_op_param_name_and_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = TemplateSymbolBindingOp::Param(
        param(
            loc,
            "T",
            Some(TVarType::new(&context, StringRef::new("T")).into()),
        )
        .unwrap(),
    );

    assert!(matches!(op, TemplateSymbolBindingOp::Param(_)));
    assert_eq!(op.name(), "T");
    assert_eq!(
        op.type_opt().map(|ty| ty.to_string()),
        Some(String::from("!poly.tvar<@T>"))
    );
}

#[test]
fn owned_binding_op_expr_name_and_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let c1 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 1).into(),
        loc,
    );
    let c1_res = c1.result(0).unwrap();
    let op = TemplateSymbolBindingOp::Expr(
        expr(
            loc,
            "N",
            [Ok(c1), r#yield(loc, c1_res.into()).map(Into::into)],
        )
        .unwrap(),
    );

    assert!(matches!(op, TemplateSymbolBindingOp::Expr(_)));
    assert_eq!(op.name(), "N");
    assert_eq!(
        op.type_opt().map(|ty| ty.to_string()),
        Some(String::from("index"))
    );
}

#[test]
fn owned_binding_op_as_ref() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let c1 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 1).into(),
        loc,
    );
    let c1_res = c1.result(0).unwrap();

    let param_op = TemplateSymbolBindingOp::Param(param(loc, "T", None).unwrap());
    let expr_op = TemplateSymbolBindingOp::Expr(
        expr(
            loc,
            "N",
            [Ok(c1), r#yield(loc, c1_res.into()).map(Into::into)],
        )
        .unwrap(),
    );

    assert!(matches!(
        param_op.as_ref(),
        TemplateSymbolBindingOpRef::Param(_)
    ));
    assert!(matches!(
        expr_op.as_ref(),
        TemplateSymbolBindingOpRef::Expr(_)
    ));
    assert_eq!(param_op.as_ref().name(), param_op.name());
    assert_eq!(expr_op.as_ref().name(), expr_op.name());
}
