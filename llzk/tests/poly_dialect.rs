#![allow(unused_crate_dependencies)]
//! Integration tests for the poly dialect.

use llzk::{
    builder::{OpBuilder, OpBuilderLike as _},
    dialect::poly::{
        applymap, expr, is_applymap_op, is_expr_op, is_param_op, is_template_op,
        is_unifiable_cast_op, is_yield_op, param, template, unifiable_cast, r#yield,
    },
    prelude::*,
};
use melior::dialect::arith;
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
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let tmpl = template(&builder, loc, "tmpl", |builder| {
        let op = param(
            builder,
            loc,
            "T",
            Some(TVarType::new(&context, StringRef::new("T")).into()),
        )
        .unwrap();

        let ir = format!("{op}");
        assert_eq!(ir, "poly.param @T : !poly.tvar<@T>");
        assert!(op.type_opt().is_some());
        assert!(is_param_op(&op));

        Ok(())
    })
    .unwrap();
    assert!(tmpl.verify());
}

#[test]
fn create_template_with_param_and_expr() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let loc = Location::unknown(&context);

    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        expr(builder, loc, "N", |builder| {
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();

            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();

    assert!(tmpl.has_const_param_ops());
    assert!(tmpl.has_const_expr_ops());
    assert!(tmpl.has_const_param_named("T"));
    assert!(tmpl.has_const_expr_named("N"));
    assert_eq!(tmpl.const_param_names().len(), 1);
    assert_eq!(tmpl.const_expr_names().len(), 1);
    assert!(is_template_op(&tmpl));

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
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(
            builder,
            loc,
            "T",
            Some(TVarType::new(&context, StringRef::new("T")).into()),
        )?;

        expr(builder, loc, "N", |builder| {
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })?;

        param(builder, loc, "U", None)?;
        Ok(())
    })
    .unwrap();

    let ops = tmpl.const_binding_ops();
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0], TemplateSymbolBindingOpRef::Param(_)));
    assert!(matches!(ops[1], TemplateSymbolBindingOpRef::Expr(_)));
    assert!(matches!(ops[2], TemplateSymbolBindingOpRef::Param(_)));
    assert_eq!(
        ops.iter().map(|op| op.sym_name()).collect::<Vec<_>>(),
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
fn set_type_restriction_adds_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let op = param(&builder, loc, "T", None).unwrap();
    assert!(op.type_restriction().is_none());

    let ty = TVarType::new(&context, StringRef::new("T"));
    op.set_type_restriction(Some(ty.into()));
    assert_eq!(
        op.type_restriction().map(|t| t.to_string()),
        Some(String::from("!poly.tvar<@T>"))
    );
}

#[test]
fn set_type_restriction_clears_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let op = param(
        &builder,
        loc,
        "T",
        Some(TVarType::new(&context, StringRef::new("T")).into()),
    )
    .unwrap();
    assert!(op.type_restriction().is_some());

    op.set_type_restriction(None);
    assert!(op.type_restriction().is_none());
}

#[test]
fn empty_struct_with_one_param() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let loc = Location::unknown(&context);
    let typ = StructType::new(
        SymbolRefAttribute::new_from_str(&context, "tmpl", &["empty"]),
        &[FlatSymbolRefAttribute::new(&context, "T").into()],
    );

    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        builder.insert(loc, |_, loc| {
            dialect::r#struct::def(
                loc,
                "empty",
                [
                    dialect::r#struct::helpers::compute_fn(loc, typ, &[], None).map(Into::into),
                    dialect::r#struct::helpers::constrain_fn(loc, typ, &[], None).map(Into::into),
                ],
            )
            .unwrap()
            .into()
        });
        Ok(())
    })
    .unwrap();

    assert_test!(tmpl, module, @file "expected/empty_struct_with_one_param.mlir");
}

#[test]
fn create_expr_and_get_type() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let loc = Location::unknown(&context);
    let tmpl = template(&builder, loc, "tmpl", |builder| {
        let op = expr(builder, loc, "Two", |builder| {
            let c2_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 2).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c2_res.into())?;
            Ok(())
        })
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
        Ok(())
    })
    .unwrap();
    assert!(tmpl.verify());
}

#[test]
fn create_yield() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let block = module.body();
    let c3 = arith::constant(
        &context,
        IntegerAttribute::new(Type::index(&context), 3).into(),
        loc,
    );
    let c3 = block.append_operation(c3);
    let y = r#yield(&builder, loc, c3.result(0).unwrap().into()).unwrap();

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
    let module = llzk_module(location, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let block = module.body();
    let operands = ops
        .iter()
        .map(|i| create_index_constant(&context, &block, location, *i))
        .collect::<Vec<_>>();

    let applymap_op = applymap(&builder, location, affine_map, &operands);
    assert!(applymap_op.verify(), "op {applymap_op} failed to verify");
    assert!(is_applymap_op(&applymap_op));
    let ir = format!("{block}");
    assert_eq!(ir, expected);
}

#[test]
fn create_unifiable_cast() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = llzk_module(location, None);
    let block = module.body();

    let affine_map_str = "affine_map<()[s0, s1] -> (s0 + s1)>";
    let affine_map =
        Attribute::parse(&context, affine_map_str).expect("could not parse affine_map attribute");
    let array_ty = ArrayType::new(
        FeltType::new(&context).into(),
        &[FlatSymbolRefAttribute::new(&context, "N").into()],
    );
    let array_op = dialect::array::new(
        &OpBuilder::at_block_begin(&context, module.body()),
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
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let op = TemplateSymbolBindingOpRef::Param(
        param(
            &builder,
            loc,
            "T",
            Some(TVarType::new(&context, StringRef::new("T")).into()),
        )
        .unwrap(),
    );

    assert!(matches!(op, TemplateSymbolBindingOpRef::Param(_)));
    assert_eq!(op.sym_name(), "T");
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
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let op = TemplateSymbolBindingOpRef::Expr(
        expr(&builder, loc, "N", |builder| {
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })
        .unwrap(),
    );

    assert!(matches!(op, TemplateSymbolBindingOpRef::Expr(_)));
    assert_eq!(op.sym_name(), "N");
    assert_eq!(
        op.type_opt().map(|ty| ty.to_string()),
        Some(String::from("index"))
    );
}

// ── TemplateParamOpLike ──────────────────────────────────────────────────────

#[test]
fn param_type_restriction_some() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let op = param(
        &builder,
        loc,
        "T",
        Some(TVarType::new(&context, StringRef::new("T")).into()),
    )
    .unwrap();
    assert_eq!(
        op.type_restriction().map(|t| t.to_string()),
        Some(String::from("!poly.tvar<@T>"))
    );
}

#[test]
fn param_type_restriction_none() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let op = param(&builder, loc, "T", None).unwrap();
    assert!(op.type_restriction().is_none());
}

// ── TemplateSymbolBindingOpLike ──────────────────────────────────────────────

#[test]
fn sym_name_attr_value() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let op = param(&builder, loc, "MyParam", None).unwrap();
    // sym_name_attr() returns the underlying StringAttribute
    let attr = op.sym_name_attr();
    assert_eq!(attr.value(), "MyParam");
    // sym_name() must agree
    assert_eq!(op.sym_name(), attr.value());
}

// ── TemplateOpLike ───────────────────────────────────────────────────────────

#[test]
fn body_region_and_body() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        Ok(())
    })
    .unwrap();

    // body_region() contains exactly one block
    let region = tmpl.body_region();
    let first = region.first_block();
    assert!(first.is_some());
    assert!(first.unwrap().next_in_region().is_none());

    // body() returns the single block with no block arguments
    let block = tmpl.body();
    assert_eq!(block.argument_count(), 0);
    // The param op we inserted must be present
    assert!(block.first_operation().is_some());
}

#[test]
fn has_const_named_negative() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        Ok(())
    })
    .unwrap();

    assert!(!tmpl.has_const_param_named("NotHere"));
    // "T" is a param, not an expr
    assert!(!tmpl.has_const_expr_named("T"));
}

#[test]
fn has_const_ops_false() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    // Template with only params — has_const_expr_ops must be false
    let params_only = template(&builder, loc, "params_only", |builder| {
        param(builder, loc, "T", None)?;
        Ok(())
    })
    .unwrap();
    assert!(params_only.has_const_param_ops());
    assert!(!params_only.has_const_expr_ops());

    let exprs_only = template(&builder, loc, "exprs_only", |builder| {
        expr(builder, loc, "N", |builder| {
            // Template with only an expr — has_const_param_ops must be false
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();
    assert!(!exprs_only.has_const_param_ops());
    assert!(exprs_only.has_const_expr_ops());
}

#[test]
fn const_names_content() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        param(builder, loc, "U", None)?;
        expr(builder, loc, "N", |builder| {
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();

    let param_names: Vec<String> = tmpl
        .const_param_names()
        .into_iter()
        .map(|a| a.value().to_owned())
        .collect();
    assert_eq!(param_names, ["T", "U"]);

    let expr_names: Vec<String> = tmpl
        .const_expr_names()
        .into_iter()
        .map(|a| a.value().to_owned())
        .collect();
    assert_eq!(expr_names, ["N"]);
}

#[test]
fn const_binding_ops_empty_template() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let tmpl = template(&builder, loc, "empty", |_| Ok(())).unwrap();
    assert!(tmpl.const_binding_ops().is_empty());
    assert!(!tmpl.has_const_param_ops());
    assert!(!tmpl.has_const_expr_ops());
}

// ── Display impls ────────────────────────────────────────────────────────────

#[test]
fn display_template_symbol_binding_op() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let op_ref = TemplateSymbolBindingOpRef::Param(param(&builder, loc, "T", None).unwrap());
    let s_ref = format!("{}", op_ref);
    assert!(s_ref.contains("poly.param"));
    assert!(s_ref.contains("sym_name = \"T\""));
}

// ── From conversions ─────────────────────────────────────────────────────────

#[test]
fn from_conversions_for_binding_op_ref() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = llzk_module(loc, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let tmpl = template(&builder, loc, "tmpl", |builder| {
        param(builder, loc, "T", None)?;
        expr(builder, loc, "N", |builder| {
            let c1_res = builder
                .insert(loc, |context, loc| {
                    arith::constant(
                        context,
                        IntegerAttribute::new(Type::index(context), 1).into(),
                        loc,
                    )
                })
                .result(0)
                .unwrap();
            r#yield(builder, loc, c1_res.into())?;
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();

    let ops = tmpl.const_binding_ops();
    let TemplateSymbolBindingOpRef::Param(param_ref) = ops[0] else {
        panic!("expected param at index 0");
    };
    let TemplateSymbolBindingOpRef::Expr(expr_ref) = ops[1] else {
        panic!("expected expr at index 1");
    };

    // From<TemplateParamOpRef> for TemplateSymbolBindingOpRef
    let from_param: TemplateSymbolBindingOpRef = param_ref.into();
    assert!(matches!(from_param, TemplateSymbolBindingOpRef::Param(_)));

    // From<TemplateExprOpRef> for TemplateSymbolBindingOpRef
    let from_expr: TemplateSymbolBindingOpRef = expr_ref.into();
    assert!(matches!(from_expr, TemplateSymbolBindingOpRef::Expr(_)));

    // From<TemplateSymbolBindingOpRef> for OperationRef — pointer must be non-null
    let op_ref: OperationRef = from_param.into();
    assert!(!op_ref.to_raw().ptr.is_null());
}
