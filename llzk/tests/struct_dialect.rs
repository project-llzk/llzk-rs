#![allow(unused_crate_dependencies)]
//! Integration tests for the struct dialect.

use llzk::{
    builder::{OpBuilder, OpBuilderLike},
    prelude::*,
};

mod common;

fn default_funcs<'c>(
    builder: &impl OpBuilderLike<'c>,
    loc: Location<'c>,
    typ: StructType<'c>,
) -> Result<(), LlzkError> {
    dialect::r#struct::helpers::compute_fn(builder, loc, typ, &[], None)?;
    dialect::r#struct::helpers::constrain_fn(builder, loc, typ, &[], None)?;
    Ok(())
}

fn product_only_funcs<'c>(
    builder: &impl OpBuilderLike<'c>,
    loc: Location<'c>,
    typ: StructType<'c>,
) -> Result<(), LlzkError> {
    dialect::r#struct::helpers::product_fn(builder, loc, typ, &[], None)?;
    Ok(())
}

#[test]
fn struct_type_with_flat_name() {
    common::setup();
    let name = "flat";
    let context = LlzkContext::new();
    let typ = StructType::from_str(&context, name);
    assert_eq!(typ.name().to_string(), format!("@{}", name));
}

#[test]
fn struct_type_with_non_flat_name() {
    common::setup();
    let context = LlzkContext::new();
    let a = SymbolRefAttribute::new_from_str(&context, "root", &["a", "b"]);
    let typ = StructType::new(a, &[]);
    assert_eq!(typ.name(), a);
}

#[test]
fn empty_struct() {
    common::setup();
    let name = "empty";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str(&context, name);
    assert_eq!(typ.name().to_string(), format!("@{}", name));

    let builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&builder, loc, name, |builder| {
        default_funcs(builder, loc, typ)
    })
    .unwrap();

    assert_test!(s, module, @file "expected/empty_struct.mlir" );
}

#[test]
fn struct_with_one_member() {
    common::setup();
    let name = "one_member";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);
    assert_eq!(typ.name().to_string(), format!("@{}", name));

    let builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&builder, loc, name, |builder| {
        dialect::r#struct::member(
            builder,
            loc,
            "foo",
            Type::index(&context),
            false,
            false,
            false,
        )?;
        default_funcs(builder, loc, typ)
    })
    .unwrap();
    assert!(s.find_member_def("foo").is_some());
    assert_eq!(s.member_defs().len(), 1);

    assert_test!(s, module, @file "expected/struct_with_one_member.mlir");
}

#[test]
fn signal_column_and_public_member() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let member = dialect::r#struct::member(
        &builder,
        loc,
        "foo",
        FeltType::new(&context),
        true,
        true,
        true,
    )
    .unwrap();

    assert!(member.signal());
    assert!(member.column());
    assert!(member.has_public_attr());
    assert!(member.to_string().contains("column"));
    assert!(member.to_string().contains("signal"));
    assert!(member.to_string().contains("llzk.pub"));

    member.set_signal(false);
    member.set_column(false);
    assert!(!member.signal());
    assert!(!member.column());

    member.set_signal(true);
    member.set_column(true);
    assert!(member.signal());
    assert!(member.column());
}

#[test]
fn empty_struct_with_pub_inputs() {
    common::setup();
    let name = "empty";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);
    assert_eq!(typ.name().to_string(), format!("@{}", name));

    let inputs = vec![(FeltType::new(&context).into(), Location::unknown(&context))];
    let arg_attrs = vec![vec![PublicAttribute::new_named_attr(&context)]];
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&builder, loc, name, |builder| {
        dialect::r#struct::helpers::compute_fn(
            builder,
            loc,
            typ,
            inputs.as_slice(),
            Some(arg_attrs.as_slice()),
        )?;
        dialect::r#struct::helpers::constrain_fn(
            builder,
            loc,
            typ,
            inputs.as_slice(),
            Some(arg_attrs.as_slice()),
        )?;
        Ok(())
    })
    .unwrap();

    assert_test!(s, module, @file "expected/empty_struct_with_pub_inputs.mlir");
}

#[test]
fn struct_readm() {
    common::setup();
    let name = "read_member";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);

    let module_builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&module_builder, loc, name, |builder| {
        dialect::r#struct::member(
            builder,
            loc,
            "foo",
            Type::index(&context),
            false,
            false,
            false,
        )?;
        default_funcs(builder, loc, typ)
    })
    .unwrap();

    let constrain_body = s
        .constrain_func()
        .expect("failed to get constrain function")
        .body()
        .expect("failed to get body region")
        .first_block()
        .expect("failed to get first block");

    let self_value: Value = constrain_body.argument(0).unwrap().into();
    let builder = OpBuilder::new(
        &context,
        llzk::builder::EntryPoint::Before(constrain_body.terminator().unwrap()),
    );
    let readm_op =
        dialect::r#struct::readm(&builder, loc, Type::index(&context), self_value, "foo").unwrap();

    assert_test!(readm_op, module, @file "expected/read_member.mlir");
}

#[test]
fn struct_readm_with_literal_offset() {
    common::setup();
    let name = "read_member_offset";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);

    let module_builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&module_builder, loc, name, |builder| {
        dialect::r#struct::member(
            builder,
            loc,
            "foo",
            Type::index(&context),
            false,
            true,
            false,
        )?;
        default_funcs(builder, loc, typ)
    })
    .unwrap();

    let constrain_body = s
        .constrain_func()
        .expect("failed to get constrain function")
        .body()
        .expect("failed to get body region")
        .first_block()
        .expect("failed to get first block");

    let self_value: Value = constrain_body.argument(0).unwrap().into();
    let builder = OpBuilder::new(
        &context,
        llzk::builder::EntryPoint::Before(constrain_body.terminator().unwrap()),
    );
    let readm_op = dialect::r#struct::readm_with_offset(
        &builder,
        loc,
        Type::index(&context),
        self_value,
        "foo",
        1,
    )
    .unwrap();

    assert!(dialect::r#struct::is_struct_readm(&readm_op));
    assert!(readm_op.verify());
}

#[test]
fn product_only_struct_product_func() {
    common::setup();
    let name = "product_only";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);

    let builder = OpBuilder::at_block_begin(&context, module.body());
    let s = dialect::r#struct::def(&builder, loc, name, |builder| {
        product_only_funcs(builder, loc, typ)
    })
    .unwrap();

    assert!(s.compute_func().is_none());
    assert!(s.constrain_func().is_none());
    let product = s.product_func().expect("failed to get product function");
    assert!(product.name_is_product());
    assert!(product.is_struct_product());
}
