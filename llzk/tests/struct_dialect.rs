#![allow(unused_crate_dependencies)]
//! Integration tests for the struct dialect.

use llzk::{builder::OpBuilder, prelude::*};

mod common;

fn default_funcs<'c>(
    loc: Location<'c>,
    typ: StructType<'c>,
) -> [Result<Operation<'c>, LlzkError>; 2] {
    [
        dialect::r#struct::helpers::compute_fn(loc, typ, &[], None).map(Into::into),
        dialect::r#struct::helpers::constrain_fn(loc, typ, &[], None).map(Into::into),
    ]
}

fn product_only_funcs<'c>(
    loc: Location<'c>,
    typ: StructType<'c>,
) -> [Result<Operation<'c>, LlzkError>; 1] {
    [dialect::r#struct::helpers::product_fn(loc, typ, &[], None).map(Into::into)]
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

    let s = dialect::r#struct::def(loc, name, default_funcs(loc, typ)).unwrap();
    let s = module.body().append_operation(s.into());

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

    let mut region_ops = vec![
        dialect::r#struct::member(loc, "foo", Type::index(&context), false, false, false)
            .map(Into::into),
    ];
    region_ops.extend(default_funcs(loc, typ));

    let s = dialect::r#struct::def(loc, name, region_ops).unwrap();
    assert!(s.find_member_def("foo").is_some());
    assert_eq!(s.member_defs().len(), 1);
    let s = module.body().append_operation(s.into());

    assert_test!(s, module, @file "expected/struct_with_one_member.mlir");
}

#[test]
fn signal_column_and_public_member() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let member =
        dialect::r#struct::member(loc, "foo", FeltType::new(&context), true, true, true).unwrap();

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
    let s = dialect::r#struct::def(loc, name, {
        [
            dialect::r#struct::helpers::compute_fn(
                loc,
                typ,
                inputs.as_slice(),
                Some(arg_attrs.as_slice()),
            )
            .map(Into::into),
            dialect::r#struct::helpers::constrain_fn(
                loc,
                typ,
                inputs.as_slice(),
                Some(arg_attrs.as_slice()),
            )
            .map(Into::into),
        ]
    })
    .unwrap();
    let s = module.body().append_operation(s.into());

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

    let mut region_ops = vec![
        dialect::r#struct::member(loc, "foo", Type::index(&context), false, false, false)
            .map(Into::into),
    ];
    region_ops.extend(default_funcs(loc, typ));

    let s = dialect::r#struct::def(loc, name, region_ops).unwrap();
    let s = StructDefOpRef::try_from(module.body().append_operation(s.into())).unwrap();

    let constrain_body = s
        .constrain_func()
        .expect("failed to get constrain function")
        .region(0)
        .expect("failed to get first region")
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
fn product_only_struct_product_func() {
    common::setup();
    let name = "product_only";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);

    let s = dialect::r#struct::def(loc, name, product_only_funcs(loc, typ)).unwrap();
    let s = StructDefOpRef::try_from(module.body().append_operation(s.into())).unwrap();

    assert!(s.compute_func().is_none());
    assert!(s.constrain_func().is_none());
    let product = s.product_func().expect("failed to get product function");
    assert!(product.name_is_product());
    assert!(product.is_struct_product());
}
