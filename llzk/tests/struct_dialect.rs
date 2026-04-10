use llzk::prelude::*;
use melior::ir::Location;

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
    let a = SymbolRefAttribute::new(&context, "root", &["a", "b"]);
    let typ = StructType::new(a, &[]);
    assert_eq!(typ.name(), a);
}

#[test]
fn empty_struct() {
    common::setup();
    let name = "empty";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
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
    let module = llzk_module(Location::unknown(&context));
    let loc = Location::unknown(&context);
    let typ = StructType::from_str_params(&context, name, &[]);
    assert_eq!(typ.name().to_string(), format!("@{}", name));

    let mut region_ops = vec![
        dialect::r#struct::member(loc, "foo", Type::index(&context), false, false).map(Into::into),
    ];
    region_ops.extend(default_funcs(loc, typ));

    let s = dialect::r#struct::def(loc, name, region_ops).unwrap();
    assert!(s.get_member_def("foo").is_some());
    assert_eq!(s.get_member_defs().len(), 1);
    let s = module.body().append_operation(s.into());

    assert_test!(s, module, @file "expected/struct_with_one_member.mlir");
}

#[test]
fn empty_struct_with_pub_inputs() {
    common::setup();
    let name = "empty";
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
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
