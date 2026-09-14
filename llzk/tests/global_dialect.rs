#![allow(unused_crate_dependencies)]
//! Integration tests for the global dialect.

use llzk::{
    builder::{OpBuilder, OpBuilderLike as _},
    dialect::global::{def, is_def_op, is_read_op, is_write_op, read, write},
    prelude::*,
};

mod common;

#[test]
fn create_global_def_with_initializer() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = LlzkModuleBuilder::create(location, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let felt_type: Type = FeltType::new(&context).into();

    let global = def(
        &builder,
        location,
        "constant",
        felt_type,
        true,
        Some(FeltConstAttribute::new(&context, 42, None).into()),
    );

    assert!(global.verify());
    assert!(is_def_op(&global));
    assert_eq!(
        format!("{global}"),
        "global.def const @constant : !felt.type =  42"
    );
}

#[test]
fn create_uninitialized_mutable_global() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = LlzkModuleBuilder::create(location, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let global = def(
        &builder,
        location,
        "mutable",
        Type::index(&context),
        false,
        None,
    );

    assert!(global.verify());
    assert!(is_def_op(&global));
    assert_eq!(format!("{global}"), "global.def @mutable : index");
}

#[test]
fn create_global_reads_and_write() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = LlzkModuleBuilder::create(location, None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let felt_type: Type = FeltType::new(&context).into();

    let constant = def(
        &builder,
        location,
        "constant",
        felt_type,
        true,
        Some(FeltConstAttribute::new(&context, 1, None).into()),
    );
    let mutable = def(
        &builder,
        location,
        "mutable",
        felt_type,
        false,
        Some(FeltConstAttribute::new(&context, 2, None).into()),
    );
    let function = dialect::function::def(
        &builder,
        location,
        "access",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
        llzk::dialect::empty_region,
    )
    .unwrap();
    function.set_allow_witness_attr(true);

    let block = function
        .body()
        .expect("function.def must have a body region")
        .first_block()
        .expect("function.def must have an entry block");
    builder.set_insertion_point_at_start(block);
    let constant_read = read(
        &builder,
        location,
        SymbolRefAttribute::new_from_str(&context, "constant", &[]),
        true,
        felt_type,
    );
    let mutable_read = read(
        &builder,
        location,
        SymbolRefAttribute::new_from_str(&context, "mutable", &[]),
        false,
        felt_type,
    );
    let global_write = write(
        &builder,
        location,
        SymbolRefAttribute::new_from_str(&context, "mutable", &[]),
        block
            .argument(0)
            .expect("function argument must exist")
            .into(),
    );
    dialect::function::r#return(
        &builder,
        location,
        &[mutable_read
            .result(0)
            .expect("read must produce a result")
            .into()],
    );

    assert!(constant.verify());
    assert!(mutable.verify());
    assert!(function.verify());
    assert!(is_read_op(&constant_read));
    assert!(is_read_op(&mutable_read));
    assert!(is_write_op(&global_write));
    assert_eq!(
        format!("{constant_read}"),
        "%0 = global.read const @constant : !felt.type"
    );
    assert_eq!(
        format!("{mutable_read}"),
        "%1 = global.read @mutable : !felt.type"
    );
    assert_eq!(
        format!("{global_write}"),
        "global.write @mutable = %arg0 : !felt.type"
    );
}
