//! Integration tests for the function dialect.

use llzk::{
    attributes::array::ArrayAttribute,
    builder::{OpBuilder, OpBuilderLike},
    map_operands::MapOperandsBuilder,
    prelude::*,
};
use llzk_sys::{FUNCTION_ARG_NAME_ATTR_NAME, FUNCTION_RES_NAME_ATTR_NAME};
use melior::ir::Identifier;

mod common;

#[test]
fn empty_function() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let f = dialect::function::def(
        loc,
        "empty",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = Block::new(&[]);
        block.append_operation(dialect::function::r#return(loc, &[]));
        f.region(0)
            .expect("function.def must have at least 1 region")
            .append_block(block);
    }

    assert_eq!(f.region_count(), 1);
    let f = module.body().append_operation(f.into());
    assert!(f.verify());
    log::info!("Op passed verification");
    let ir = format!("{f}");
    let expected = r"function.def @empty() {
  function.return
}";
    assert_eq!(ir, expected);
}

#[test]
fn function_call() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let f = dialect::function::def(
        loc,
        "recursive",
        FunctionType::new(&context, &[], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = Block::new(&[]);
        let builder = OpBuilder::at_block_begin(&context, &block);
        // Build call to itself
        let name = FlatSymbolRefAttribute::new(&context, "recursive");
        let v = block
            .append_operation(
                dialect::function::call(&builder, loc, name, &[], &[felt_type])
                    .unwrap()
                    .into(),
            )
            .result(0)
            .map(Value::from)
            .unwrap();
        // Build return operation
        block.append_operation(dialect::function::r#return(loc, &[v]));
        // Add Block to function
        f.region(0)
            .expect("function.def must have at least 1 region")
            .append_block(block);
    }

    assert_eq!(f.region_count(), 1);
    let f = module.body().append_operation(f.into());
    assert_test!(f, module, @file "expected/function_call.mlir");
}

#[test]
fn function_call_with_map_operands() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let f = dialect::function::def(
        loc,
        "recursive",
        FunctionType::new(&context, &[], &[felt_type]),
        &[],
        None,
    )
    .unwrap();
    {
        let block = Block::new(&[]);
        let builder = OpBuilder::at_block_begin(&context, &block);
        // Build call to itself
        let name = FlatSymbolRefAttribute::new(&context, "recursive");
        let map_operands = MapOperandsBuilder::new();
        let v = block
            .append_operation(
                dialect::function::call_with_map_operands(
                    &builder,
                    loc,
                    name,
                    &[],
                    &[felt_type],
                    map_operands,
                )
                .unwrap()
                .into(),
            )
            .result(0)
            .map(Value::from)
            .unwrap();
        // Build return operation
        block.append_operation(dialect::function::r#return(loc, &[v]));
        // Add Block to function
        f.region(0)
            .expect("function.def must have at least 1 region")
            .append_block(block);
    }

    assert_eq!(f.region_count(), 1);
    let f = module.body().append_operation(f.into());
    assert_test!(f, module, @file "expected/function_call.mlir");
}

fn make_empty_struct<'c>(context: &'c LlzkContext, name: &str) -> StructDefOp<'c> {
    let loc = Location::unknown(context);
    let typ = StructType::from_str(context, name);
    dialect::r#struct::def(loc, name, {
        [
            dialect::r#struct::helpers::compute_fn(loc, typ, &[], None).map(Into::into),
            dialect::r#struct::helpers::constrain_fn(loc, typ, &[], None).map(Into::into),
        ]
    })
    .unwrap()
}

fn make_product_struct<'c>(context: &'c LlzkContext, name: &str) -> StructDefOp<'c> {
    let loc = Location::unknown(context);
    let typ = StructType::from_str(context, name);
    dialect::r#struct::def(loc, name, {
        [dialect::r#struct::helpers::product_fn(loc, typ, &[], None).map(Into::into)]
    })
    .unwrap()
}

#[test]
fn func_def_op_self_value_of_compute() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let module_body = module.body();

    let s = make_empty_struct(&context, "StructA");
    let s = StructDefOpRef::try_from(module_body.append_operation(s.into())).unwrap();
    llzk::operation::verify_operation_with_diags(&s).expect("verification failed");
    log::info!("Struct passed verification");

    let compute_fn = s.compute_func().expect("failed to get compute function");
    let self_val = compute_fn.self_value_of_compute().unwrap();
    // Get the expected value. The first operation in the compute function is
    // the CreateStructOp, whose first result is the self value.
    let expected = compute_fn
        .region(0)
        .expect("failed to get first region")
        .first_block()
        .expect("failed to get first block")
        .first_operation()
        .expect("failed to get first operation")
        .result(0)
        .expect("failed to get first result");

    similar_asserts::assert_eq!(self_val, expected.into());
}

#[test]
fn func_def_op_self_value_of_constrain() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let module_body = module.body();

    let s = make_empty_struct(&context, "StructA");
    let s = StructDefOpRef::try_from(module_body.append_operation(s.into())).unwrap();
    llzk::operation::verify_operation_with_diags(&s).expect("verification failed");
    log::info!("Struct passed verification");

    let constrain_fn = s
        .constrain_func()
        .expect("failed to get constrain function");
    let self_val = constrain_fn.self_value_of_constrain().unwrap();
    // Get the expected value. The first argument of the function is the self value.
    let expected = constrain_fn
        .argument(0)
        .expect("failed to get first argument");

    similar_asserts::assert_eq!(self_val, expected.into());
}

#[test]
fn call_op_self_value_of_compute() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let module_body = module.body();

    let s1 = make_empty_struct(&context, "StructA");
    let s1 = StructDefOpRef::try_from(module_body.append_operation(s1.into())).unwrap();
    assert!(s1.verify());
    log::info!("Struct 1 passed verification");

    let s2 = make_empty_struct(&context, "StructB");
    let s2 = StructDefOpRef::try_from(module_body.append_operation(s2.into())).unwrap();
    assert!(s2.verify());
    log::info!("Struct 2 passed verification");

    let s2_compute_body = s2
        .compute_func()
        .expect("failed to get compute function")
        .region(0)
        .expect("failed to get first region")
        .first_block()
        .expect("failed to get first block");
    let builder = OpBuilder::at_block_end(&context, s2_compute_body);
    let loc = Location::unknown(&context);
    let call = builder.insert(loc, |_, loc| {
        let name = SymbolRefAttribute::new_from_str(&context, "StructA", &["compute"]);
        dialect::function::call(&builder, loc, name, &[], &[s1.r#type()])
            .unwrap()
            .into()
    });

    assert_test!(module.as_operation(), module, @file "expected/call_op_self_value_of_compute.mlir" );

    // Now actually test the `self_value_of_compute` function
    let call = CallOpRef::try_from(call).unwrap();
    let self_val = call.self_value_of_compute();
    similar_asserts::assert_eq!(
        format!("{}", self_val.unwrap()),
        // Yes, the line does have a trailing space, here and in the entire IR above.
        "%0 = function.call @StructA::@compute() : () -> !struct.type<@StructA> "
    );
}

#[test]
fn call_op_product_classifiers() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let module_body = module.body();

    let s1 = make_product_struct(&context, "StructProdA");
    let s1 = StructDefOpRef::try_from(module_body.append_operation(s1.into())).unwrap();
    assert!(s1.verify());

    let s2 = make_product_struct(&context, "StructProdB");
    let s2 = StructDefOpRef::try_from(module_body.append_operation(s2.into())).unwrap();
    assert!(s2.verify());

    let product_body = s2
        .product_func()
        .expect("failed to get product function")
        .region(0)
        .expect("failed to get first region")
        .first_block()
        .expect("failed to get first block");
    let builder = OpBuilder::at_block_end(&context, product_body);
    let loc = Location::unknown(&context);
    let call = builder.insert(loc, |_, loc| {
        let name = SymbolRefAttribute::new_from_str(&context, "StructProdA", &["product"]);
        dialect::function::call(&builder, loc, name, &[], &[s1.r#type()])
            .unwrap()
            .into()
    });

    let call = CallOpRef::try_from(call).unwrap();
    assert!(call.callee_is_product());
    assert!(call.callee_is_struct_product());
    assert!(!call.callee_is_compute());
    assert!(!call.callee_is_constrain());
}

#[test]
fn func_def_op_ref_from_borrow_equals_original() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);

    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();

    // Convert a shared borrow into a FuncDefOpRef
    let op_ref = FuncDefOpRef::from(&op);

    // The ref must point to the same underlying operation.
    assert_eq!(op_ref, op);
}

#[test]
fn func_def_op_ref_from_borrow_does_not_drop_original() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);

    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();

    {
        let op_ref = FuncDefOpRef::from(&op);
        // Use the ref so it isn't optimized away.
        assert_eq!(op_ref.region_count(), 1);
    } // `op_ref` drops here — `op` must still be alive.

    // `op` is still valid: its Drop will run mlirOperationDestroy exactly once.
    assert_eq!(op.region_count(), 1);
}

// Tests for FuncDefOpLike methods added in e2157c6

#[test]
fn func_def_op_get_function_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[felt_type], &[]),
        &[],
        None,
    )
    .unwrap();
    let result = op.function_type().unwrap();
    assert_eq!(result.input_count(), 1);
    assert_eq!(result.result_count(), 0);
}

#[test]
fn func_def_op_set_function_type() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(op.function_type().unwrap().input_count(), 0);
    op.set_function_type(FunctionType::new(&context, &[felt_type], &[]));
    assert_eq!(op.function_type().unwrap().input_count(), 1);
}

#[test]
fn func_def_op_get_sym_name() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(op.sym_name().unwrap().value(), "my_func");
}

#[test]
fn func_def_op_set_sym_name() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(op.sym_name().unwrap().value(), "my_func");
    op.set_sym_name(StringAttribute::new(&context, "new_name"));
    assert_eq!(op.sym_name().unwrap().value(), "new_name");
}

#[test]
fn func_def_op_is_declaration() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    // A freshly created FuncDefOp has no body blocks — it is a declaration.
    assert!(op.is_declaration());

    // After appending a block, it is no longer a declaration.
    let block = Block::new(&[]);
    block.append_operation(dialect::function::r#return(loc, &[]));
    op.region(0)
        .expect("function.def must have at least 1 region")
        .append_block(block);
    assert!(!op.is_declaration());
}

#[test]
fn func_def_op_get_body() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let op = dialect::function::def(
        loc,
        "my_func",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    // A freshly created FuncDefOp already has a (empty) region; get_body returns Ok.
    assert!(op.body().is_ok());

    // After appending a block, get_body still succeeds.
    let block = Block::new(&[]);
    block.append_operation(dialect::function::r#return(loc, &[]));
    op.region(0)
        .expect("function.def must have at least 1 region")
        .append_block(block);
    assert!(op.body().is_ok());
}

#[test]
fn func_def_op_arg_name_round_trip() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "named_arg",
        FunctionType::new(&context, &[felt_type], &[]),
        &[],
        None,
    )
    .unwrap();

    assert!(op.arg_name_attr(0).unwrap().is_none());
    assert!(!op.has_arg_name(0));
    op.set_arg_name(0, "input").unwrap();
    assert!(op.has_arg_name(0));
    assert_eq!(op.arg_name_attr(0).unwrap().unwrap().value(), "input");
    assert_eq!(op.arg_name(0).unwrap(), Some("input".to_string()));
}

#[test]
fn func_def_op_res_name_round_trip() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "named_res",
        FunctionType::new(&context, &[], &[felt_type]),
        &[],
        None,
    )
    .unwrap();

    assert!(op.res_name_attr(0).unwrap().is_none());
    assert!(!op.has_res_name(0));
    op.set_res_name(0, "output").unwrap();
    assert!(op.has_res_name(0));
    assert_eq!(op.res_name_attr(0).unwrap().unwrap().value(), "output");
    assert_eq!(op.res_name(0).unwrap(), Some("output".to_string()));
}

#[test]
fn func_def_op_def_with_signature_attrs_prints_named_arg_and_result() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def_with_signature_attrs(
        loc,
        "named_signature",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        Some(&[vec![dialect::function::arg_name_attr(&context, "input")]]),
        Some(&[vec![dialect::function::res_name_attr(&context, "output")]]),
    )
    .unwrap();

    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    block.append_operation(dialect::function::r#return(loc, &[arg]));
    op.region(0).unwrap().append_block(block);

    let ir = format!("{op}");
    assert!(ir.contains("{function.arg_name = \"input\"}"));
    assert!(ir.contains("{function.res_name = \"output\"}"));
}

#[test]
fn func_def_op_signature_name_accessors_return_out_of_bounds() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "bounds",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();

    assert_eq!(
        op.arg_name_attr(1).unwrap_err(),
        LlzkError::OutOfBoundsArgument(Some("@bounds".to_string()), 1)
    );
    assert_eq!(
        op.res_name_attr(1).unwrap_err(),
        LlzkError::OutOfBoundsArgument(Some("@bounds".to_string()), 1)
    );
    assert_eq!(
        op.set_arg_name(1, "input").unwrap_err(),
        LlzkError::OutOfBoundsArgument(Some("@bounds".to_string()), 1)
    );
    assert_eq!(
        op.set_res_name(1, "output").unwrap_err(),
        LlzkError::OutOfBoundsArgument(Some("@bounds".to_string()), 1)
    );
}

#[test]
fn func_def_op_arg_and_res_attrs_round_trip() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let op = dialect::function::def(
        loc,
        "attrs_round_trip",
        FunctionType::new(&context, &[felt_type], &[felt_type]),
        &[],
        None,
    )
    .unwrap();

    let arg_attrs = ArrayAttribute::new(
        &context,
        &[Attribute::from(unsafe {
            melior::ir::Attribute::from_raw(mlir_sys::mlirDictionaryAttrGet(
                context.to_raw(),
                1,
                [{
                    mlir_sys::mlirNamedAttributeGet(
                        Identifier::new(&context, FUNCTION_ARG_NAME_ATTR_NAME.as_ref()).to_raw(),
                        StringAttribute::new(&context, "input").to_raw(),
                    )
                }]
                .as_ptr(),
            ))
        })],
    );
    let res_attrs = ArrayAttribute::new(
        &context,
        &[Attribute::from(unsafe {
            melior::ir::Attribute::from_raw(mlir_sys::mlirDictionaryAttrGet(
                context.to_raw(),
                1,
                [{
                    mlir_sys::mlirNamedAttributeGet(
                        Identifier::new(&context, FUNCTION_RES_NAME_ATTR_NAME.as_ref()).to_raw(),
                        StringAttribute::new(&context, "output").to_raw(),
                    )
                }]
                .as_ptr(),
            ))
        })],
    );

    op.set_arg_attrs(arg_attrs);
    op.set_res_attrs(res_attrs);

    assert_eq!(op.arg_attrs().unwrap().len(), 1);
    assert_eq!(op.res_attrs().unwrap().len(), 1);
    assert_eq!(op.arg_name_attr(0).unwrap().unwrap().value(), "input");
    assert_eq!(op.res_name_attr(0).unwrap().unwrap().value(), "output");
}

// Tests for CallOpLike methods added in e2157c6

fn make_call_op_in_block<'c, 'a>(
    context: &'c LlzkContext,
    loc: Location<'c>,
    block: &Block<'c>,
    args: &[Value<'c, '_>],
) -> CallOpRef<'c, 'a> {
    let builder = OpBuilder::at_block_begin(context, block);
    let name = FlatSymbolRefAttribute::new(context, "callee");
    let call = block.append_operation(
        dialect::function::call(&builder, loc, name, args, &[] as &[Type])
            .unwrap()
            .into(),
    );
    CallOpRef::try_from(call).unwrap()
}

#[test]
fn call_op_arg_operand_count_zero() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert_eq!(call.arg_operand_count(), 0);
}

#[test]
fn call_op_arg_operand_count_nonzero() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    let call = make_call_op_in_block(&context, loc, &block, &[arg]);
    assert_eq!(call.arg_operand_count(), 1);
}

#[test]
fn call_op_arg_operand_at() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    let call = make_call_op_in_block(&context, loc, &block, &[arg]);
    assert_eq!(call.arg_operand_at(0), arg);
}

#[test]
fn call_op_set_arg_operands() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    // Build call with no args initially.
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert_eq!(call.arg_operand_count(), 0);
    call.set_arg_operands(&[arg]);
    assert_eq!(call.arg_operand_count(), 1);
}

#[test]
fn call_op_map_operand_count_zero() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let builder = OpBuilder::at_block_begin(&context, &block);
    let name = FlatSymbolRefAttribute::new(&context, "callee");
    let call = block.append_operation(
        dialect::function::call_with_map_operands(
            &builder,
            loc,
            name,
            &[],
            &[] as &[Type],
            MapOperandsBuilder::new(),
        )
        .unwrap()
        .into(),
    );
    let call = CallOpRef::try_from(call).unwrap();
    assert_eq!(call.map_operand_count(), 0);
}

#[test]
fn call_op_set_map_operands() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert_eq!(call.map_operand_count(), 0);
    call.set_map_operands(&[]);
    assert_eq!(call.map_operand_count(), 0);
}

#[test]
fn call_op_get_callee() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    let callee = call.callee().unwrap();
    assert_eq!(callee.root().as_str().unwrap(), "callee");
}

#[test]
fn call_op_set_callee() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert_eq!(call.callee().unwrap().root().as_str().unwrap(), "callee");
    call.set_callee(SymbolRefAttribute::new_from_str(
        &context,
        "new_callee",
        &[],
    ));
    assert_eq!(
        call.callee().unwrap().root().as_str().unwrap(),
        "new_callee"
    );
}

#[test]
fn call_op_get_template_params_none() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert!(call.template_params().unwrap().is_none());
}

#[test]
fn call_op_set_template_params() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert!(call.template_params().unwrap().is_none());
    // Set a non-None value.
    call.set_template_params(Some(ArrayAttribute::new(&context, &[])));
    assert!(call.template_params().unwrap().is_some());
    // Clear back to None.
    call.set_template_params(None);
    assert!(call.template_params().unwrap().is_none());
}

#[test]
fn call_with_template_params_only_attrs() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let module = Module::new(loc);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let name = FlatSymbolRefAttribute::new(&context, "callee");
    let felt_type: Type = FeltType::new(&context).into();
    let call = dialect::function::call_with_template_params(
        &builder,
        loc,
        name,
        &[],
        &[] as &[Type],
        &[TypeAttribute::new(felt_type)],
    )
    .unwrap();
    let template_params = call.template_params().unwrap().unwrap();
    assert_eq!(template_params.len(), 1);
}

#[test]
fn call_with_template_params_only_args() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    let builder = OpBuilder::at_block_begin(&context, &block);
    let name = FlatSymbolRefAttribute::new(&context, "callee");
    let call = block.append_operation(
        dialect::function::call_with_template_params(
            &builder,
            loc,
            name,
            &[arg],
            &[] as &[Type],
            &[] as &[Attribute],
        )
        .unwrap()
        .into(),
    );
    let call = CallOpRef::try_from(call).unwrap();
    assert_eq!(call.arg_operand_count(), 1);
    assert!(call.template_params().unwrap().is_none());
}

#[test]
fn call_with_template_params_no_empties() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    let builder = OpBuilder::at_block_begin(&context, &block);
    let name = FlatSymbolRefAttribute::new(&context, "callee");
    let call = block.append_operation(
        dialect::function::call_with_template_params(
            &builder,
            loc,
            name,
            &[arg],
            &[felt_type],
            &[TypeAttribute::new(felt_type)],
        )
        .unwrap()
        .into(),
    );
    let call = CallOpRef::try_from(call).unwrap();
    assert_eq!(call.arg_operand_count(), 1);
    assert!(call.template_params().unwrap().is_some());
}
