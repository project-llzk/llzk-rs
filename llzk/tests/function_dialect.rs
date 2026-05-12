use llzk::{
    attributes::array::ArrayAttribute,
    builder::{OpBuilder, OpBuilderLike},
    map_operands::MapOperandsBuilder,
    prelude::*,
};
use melior::ir::{Location, r#type::FunctionType};

mod common;

#[test]
fn empty_function() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
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
    let module = llzk_module(Location::unknown(&context));
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
        let builder =
            OpBuilder::at_block_begin(&context, unsafe { BlockRef::from_raw(block.to_raw()) });
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
    let module = llzk_module(Location::unknown(&context));
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
        let builder =
            OpBuilder::at_block_begin(&context, unsafe { BlockRef::from_raw(block.to_raw()) });
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

#[test]
fn func_def_op_self_value_of_compute() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context));
    let module_body = module.body();

    let s = make_empty_struct(&context, "StructA");
    let s = StructDefOpRef::try_from(module_body.append_operation(s.into())).unwrap();
    llzk::operation::verify_operation_with_diags(&s).expect("verification failed");
    log::info!("Struct passed verification");

    let compute_fn = s
        .get_compute_func()
        .expect("failed to get compute function");
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
    let module = llzk_module(Location::unknown(&context));
    let module_body = module.body();

    let s = make_empty_struct(&context, "StructA");
    let s = StructDefOpRef::try_from(module_body.append_operation(s.into())).unwrap();
    llzk::operation::verify_operation_with_diags(&s).expect("verification failed");
    log::info!("Struct passed verification");

    let constrain_fn = s
        .get_constrain_func()
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
    let module = llzk_module(Location::unknown(&context));
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
        .get_compute_func()
        .expect("failed to get compute function")
        .region(0)
        .expect("failed to get first region")
        .first_block()
        .expect("failed to get first block");
    let builder = OpBuilder::at_block_begin(&context, s2_compute_body);
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
    let result = op.get_function_type().unwrap();
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
    assert_eq!(op.get_function_type().unwrap().input_count(), 0);
    op.set_function_type(FunctionType::new(&context, &[felt_type], &[]));
    assert_eq!(op.get_function_type().unwrap().input_count(), 1);
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
    assert_eq!(op.get_sym_name().unwrap().value(), "my_func");
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
    assert_eq!(op.get_sym_name().unwrap().value(), "my_func");
    op.set_sym_name(StringAttribute::new(&context, "new_name"));
    assert_eq!(op.get_sym_name().unwrap().value(), "new_name");
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
    assert!(op.get_body().is_ok());

    // After appending a block, get_body still succeeds.
    let block = Block::new(&[]);
    block.append_operation(dialect::function::r#return(loc, &[]));
    op.region(0)
        .expect("function.def must have at least 1 region")
        .append_block(block);
    assert!(op.get_body().is_ok());
}

// Tests for CallOpLike methods added in e2157c6

fn make_call_op_in_block<'c, 'a>(
    context: &'c LlzkContext,
    loc: Location<'c>,
    block: &Block<'c>,
    args: &[Value<'c, '_>],
) -> CallOpRef<'c, 'a> {
    let builder = OpBuilder::new(context);
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
    let builder = OpBuilder::new(&context);
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
    let callee = call.get_callee().unwrap();
    assert_eq!(callee.root().as_str().unwrap(), "callee");
}

#[test]
fn call_op_set_callee() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert_eq!(
        call.get_callee().unwrap().root().as_str().unwrap(),
        "callee"
    );
    call.set_callee(SymbolRefAttribute::new_from_str(
        &context,
        "new_callee",
        &[],
    ));
    assert_eq!(
        call.get_callee().unwrap().root().as_str().unwrap(),
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
    assert!(call.get_template_params().unwrap().is_none());
}

#[test]
fn call_op_set_template_params() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let block = Block::new(&[]);
    let call = make_call_op_in_block(&context, loc, &block, &[]);
    assert!(call.get_template_params().unwrap().is_none());
    // Set a non-None value.
    call.set_template_params(Some(ArrayAttribute::new(&context, &[])));
    assert!(call.get_template_params().unwrap().is_some());
    // Clear back to None.
    call.set_template_params(None);
    assert!(call.get_template_params().unwrap().is_none());
}

#[test]
fn call_with_template_params_only_attrs() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let builder = OpBuilder::new(&context);
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
    let template_params = call.get_template_params().unwrap().unwrap();
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
    let builder = OpBuilder::new(&context);
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
    assert!(call.get_template_params().unwrap().is_none());
}

#[test]
fn call_with_template_params_no_empties() {
    common::setup();
    let context = LlzkContext::new();
    let loc = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, loc)]);
    let arg: Value = block.argument(0).unwrap().into();
    let builder = OpBuilder::new(&context);
    let name = FlatSymbolRefAttribute::new(&context, "callee");
    let felt_type: Type = FeltType::new(&context).into();
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
    assert!(call.get_template_params().unwrap().is_some());
}
