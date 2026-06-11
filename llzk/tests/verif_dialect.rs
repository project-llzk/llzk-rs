use llzk::{
    attributes::NamedAttribute,
    builder::{OpBuilder, OpBuilderLike},
    prelude::*,
    value_ext::{OwningValueRange, ValueRange},
};
use melior::{
    dialect::arith,
    ir::{Identifier, attribute::DenseI32ArrayAttribute},
};

mod common;

fn make_function_target<'c>(
    context: &'c LlzkContext,
    module: &'c Module<'c>,
    name: &str,
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> FuncDefOpRef<'c, 'c> {
    let loc = Location::unknown(context);
    let felt_type: Type = FeltType::new(context).into();
    let func = dialect::function::def(
        loc,
        name,
        FunctionType::new(context, &[felt_type], &[felt_type]),
        &[],
        arg_attrs,
    )
    .unwrap();
    {
        let block = Block::new(&[(felt_type, loc)]);
        let arg: Value = block.argument(0).unwrap().into();
        block.append_operation(dialect::function::r#return(loc, &[arg]));
        func.region(0).unwrap().append_block(block);
    }
    FuncDefOpRef::try_from(module.body().append_operation(func.into())).unwrap()
}

fn make_zero_arg_function_target<'c>(
    context: &'c LlzkContext,
    module: &'c Module<'c>,
    name: &str,
) -> FuncDefOpRef<'c, 'c> {
    let loc = Location::unknown(context);
    let func =
        dialect::function::def(loc, name, FunctionType::new(context, &[], &[]), &[], None).unwrap();
    {
        let block = Block::new(&[]);
        block.append_operation(dialect::function::r#return(loc, &[]));
        func.region(0).unwrap().append_block(block);
    }
    FuncDefOpRef::try_from(module.body().append_operation(func.into())).unwrap()
}

fn make_struct_target<'c>(
    context: &'c LlzkContext,
    module: &'c Module<'c>,
    name: &str,
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> StructDefOpRef<'c, 'c> {
    let loc = Location::unknown(context);
    let typ = StructType::from_str_params(context, name, &[]);
    let felt_type: Type = FeltType::new(context).into();
    let inputs = [(felt_type, loc)];
    let ops = [
        dialect::r#struct::helpers::compute_fn(loc, typ, &inputs, arg_attrs).map(Into::into),
        dialect::r#struct::helpers::constrain_fn(loc, typ, &inputs, arg_attrs).map(Into::into),
    ];
    let strukt = dialect::r#struct::def(loc, name, ops).unwrap();
    StructDefOpRef::try_from(module.body().append_operation(strukt.into())).unwrap()
}

fn bool_constant<'c>(context: &'c LlzkContext, value: bool) -> Operation<'c> {
    let loc = Location::unknown(context);
    let bool_type: Type = IntegerType::new(context, 1).into();
    arith::constant(
        context,
        IntegerAttribute::new(bool_type, i64::from(value)).into(),
        loc,
    )
}

#[test]
fn contract_from_function_target() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(
        &context,
        &module,
        "target_fn",
        Some(&[vec![dialect::function::arg_name_attr(&context, "input")]]),
    );
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "target_contract",
        target.fully_qualified_name(),
    )
    .unwrap();

    verify_operation_with_diags(&contract).unwrap();
    assert_eq!(contract.sym_name().unwrap().value(), "target_contract");
    assert_eq!(contract.target().unwrap().to_string(), "@target_fn");
    assert!(contract.has_func_target());
    assert!(!contract.has_struct_target());
    assert!(contract.has_arg_name(0));
    let collected = contract.inputs().collect::<Vec<_>>();
    assert_eq!(collected.len(), 2);
    assert_eq!(
        Value::from(collected[0]),
        Value::from(contract.argument(0).unwrap())
    );
    assert_eq!(
        Value::from(collected[1]),
        Value::from(contract.argument(1).unwrap())
    );
    let mut iter = contract.inputs();
    assert_eq!(iter.len(), 2);
    assert_eq!(
        Value::from(iter.next().unwrap()),
        Value::from(contract.argument(0).unwrap())
    );
    assert_eq!(iter.len(), 1);
    assert_eq!(
        Value::from(iter.next_back().unwrap()),
        Value::from(contract.argument(1).unwrap())
    );
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    assert_eq!(
        contract.function_type().unwrap().to_string(),
        "(!felt.type, !felt.type) -> ()"
    );
    assert_eq!(
        contract.fully_qualified_name().to_string(),
        "@target_contract"
    );
    assert!(format!("{contract}").contains("verif.contract @target_contract"));
    assert!(dialect::verif::is_contract(&contract));
}

#[test]
fn contract_from_struct_target() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let arg_attrs = [vec![
        PublicAttribute::new_named_attr(&context),
        dialect::function::arg_name_attr(&context, "input"),
    ]];
    let target = make_struct_target(&context, &module, "StructTarget", Some(&arg_attrs));
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "struct_contract",
        target.fully_qualified_name(),
    )
    .unwrap();

    verify_operation_with_diags(&contract).unwrap();
    assert!(contract.has_struct_target());
    assert!(!contract.has_func_target());
    assert!(contract.argument(0).is_ok());
    assert!(contract.has_arg_name(1));
    assert!(contract.arg_is_pub(1));
    let collected = contract.inputs().collect::<Vec<_>>();
    assert_eq!(collected.len(), 2);
    assert_eq!(
        Value::from(collected[0]),
        Value::from(contract.argument(0).unwrap())
    );
    assert_eq!(
        Value::from(collected[1]),
        Value::from(contract.argument(1).unwrap())
    );
    let mut iter = contract.inputs();
    assert_eq!(
        Value::from(iter.next_back().unwrap()),
        Value::from(contract.argument(1).unwrap())
    );
    assert_eq!(
        Value::from(iter.next().unwrap()),
        Value::from(contract.argument(0).unwrap())
    );
    assert!(iter.next().is_none());
}

#[test]
fn include_flat() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "callee_fn", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let contract_a = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "contract_a",
        target.fully_qualified_name(),
    )
    .unwrap();
    let contract_b = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "contract_b",
        target.fully_qualified_name(),
    )
    .unwrap();

    let body = contract_a.body().unwrap().first_block().unwrap();
    let arg0: Value = contract_a.argument(0).unwrap().into();
    let arg1: Value = contract_a.argument(1).unwrap().into();
    let builder = OpBuilder::at_block_begin(&context, body);
    let include = dialect::verif::include(
        &builder,
        Location::unknown(&context),
        SymbolRefAttribute::new_from_str(&context, "contract_b", &[]),
        &[arg0, arg1],
        None,
    )
    .unwrap();

    verify_operation_with_diags(&contract_a).unwrap();
    assert!(dialect::verif::is_include(&include));
    assert_eq!(include.arg_operand_count(), 2);
    assert_eq!(include.arg_operand_at(0), arg0);
    assert_eq!(include.arg_operand_at(1), arg1);
    let collected = include.arg_operands().collect::<Vec<_>>();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], include.arg_operand_at(0));
    assert_eq!(collected[1], include.arg_operand_at(1));
    let mut iter = include.arg_operands();
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.next().unwrap(), include.arg_operand_at(0));
    assert_eq!(iter.len(), 1);
    assert_eq!(iter.next_back().unwrap(), include.arg_operand_at(1));
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    assert_eq!(include.callee().unwrap().to_string(), "@contract_b");
    assert_eq!(
        include.type_signature().unwrap().to_string(),
        contract_b.function_type().unwrap().to_string()
    );
    assert_eq!(
        include.resolve_callable().unwrap(),
        OperationRef::from(contract_b)
    );
}

#[test]
fn function_arg_name_attr_helper() {
    common::setup();
    let context = LlzkContext::new();
    let (identifier, attr) = dialect::function::arg_name_attr(&context, "input");

    assert_eq!(identifier, Identifier::new(&context, "function.arg_name"));
    let attr = StringAttribute::try_from(attr).unwrap();
    assert_eq!(attr.value(), "input");
}

#[test]
fn function_res_name_attr_helper() {
    common::setup();
    let context = LlzkContext::new();
    let (identifier, attr) = dialect::function::res_name_attr(&context, "output");

    assert_eq!(identifier, Identifier::new(&context, "function.res_name"));
    let attr = StringAttribute::try_from(attr).unwrap();
    assert_eq!(attr.value(), "output");
}

#[test]
fn include_with_map_operands_empty_groups() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "map_callee_fn", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let contract_a = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "map_contract_a",
        target.fully_qualified_name(),
    )
    .unwrap();
    let _contract_b = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "map_contract_b",
        target.fully_qualified_name(),
    )
    .unwrap();

    let body = contract_a.body().unwrap().first_block().unwrap();
    let arg0: Value = contract_a.argument(0).unwrap().into();
    let arg1: Value = contract_a.argument(1).unwrap().into();
    let builder = OpBuilder::at_block_begin(&context, body);
    let include = dialect::verif::include_with_map_operands_slice(
        &builder,
        Location::unknown(&context),
        SymbolRefAttribute::new_from_str(&context, "map_contract_b", &[]),
        &[arg0, arg1],
        None,
        &[],
        &[],
    )
    .unwrap();

    verify_operation_with_diags(&contract_a).unwrap();
    let collected = include.arg_operands().collect::<Vec<_>>();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], include.arg_operand_at(0));
    assert_eq!(collected[1], include.arg_operand_at(1));
    assert_eq!(include.map_operand_count(), 0);
    assert_eq!(
        include.num_dims_per_map().unwrap().to_string(),
        DenseI32ArrayAttribute::new(&context, &[]).to_string()
    );
    assert_eq!(
        include.map_op_group_sizes().unwrap().to_string(),
        DenseI32ArrayAttribute::new(&context, &[]).to_string()
    );
}

#[test]
fn include_flat_no_arg_operands() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_zero_arg_function_target(&context, &module, "empty_callee_fn");
    let builder = OpBuilder::at_block_begin(&context, module.body());

    let contract_a = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "empty_contract_a",
        target.fully_qualified_name(),
    )
    .unwrap();
    let _contract_b = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "empty_contract_b",
        target.fully_qualified_name(),
    )
    .unwrap();

    let body = contract_a.body().unwrap().first_block().unwrap();
    let builder = OpBuilder::at_block_begin(&context, body);
    let include = dialect::verif::include(
        &builder,
        Location::unknown(&context),
        SymbolRefAttribute::new_from_str(&context, "empty_contract_b", &[]),
        &[],
        None,
    )
    .unwrap();

    verify_operation_with_diags(&contract_a).unwrap();
    let mut iter = include.arg_operands();
    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
}

#[test]
fn require_compute_op() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "cond_target", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "require_compute_contract",
        target.fully_qualified_name(),
    )
    .unwrap();
    let body = contract.body().unwrap().first_block().unwrap();
    let builder = OpBuilder::at_block_end(&context, body);
    let true_val: Value = builder.insert(Location::unknown(&context), |_, _| bool_constant(&context, true))
        .result(0)
        .unwrap()
        .into();
    let false_val: Value = builder
        .insert(Location::unknown(&context), |_, _| bool_constant(&context, false))
        .result(0)
        .unwrap()
        .into();
    let op =
        dialect::verif::require_compute(&builder, Location::unknown(&context), true_val).unwrap();
    assert!(dialect::verif::is_require_compute(&op));
    assert_eq!(op.condition(), true_val);
    op.set_condition(false_val);
    assert_eq!(op.condition(), false_val);
    verify_operation_with_diags(&contract).unwrap();
}

#[test]
fn require_constrain_op() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "cond_target", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "require_constrain_contract",
        target.fully_qualified_name(),
    )
    .unwrap();
    let body = contract.body().unwrap().first_block().unwrap();
    let builder = OpBuilder::at_block_end(&context, body);
    let true_val: Value = builder.insert(Location::unknown(&context), |_, _| bool_constant(&context, true))
        .result(0)
        .unwrap()
        .into();
    let false_val: Value = builder
        .insert(Location::unknown(&context), |_, _| bool_constant(&context, false))
        .result(0)
        .unwrap()
        .into();
    let op =
        dialect::verif::require_constrain(&builder, Location::unknown(&context), true_val).unwrap();
    assert!(dialect::verif::is_require_constrain(&op));
    assert_eq!(op.condition(), true_val);
    op.set_condition(false_val);
    assert_eq!(op.condition(), false_val);
    verify_operation_with_diags(&contract).unwrap();
}

#[test]
fn ensure_compute_op() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "cond_target", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "ensure_compute_contract",
        target.fully_qualified_name(),
    )
    .unwrap();
    let body = contract.body().unwrap().first_block().unwrap();
    let builder = OpBuilder::at_block_end(&context, body);
    let true_val: Value = builder.insert(Location::unknown(&context), |_, _| bool_constant(&context, true))
        .result(0)
        .unwrap()
        .into();
    let false_val: Value = builder
        .insert(Location::unknown(&context), |_, _| bool_constant(&context, false))
        .result(0)
        .unwrap()
        .into();
    let op =
        dialect::verif::ensure_compute(&builder, Location::unknown(&context), true_val).unwrap();
    assert!(dialect::verif::is_ensure_compute(&op));
    assert_eq!(op.condition(), true_val);
    op.set_condition(false_val);
    assert_eq!(op.condition(), false_val);
    verify_operation_with_diags(&contract).unwrap();
}

#[test]
fn ensure_constrain_op() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "cond_target", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "ensure_constrain_contract",
        target.fully_qualified_name(),
    )
    .unwrap();
    let body = contract.body().unwrap().first_block().unwrap();
    let builder = OpBuilder::at_block_end(&context, body);
    let true_val: Value = builder.insert(Location::unknown(&context), |_, _| bool_constant(&context, true))
        .result(0)
        .unwrap()
        .into();
    let false_val: Value = builder
        .insert(Location::unknown(&context), |_, _| bool_constant(&context, false))
        .result(0)
        .unwrap()
        .into();
    let op =
        dialect::verif::ensure_constrain(&builder, Location::unknown(&context), true_val).unwrap();
    assert!(dialect::verif::is_ensure_constrain(&op));
    assert_eq!(op.condition(), true_val);
    op.set_condition(false_val);
    assert_eq!(op.condition(), false_val);
    verify_operation_with_diags(&contract).unwrap();
}

#[test]
fn include_map_operand_setter_roundtrip() {
    common::setup();
    let context = LlzkContext::new();
    let module = llzk_module(Location::unknown(&context), None);
    let target = make_function_target(&context, &module, "setter_target", None);
    let builder = OpBuilder::at_block_begin(&context, module.body());
    let contract = dialect::verif::contract(
        &builder,
        Location::unknown(&context),
        "setter_contract",
        target.fully_qualified_name(),
    )
    .unwrap();
    let body = contract.body().unwrap().first_block().unwrap();

    let builder = OpBuilder::at_block_begin(&context, body);
    let arg0: Value = contract.argument(0).unwrap().into();
    let arg1: Value = contract.argument(1).unwrap().into();
    let include = dialect::verif::include(
        &builder,
        Location::unknown(&context),
        SymbolRefAttribute::new_from_str(&context, "setter_contract", &[]),
        &[arg0, arg1],
        None,
    )
    .unwrap();

    let group = OwningValueRange::from([arg0].as_slice());
    let group = ValueRange::try_from(&group).unwrap();
    include.set_map_operands(&[group]);
    include.set_num_dims_per_map(DenseI32ArrayAttribute::new(&context, &[0]));
    assert_eq!(include.map_operand_count(), 1);
    assert_eq!(include.map_operand_at(0), arg0);
}
