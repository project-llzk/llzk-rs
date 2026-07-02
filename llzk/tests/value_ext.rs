#![allow(unused_crate_dependencies)]
//! Integration tests for value extension helpers.

use llzk::{
    attributes::array::AffineMapAttribute,
    prelude::*,
    value_ext::{has_uses, replace_all_uses_in_block_with, users_of},
};

mod common;

#[test]
fn replace_all_uses_in_block_with_handles_repeated_operands() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block_arg = (felt_type, location);
    let block = Block::new(&[block_arg, block_arg]);
    let orig = block.argument(0).unwrap();
    let replacement = block.argument(1).unwrap();
    let replacement_value: Value = replacement.into();

    let use_before = block
        .append_operation(dialect::felt::mul(location, orig.into(), replacement_value).unwrap());
    let repeated_use =
        block.append_operation(dialect::felt::add(location, orig.into(), orig.into()).unwrap());
    let use_after = block
        .append_operation(dialect::felt::sub(location, orig.into(), replacement_value).unwrap());

    replace_all_uses_in_block_with(orig.owner(), orig, replacement);

    assert!(!has_uses(orig), "all uses of orig should be replaced");
    for op in [use_before, repeated_use, use_after] {
        for operand in op.operands() {
            assert_eq!(operand, replacement_value, "unexpected operand in {op}");
        }
    }
}

#[test]
fn replace_all_uses_in_block_with_only_replaces_orig() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block_arg = (felt_type, location);
    let block = Block::new(&[block_arg, block_arg, block_arg]);
    let orig = block.argument(0).unwrap();
    let replacement = block.argument(1).unwrap();
    let untouched = block.argument(2).unwrap();
    let replacement_value: Value = replacement.into();
    let untouched_value: Value = untouched.into();

    let mixed_use = block
        .append_operation(dialect::felt::add(location, orig.into(), untouched.into()).unwrap());
    let non_orig_use = block.append_operation(
        dialect::felt::mul(location, untouched.into(), replacement.into()).unwrap(),
    );

    replace_all_uses_in_block_with(orig.owner(), orig, replacement);

    assert!(!has_uses(orig), "all uses of orig should be replaced");
    assert_eq!(
        mixed_use.operands().collect::<Vec<_>>().as_slice(),
        &[replacement_value, untouched_value],
        "only the orig operand should be replaced in {mixed_use}"
    );
    assert_eq!(
        non_orig_use.operands().collect::<Vec<_>>().as_slice(),
        &[untouched_value, replacement_value],
        "operands that were not orig should remain unchanged in {non_orig_use}"
    );
}

// Tests for users_of

#[test]
fn users_of_empty_when_no_uses() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, location)]);
    let arg = block.argument(0).unwrap();
    let users = users_of(arg);
    assert!(users.is_empty());
}

#[test]
fn users_of_returns_single_user() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, location)]);
    let arg = block.argument(0).unwrap();
    block.append_operation(dialect::felt::neg(location, arg.into()).unwrap());
    let users = users_of(arg);
    assert_eq!(users.len(), 1);
}

#[test]
fn users_of_returns_multiple_users() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, location)]);
    let arg: Value = block.argument(0).unwrap().into();
    block.append_operation(dialect::felt::neg(location, arg).unwrap());
    block.append_operation(dialect::felt::inv(location, arg).unwrap());
    let users = users_of(arg);
    assert_eq!(users.len(), 2);
}

// Tests for print_operation, print_block, print_region

#[test]
fn print_operation_does_not_panic() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, location)]);
    let arg: Value = block.argument(0).unwrap().into();
    let op = block.append_operation(dialect::felt::neg(location, arg).unwrap());
    print_operation(&op);
}

#[test]
fn print_block_does_not_panic() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let block = Block::new(&[(felt_type, location)]);
    let arg: Value = block.argument(0).unwrap().into();
    block.append_operation(dialect::felt::neg(location, arg).unwrap());
    print_block(&block);
}

#[test]
fn print_region_does_not_panic() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let felt_type: Type = FeltType::new(&context).into();
    let func_type = melior::ir::r#type::FunctionType::new(&context, &[felt_type], &[]);
    let func = dialect::function::def(location, "test_fn", func_type, &[], None).unwrap();
    let block = Block::new(&[(felt_type, location)]);
    let arg: Value = block.argument(0).unwrap().into();
    block.append_operation(dialect::felt::neg(location, arg).unwrap());
    block.append_operation(dialect::function::r#return(location, &[]));
    let region = func.region(0).expect("function must have a region");
    region.append_block(block);
    print_region(&region);
}

// Tests for AffineMapAttribute::identity

#[test]
fn affine_map_attribute_identity_zero_dims() {
    common::setup();
    let context = LlzkContext::new();
    let attr = AffineMapAttribute::identity(&context, 0);
    let attr: melior::ir::Attribute = attr.into();
    // The attribute should have a string representation containing "affine_map"
    assert!(attr.to_string().contains("affine_map"));
}

#[test]
fn affine_map_attribute_identity_one_dim() {
    common::setup();
    let context = LlzkContext::new();
    let attr = AffineMapAttribute::identity(&context, 1);
    let attr: melior::ir::Attribute = attr.into();
    // Identity map for 1 dimension is (d0) -> (d0)
    let repr = attr.to_string();
    assert!(repr.contains("affine_map"));
    assert!(repr.contains("d0"));
}

#[test]
fn affine_map_attribute_identity_multi_dim() {
    common::setup();
    let context = LlzkContext::new();
    let attr = AffineMapAttribute::identity(&context, 3);
    let attr: melior::ir::Attribute = attr.into();
    // Identity map for 3 dimensions should reference d0, d1, d2
    let repr = attr.to_string();
    assert!(repr.contains("d0"));
    assert!(repr.contains("d1"));
    assert!(repr.contains("d2"));
}
