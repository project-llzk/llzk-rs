use llzk::{
    prelude::*,
    value_ext::{has_uses, replace_all_uses_in_block_with},
};
use melior::ir::{Location, Type};

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
