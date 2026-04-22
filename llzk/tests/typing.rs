use llzk::{context::LlzkContext, prelude::*, typing};
use melior::{StringRef, ir::r#type::Type};

mod common;

#[test]
fn identical_types_unify() {
    common::setup();
    let context = LlzkContext::new();
    let index = Type::index(&context);

    assert!(typing::types_equal_or_unifiable(index, index));
    assert!(typing::is_unifiable_with(index, index));
}

#[test]
fn identical_types_unify_with_empty_prefix() {
    common::setup();
    let context = LlzkContext::new();
    let felt: Type = FeltType::new(&context).into();

    assert!(typing::types_unify(felt, felt, &[]));
}

#[test]
fn identical_types_unify_with_prefix() {
    common::setup();
    let context = LlzkContext::new();
    let tvar: Type = TVarType::new(&context, StringRef::new("T")).into();
    let prefix = [StringRef::new("compute"), StringRef::new("StructA")];

    assert!(typing::types_unify(tvar, tvar, &prefix));
}
