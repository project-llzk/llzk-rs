use crate::{
    LlzkBoolFeltCmpPredicate, llzkAttributeIsA_Bool_FeltCmpPredicateAttr,
    llzkBool_FeltCmpPredicateAttrGet, mlirGetDialectHandle__llzk__boolean__,
    sanity_tests::{TestContext, context},
};
use mlir_sys::mlirUnitAttrGet;
use rstest::rstest;
use std::ptr::null;

#[test]
fn test_mlir_get_dialect_handle_llzk_boolean() {
    unsafe {
        mlirGetDialectHandle__llzk__boolean__();
    }
}

#[rstest]
fn test_llzk_felt_cmp_predicate_attr_get(
    context: TestContext,
    #[values(
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_EQ,
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_NE,
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_LT,
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_LE,
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_GT,
        crate::LlzkBoolFeltCmpPredicate_LlzkBoolFeltCmpPredicate_GE
    )]
    cmp: LlzkBoolFeltCmpPredicate,
) {
    unsafe {
        let attr = llzkBool_FeltCmpPredicateAttrGet(context.ctx, cmp);
        assert_ne!(attr.ptr, null());
    }
}

#[rstest]
fn test_llzk_attribute_is_a_felt_cmp_predicate_attr_pass(context: TestContext) {
    unsafe {
        let attr = llzkBool_FeltCmpPredicateAttrGet(context.ctx, 0);
        assert!(llzkAttributeIsA_Bool_FeltCmpPredicateAttr(attr));
    }
}

#[rstest]
fn test_llzk_attribute_is_a_felt_cmp_predicate_attr_fail(context: TestContext) {
    unsafe {
        let attr = mlirUnitAttrGet(context.ctx);
        assert!(!llzkAttributeIsA_Bool_FeltCmpPredicateAttr(attr));
    }
}
