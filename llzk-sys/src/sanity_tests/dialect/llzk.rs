use std::ptr::null;

use rstest::rstest;

use crate::{
    llzkAttributeIsA_Llzk_PublicAttr, llzkLlzk_PublicAttrGet, mlirGetDialectHandle__llzk__,
    sanity_tests::{TestContext, context},
};

#[test]
fn test_mlir_get_dialect_handle_llzk() {
    unsafe {
        mlirGetDialectHandle__llzk__();
    }
}

#[rstest]
fn test_llzk_public_attr_get(context: TestContext) {
    unsafe {
        let attr = llzkLlzk_PublicAttrGet(context.ctx);
        assert_ne!(attr.ptr, null());
    };
}

#[rstest]
fn test_llzk_attribute_is_a_public_attr_pass(context: TestContext) {
    unsafe {
        let attr = llzkLlzk_PublicAttrGet(context.ctx);
        assert!(llzkAttributeIsA_Llzk_PublicAttr(attr));
    };
}
