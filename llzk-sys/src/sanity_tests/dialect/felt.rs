use std::ptr::null;

use mlir_sys::{mlirIndexTypeGet, mlirIntegerAttrGet};
use rstest::rstest;

use crate::{
    llzkAttributeIsA_Felt_FeltConstAttr, llzkFelt_FeltConstAttrGetFromPartsUnspecified,
    llzkFelt_FeltConstAttrGetFromStringUnspecified, llzkFelt_FeltConstAttrGetUnspecified,
    llzkFelt_FeltConstAttrGetWithBitsUnspecified, llzkFelt_FeltTypeGetUnspecified,
    llzkTypeIsA_Felt_FeltType, mlirGetDialectHandle__llzk__felt__,
    sanity_tests::{TestContext, context, str_ref},
};

#[test]
fn test_mlir_get_dialect_handle_llzk_felt() {
    unsafe {
        mlirGetDialectHandle__llzk__felt__();
    }
}

#[rstest]
fn test_llzk_felt_const_attr_get(context: TestContext) {
    unsafe {
        let attr = llzkFelt_FeltConstAttrGetUnspecified(context.ctx, 0);
        assert_ne!(attr.ptr, null());
    };
}

#[rstest]
fn test_llzk_felt_const_attr_get_with_bits(context: TestContext) {
    unsafe {
        let attr = llzkFelt_FeltConstAttrGetWithBitsUnspecified(context.ctx, 128, 0);
        assert_ne!(attr.ptr, null());
    };
}

#[rstest]
fn test_llzk_felt_const_attr_get_from_str(context: TestContext) {
    unsafe {
        let attr = llzkFelt_FeltConstAttrGetFromStringUnspecified(context.ctx, 64, str_ref("123"));
        assert_ne!(attr.ptr, null());
    };
}

#[rstest]
fn test_llzk_felt_const_attr_get_from_parts(context: TestContext) {
    unsafe {
        let parts = [123, 0];
        let attr = llzkFelt_FeltConstAttrGetFromPartsUnspecified(
            context.ctx,
            128,
            parts.as_ptr(),
            parts.len() as isize,
        );
        assert_ne!(attr.ptr, null());
    };
}

#[rstest]
fn test_llzk_attribute_is_a_felt_const_attr_pass(context: TestContext) {
    unsafe {
        let attr = llzkFelt_FeltConstAttrGetUnspecified(context.ctx, 0);
        assert!(llzkAttributeIsA_Felt_FeltConstAttr(attr));
    };
}

#[rstest]
fn test_llzk_attribute_is_a_felt_const_attr_fail(context: TestContext) {
    unsafe {
        let attr = mlirIntegerAttrGet(mlirIndexTypeGet(context.ctx), 0);
        assert!(!llzkAttributeIsA_Felt_FeltConstAttr(attr));
    };
}

#[rstest]
fn test_llzk_felt_type_get(context: TestContext) {
    unsafe {
        let r#type = llzkFelt_FeltTypeGetUnspecified(context.ctx);
        assert_ne!(r#type.ptr, null());
    };
}

#[rstest]
fn test_llzk_type_is_a_felt_type_pass(context: TestContext) {
    unsafe {
        let r#type = llzkFelt_FeltTypeGetUnspecified(context.ctx);
        assert!(llzkTypeIsA_Felt_FeltType(r#type));
    };
}

#[rstest]
fn test_llzk_type_is_a_felt_type_fail(context: TestContext) {
    unsafe {
        let r#type = mlirIndexTypeGet(context.ctx);
        assert!(!llzkTypeIsA_Felt_FeltType(r#type));
    };
}
