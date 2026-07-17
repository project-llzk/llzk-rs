use crate::{
    llzkInclude_IncludeOpBuildInferredContext, mlirGetDialectHandle__llzk__include__,
    mlirOpBuilderCreate, mlirOpBuilderDestroy,
    sanity_tests::{TestContext, context, str_ref},
};
use mlir_sys::{mlirLocationUnknownGet, mlirOperationDestroy};
use rstest::rstest;
use std::ptr::null_mut;

#[test]
fn test_mlir_get_dialect_handle_llzk_include() {
    unsafe {
        mlirGetDialectHandle__llzk__include__();
    }
}

#[rstest]
fn test_llzk_include_op_create(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let op = llzkInclude_IncludeOpBuildInferredContext(
            builder,
            location,
            str_ref("test"),
            str_ref("test.mlir"),
        );

        assert_ne!(op.ptr, null_mut());
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}
