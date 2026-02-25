use std::ptr::null_mut;

use mlir_sys::{mlirLocationUnknownGet, mlirOperationDestroy};
use rstest::rstest;

use crate::{
    llzkInclude_IncludeOpCreateInferredContext, mlirGetDialectHandle__llzk__include__,
    sanity_tests::{TestContext, context, str_ref},
};

#[test]
fn test_mlir_get_dialect_handle_llzk_include() {
    unsafe {
        mlirGetDialectHandle__llzk__include__();
    }
}

#[rstest]
fn test_llzk_include_op_create(context: TestContext) {
    unsafe {
        let location = mlirLocationUnknownGet(context.ctx);
        let op = llzkInclude_IncludeOpCreateInferredContext(
            location,
            str_ref("test"),
            str_ref("test.mlir"),
        );

        assert_ne!(op.ptr, null_mut());
        mlirOperationDestroy(op);
    }
}
