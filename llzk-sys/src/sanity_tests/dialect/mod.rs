use crate::sanity_tests::{TestContext, context};
use mlir_sys::{
    MlirOperation, mlirIdentifierGet, mlirIndexTypeGet, mlirIntegerAttrGet, mlirLocationUnknownGet,
    mlirNamedAttributeGet, mlirOperationCreate, mlirOperationDestroy,
    mlirOperationStateAddAttributes, mlirOperationStateAddResults, mlirOperationStateGet,
    mlirStringRefCreateFromCString,
};
use rstest::fixture;
use std::ffi::CString;

struct TestOp {
    #[allow(dead_code)]
    context: TestContext,
    op: MlirOperation,
}

impl Drop for TestOp {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.op) }
    }
}

#[fixture]
fn test_op(context: TestContext) -> TestOp {
    unsafe {
        let ctx = context.ctx;
        let elt_type = mlirIndexTypeGet(ctx);
        let arith_constant_op_str = CString::new("arith.constant").unwrap();
        let value_str = CString::new("value").unwrap();
        let name = mlirStringRefCreateFromCString(arith_constant_op_str.as_ptr());
        let attr_name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString(value_str.as_ptr()));
        let location = mlirLocationUnknownGet(ctx);
        let results = [elt_type];
        let attr = mlirIntegerAttrGet(elt_type, 1);
        let attrs = [mlirNamedAttributeGet(attr_name, attr)];
        let mut op_state = mlirOperationStateGet(name, location);
        mlirOperationStateAddResults(
            &mut op_state,
            isize::try_from(results.len()).expect("results too large"),
            results.as_ptr(),
        );
        mlirOperationStateAddAttributes(
            &mut op_state,
            isize::try_from(attrs.len()).expect("attrs too large"),
            attrs.as_ptr(),
        );
        TestOp {
            context,
            op: mlirOperationCreate(&mut op_state),
        }
    }
}

mod array;
mod boolean;
mod cast;
mod felt;
mod function;
mod global;
mod include;
mod llzk;
mod pod;
mod polymorphic;
mod string;
mod r#struct;
