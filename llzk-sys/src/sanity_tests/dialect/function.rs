use crate::{
    llzkAffineMapOperandsBuilderCreate, llzkAffineMapOperandsBuilderDestroy,
    llzkFunction_CallOpBuild, llzkFunction_CallOpBuildToCallee,
    llzkFunction_CallOpBuildToCalleeWithMapOperands, llzkFunction_CallOpBuildWithMapOperands,
    llzkFunction_CallOpCalleeIsCompute, llzkFunction_CallOpCalleeIsConstrain,
    llzkFunction_CallOpCalleeIsProduct, llzkFunction_CallOpCalleeIsStructCompute,
    llzkFunction_CallOpCalleeIsStructConstrain, llzkFunction_CallOpCalleeIsStructProduct,
    llzkFunction_CallOpGetSingleResultTypeOfCompute, llzkFunction_CallOpGetTypeSignature,
    llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs, llzkFunction_FuncDefOpGetBody,
    llzkFunction_FuncDefOpGetFullyQualifiedName,
    llzkFunction_FuncDefOpGetSingleResultTypeOfCompute,
    llzkFunction_FuncDefOpHasAllowConstraintAttr, llzkFunction_FuncDefOpHasAllowWitnessAttr,
    llzkFunction_FuncDefOpHasArgPublicAttr, llzkFunction_FuncDefOpIsInStruct,
    llzkFunction_FuncDefOpIsStructCompute, llzkFunction_FuncDefOpIsStructConstrain,
    llzkFunction_FuncDefOpIsStructProduct, llzkFunction_FuncDefOpNameIsCompute,
    llzkFunction_FuncDefOpNameIsConstrain, llzkFunction_FuncDefOpNameIsProduct,
    llzkFunction_FuncDefOpSetAllowConstraintAttr, llzkFunction_FuncDefOpSetAllowWitnessAttr,
    llzkFunction_ReturnOpBuild, llzkOperationIsA_Function_CallOp,
    llzkOperationIsA_Function_FuncDefOp, llzkStruct_CreateStructOpBuild,
    llzkStruct_CreateStructOpGetResult, llzkStruct_StructDefOpBuild,
    llzkStruct_StructDefOpGetBodyRegion, llzkStruct_StructDefOpGetType,
    mlirGetDialectHandle__llzk__function__, mlirOpBuilderCreate, mlirOpBuilderDestroy,
    mlirOpBuilderSetInsertionPointToEnd,
    sanity_tests::{TestContext, context, str_ref},
};
use mlir_sys::{
    MlirAttribute, MlirBlock, MlirContext, MlirModule, MlirNamedAttribute, MlirOperation, MlirType,
    mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirDictionaryAttrGet,
    mlirFlatSymbolRefAttrGet, mlirFunctionTypeGet, mlirIdentifierGet, mlirIndexTypeGet,
    mlirLocationUnknownGet, mlirModuleCreateEmpty, mlirModuleDestroy, mlirModuleGetBody,
    mlirModuleGetOperation, mlirOperationDestroy, mlirOperationGetContext,
    mlirOperationSetAttributeByName, mlirOperationVerify, mlirRegionAppendOwnedBlock,
    mlirStringRefCreateFromCString, mlirTypeEqual, mlirUnitAttrGet,
};
use rstest::{fixture, rstest};
use std::{ffi::CString, ptr::null};

#[test]
fn test_mlir_get_dialect_handle_llzk_function() {
    unsafe {
        mlirGetDialectHandle__llzk__function__();
    }
}

fn create_func_type(ctx: MlirContext, ins: &[MlirType], outs: &[MlirType]) -> MlirType {
    unsafe {
        mlirFunctionTypeGet(
            ctx,
            isize::try_from(ins.len()).expect("ins too large"),
            ins.as_ptr(),
            isize::try_from(outs.len()).expect("outs too large"),
            outs.as_ptr(),
        )
    }
}

fn create_func_def_op(
    ctx: MlirContext,
    name: &str,
    r#type: MlirType,
    attrs: &[MlirNamedAttribute],
    arg_attrs: &[MlirAttribute],
) -> MlirOperation {
    unsafe {
        let location = mlirLocationUnknownGet(ctx);
        let name = CString::new(name).unwrap();
        let name = mlirStringRefCreateFromCString(name.as_ptr());
        llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
            location,
            name,
            r#type,
            isize::try_from(attrs.len()).expect("attrs too large"),
            attrs.as_ptr(),
            isize::try_from(arg_attrs.len()).expect("arg_attrs too large"),
            arg_attrs.as_ptr(),
        )
    }
}

struct TestModule {
    module: MlirModule,
}

impl Drop for TestModule {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.module) }
    }
}

fn empty_arg_attrs<const N: usize>(ctx: MlirContext, _: &[MlirType; N]) -> [MlirAttribute; N] {
    std::array::from_fn(|_| unsafe { mlirDictionaryAttrGet(ctx, 0, null()) })
}

fn append_empty_block(region: mlir_sys::MlirRegion) -> MlirBlock {
    unsafe {
        let block = mlirBlockCreate(0, null(), null());
        mlirRegionAppendOwnedBlock(region, block);
        block
    }
}

#[rstest]
fn test_llzk_func_def_op_create_with_attrs_and_arg_attrs(context: TestContext) {
    unsafe {
        let in_types = [mlirIndexTypeGet(context.ctx)];
        let in_attrs = empty_arg_attrs(context.ctx, &in_types);
        //let in_attrs = [mlirDictionaryAttrGet(context.ctx, 0, null())];
        let op = create_func_def_op(
            context.ctx,
            "foo",
            create_func_type(context.ctx, &in_types, &[]),
            &[],
            &in_attrs,
        );
        mlirOperationDestroy(op);
    }
}

struct TestFuncDefOp {
    #[allow(dead_code)]
    context: TestContext,
    pub op: MlirOperation,
    pub in_types: Vec<MlirType>,
    pub out_types: Vec<MlirType>,
    pub name: &'static str,
}

impl Drop for TestFuncDefOp {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.op) }
    }
}

#[fixture]
fn test_function(context: TestContext) -> TestFuncDefOp {
    let in_types = [unsafe { mlirIndexTypeGet(context.ctx) }, unsafe {
        mlirIndexTypeGet(context.ctx)
    }];
    let in_attrs = empty_arg_attrs(context.ctx, &in_types);
    let out_types = [unsafe { mlirIndexTypeGet(context.ctx) }];
    let name = "foo";
    let ctx = context.ctx;
    TestFuncDefOp {
        context,
        in_types: in_types.to_vec(),
        out_types: out_types.to_vec(),
        name,
        op: create_func_def_op(
            ctx,
            name,
            create_func_type(ctx, &in_types, &out_types),
            &[],
            &in_attrs,
        ),
    }
}

#[fixture]
fn test_function0(context: TestContext) -> TestFuncDefOp {
    let in_types = [];
    let out_types = [unsafe { mlirIndexTypeGet(context.ctx) }];
    let name = "bar";
    let ctx = context.ctx;
    TestFuncDefOp {
        context,
        in_types: in_types.to_vec(),
        out_types: out_types.to_vec(),
        name,
        op: create_func_def_op(
            ctx,
            name,
            create_func_type(ctx, &in_types, &out_types),
            &[],
            &[],
        ),
    }
}

#[rstest]
fn test_llzk_operation_is_a_func_def_op(test_function: TestFuncDefOp) {
    unsafe {
        assert!(llzkOperationIsA_Function_FuncDefOp(test_function.op));
    }
}

#[rstest]
fn test_llzk_func_def_op_get_has_allow_constraint_attr(test_function: TestFuncDefOp) {
    unsafe {
        assert!(!llzkFunction_FuncDefOpHasAllowConstraintAttr(
            test_function.op
        ));
    }
}

#[rstest]
fn test_llzk_func_def_op_set_allow_constraint_attr(test_function: TestFuncDefOp) {
    unsafe {
        assert!(!llzkFunction_FuncDefOpHasAllowConstraintAttr(
            test_function.op
        ));
        llzkFunction_FuncDefOpSetAllowConstraintAttr(test_function.op, true);
        assert!(llzkFunction_FuncDefOpHasAllowConstraintAttr(
            test_function.op
        ));
        llzkFunction_FuncDefOpSetAllowConstraintAttr(test_function.op, false);
        assert!(!llzkFunction_FuncDefOpHasAllowConstraintAttr(
            test_function.op
        ));
    }
}

#[rstest]
fn test_llzk_func_def_op_get_has_allow_witness_attr(test_function: TestFuncDefOp) {
    unsafe {
        assert!(!llzkFunction_FuncDefOpHasAllowWitnessAttr(test_function.op));
    }
}

#[rstest]
fn test_llzk_func_def_op_set_allow_witness_attr(test_function: TestFuncDefOp) {
    unsafe {
        assert!(!llzkFunction_FuncDefOpHasAllowWitnessAttr(test_function.op));
        llzkFunction_FuncDefOpSetAllowWitnessAttr(test_function.op, true);
        assert!(llzkFunction_FuncDefOpHasAllowWitnessAttr(test_function.op));
        llzkFunction_FuncDefOpSetAllowWitnessAttr(test_function.op, false);
        assert!(!llzkFunction_FuncDefOpHasAllowWitnessAttr(test_function.op));
    }
}

#[rstest]
fn test_llzk_func_def_op_get_has_arg_is_pub(test_function: TestFuncDefOp) {
    unsafe { assert!(!llzkFunction_FuncDefOpHasArgPublicAttr(test_function.op, 0)) }
}

#[rstest]
fn test_llzk_func_def_op_get_fully_qualified_name(test_function: TestFuncDefOp) {
    unsafe {
        llzkFunction_FuncDefOpGetFullyQualifiedName(test_function.op, false);
    }
}

macro_rules! false_pred_test {
    ($test:ident, $func:ident) => {
        #[rstest]
        fn $test(test_function: TestFuncDefOp) {
            unsafe {
                assert!(!$func(test_function.op));
            }
        }
    };
}

false_pred_test!(
    test_llzk_func_def_op_get_name_is_compute,
    llzkFunction_FuncDefOpNameIsCompute
);
false_pred_test!(
    test_llzk_func_def_op_get_name_is_constrain,
    llzkFunction_FuncDefOpNameIsConstrain
);
false_pred_test!(
    test_llzk_func_def_op_get_name_is_product,
    llzkFunction_FuncDefOpNameIsProduct
);
false_pred_test!(
    test_llzk_func_def_op_get_is_in_struct,
    llzkFunction_FuncDefOpIsInStruct
);
false_pred_test!(
    test_llzk_func_def_op_get_is_struct_compute,
    llzkFunction_FuncDefOpIsStructCompute
);
false_pred_test!(
    test_llzk_func_def_op_get_is_struct_constrain,
    llzkFunction_FuncDefOpIsStructConstrain
);
false_pred_test!(
    test_llzk_func_def_op_get_is_struct_product,
    llzkFunction_FuncDefOpIsStructProduct
);

#[rstest]
fn test_llzk_func_def_op_get_single_result_type_of_compute(test_function: TestFuncDefOp) {
    unsafe {
        // We want to link the function to make sure it has been implemented but we don't want to
        // call it because the precondition checks will fail with the test function.
        if llzkFunction_FuncDefOpIsStructCompute(test_function.op) {
            llzkFunction_FuncDefOpGetSingleResultTypeOfCompute(test_function.op);
        }
    }
}

#[rstest]
fn test_llzk_call_op_build(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let callee_name = str_ref(test_function0.name);
        let callee_name = mlirFlatSymbolRefAttrGet(ctx, callee_name);
        let call = llzkFunction_CallOpBuild(
            builder,
            location,
            isize::try_from(test_function0.out_types.len()).expect("out_types too large"),
            test_function0.out_types.as_ptr(),
            callee_name,
            0,
            null(),
        );
        assert!(mlirOperationVerify(call));
        mlirOperationDestroy(call);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_call_op_build_to_callee(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let call =
            llzkFunction_CallOpBuildToCallee(builder, location, test_function0.op, 0, null());
        assert!(mlirOperationVerify(call));
        mlirOperationDestroy(call);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn llzk_call_op_build_with_map_operands(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let callee_name = str_ref(test_function0.name);
        let callee_name = mlirFlatSymbolRefAttrGet(ctx, callee_name);
        let mut map_operands = llzkAffineMapOperandsBuilderCreate();
        let call = llzkFunction_CallOpBuildWithMapOperands(
            builder,
            location,
            isize::try_from(test_function0.out_types.len()).expect("out_types too large"),
            test_function0.out_types.as_ptr(),
            callee_name,
            map_operands,
            0,
            null(),
        );
        assert!(mlirOperationVerify(call));
        mlirOperationDestroy(call);
        llzkAffineMapOperandsBuilderDestroy(&mut map_operands);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn llzk_call_op_build_to_callee_with_map_operands(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let mut map_operands = llzkAffineMapOperandsBuilderCreate();
        let call = llzkFunction_CallOpBuildToCalleeWithMapOperands(
            builder,
            location,
            test_function0.op,
            map_operands,
            0,
            null(),
        );
        assert!(mlirOperationVerify(call));
        mlirOperationDestroy(call);
        llzkAffineMapOperandsBuilderDestroy(&mut map_operands);
        mlirOpBuilderDestroy(builder);
    }
}

macro_rules! call_pred_test {
    ($test:ident, $func:ident, $expected:expr) => {
        #[rstest]
        fn $test(test_function0: TestFuncDefOp) {
            unsafe {
                let ctx = mlirOperationGetContext(test_function0.op);
                let builder = mlirOpBuilderCreate(ctx);
                let location = mlirLocationUnknownGet(ctx);
                let call = llzkFunction_CallOpBuildToCallee(
                    builder,
                    location,
                    test_function0.op,
                    0,
                    null(),
                );

                assert_eq!($func(call), $expected);
                mlirOperationDestroy(call);
                mlirOpBuilderDestroy(builder);
            }
        }
    };
}

call_pred_test!(
    test_llzk_operation_is_a_call_op,
    llzkOperationIsA_Function_CallOp,
    true
);

#[rstest]
fn test_llzk_call_op_get_type_signature(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let call =
            llzkFunction_CallOpBuildToCallee(builder, location, test_function0.op, 0, null());

        let func_type = create_func_type(ctx, &test_function0.in_types, &test_function0.out_types);
        let out_type = llzkFunction_CallOpGetTypeSignature(call);
        assert!(mlirTypeEqual(func_type, out_type));

        mlirOperationDestroy(call);
        mlirOpBuilderDestroy(builder);
    }
}

call_pred_test!(
    test_llzk_call_op_get_callee_is_compute,
    llzkFunction_CallOpCalleeIsCompute,
    false
);
call_pred_test!(
    test_llzk_call_op_get_callee_is_constrain,
    llzkFunction_CallOpCalleeIsConstrain,
    false
);
call_pred_test!(
    test_llzk_call_op_get_callee_is_product,
    llzkFunction_CallOpCalleeIsProduct,
    false
);
call_pred_test!(
    test_llzk_call_op_get_callee_is_struct_compute,
    llzkFunction_CallOpCalleeIsStructCompute,
    false
);
call_pred_test!(
    test_llzk_call_op_get_callee_is_struct_constrain,
    llzkFunction_CallOpCalleeIsStructConstrain,
    false
);
call_pred_test!(
    test_llzk_call_op_get_callee_is_struct_product,
    llzkFunction_CallOpCalleeIsStructProduct,
    false
);

#[rstest]
fn test_llzk_call_op_get_callee_is_product_positive(context: TestContext) {
    let loc = unsafe { mlirLocationUnknownGet(context.ctx) };
    let module = TestModule {
        module: unsafe { mlirModuleCreateEmpty(loc) },
    };
    let builder = unsafe { mlirOpBuilderCreate(context.ctx) };

    unsafe {
        mlirOperationSetAttributeByName(
            mlirModuleGetOperation(module.module),
            str_ref("llzk.lang"),
            mlirUnitAttrGet(context.ctx),
        );
        let module_body = mlirModuleGetBody(module.module);
        mlirOpBuilderSetInsertionPointToEnd(builder, module_body);

        let struct_a = llzkStruct_StructDefOpBuild(
            builder,
            loc,
            mlirIdentifierGet(context.ctx, str_ref("StructProdA")),
        );
        let struct_a_body = append_empty_block(llzkStruct_StructDefOpGetBodyRegion(struct_a));
        let struct_a_type = llzkStruct_StructDefOpGetType(struct_a);
        let product_a = create_func_def_op(
            context.ctx,
            "product",
            create_func_type(context.ctx, &[], &[struct_a_type]),
            &[],
            &[],
        );
        mlirBlockAppendOwnedOperation(struct_a_body, product_a);
        let product_a_body = append_empty_block(llzkFunction_FuncDefOpGetBody(product_a));
        mlirOpBuilderSetInsertionPointToEnd(builder, product_a_body);
        let self_a = llzkStruct_CreateStructOpGetResult(llzkStruct_CreateStructOpBuild(
            builder,
            loc,
            struct_a_type,
        ));
        llzkFunction_ReturnOpBuild(builder, loc, 1, &self_a);

        mlirOpBuilderSetInsertionPointToEnd(builder, module_body);
        let struct_b = llzkStruct_StructDefOpBuild(
            builder,
            loc,
            mlirIdentifierGet(context.ctx, str_ref("StructProdB")),
        );
        let struct_b_body = append_empty_block(llzkStruct_StructDefOpGetBodyRegion(struct_b));
        let struct_b_type = llzkStruct_StructDefOpGetType(struct_b);
        let product_b = create_func_def_op(
            context.ctx,
            "product",
            create_func_type(context.ctx, &[], &[struct_b_type]),
            &[],
            &[],
        );
        mlirBlockAppendOwnedOperation(struct_b_body, product_b);
        let product_b_body = append_empty_block(llzkFunction_FuncDefOpGetBody(product_b));
        mlirOpBuilderSetInsertionPointToEnd(builder, product_b_body);
        let self_b = llzkStruct_CreateStructOpGetResult(llzkStruct_CreateStructOpBuild(
            builder,
            loc,
            struct_b_type,
        ));
        let call = llzkFunction_CallOpBuildToCallee(builder, loc, product_a, 0, null());
        llzkFunction_ReturnOpBuild(builder, loc, 1, &self_b);

        assert!(llzkFunction_CallOpCalleeIsProduct(call));
        assert!(llzkFunction_CallOpCalleeIsStructProduct(call));
        assert!(!llzkFunction_CallOpCalleeIsCompute(call));
        assert!(!llzkFunction_CallOpCalleeIsConstrain(call));
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_call_op_get_single_result_type_of_compute(test_function0: TestFuncDefOp) {
    unsafe {
        let ctx = mlirOperationGetContext(test_function0.op);
        let builder = mlirOpBuilderCreate(ctx);
        let location = mlirLocationUnknownGet(ctx);
        let call =
            llzkFunction_CallOpBuildToCallee(builder, location, test_function0.op, 0, null());

        // We want to link the function to make sure it has been implemented but we don't want to
        // call it because the precondition checks will fail with the test function.
        if llzkFunction_CallOpCalleeIsStructCompute(call) {
            llzkFunction_CallOpGetSingleResultTypeOfCompute(call);
        }

        mlirOperationDestroy(call);
        mlirOpBuilderDestroy(builder);
    }
}
