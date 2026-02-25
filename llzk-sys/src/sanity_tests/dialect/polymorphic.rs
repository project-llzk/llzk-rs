use std::ptr::{null, null_mut};

use mlir_sys::{
    MlirValue, mlirAffineConstantExprGet, mlirAffineMapAttrGet, mlirAffineMapEqual,
    mlirAffineMapGet, mlirAttributeEqual, mlirFlatSymbolRefAttrGet, mlirLocationUnknownGet,
    mlirOperationDestroy, mlirOperationVerify, mlirStringAttrGet, mlirStringRefEqual,
};
use rstest::rstest;

use crate::{
    MlirValueRange, llzkOperationIsA_Poly_ApplyMapOp, llzkPoly_ApplyMapOpBuild,
    llzkPoly_ApplyMapOpBuildWithAffineExpr, llzkPoly_ApplyMapOpBuildWithAffineMap,
    llzkPoly_ApplyMapOpGetAffineMap, llzkPoly_ApplyMapOpGetDimOperands,
    llzkPoly_ApplyMapOpGetNumDimOperands, llzkPoly_ApplyMapOpGetNumSymbolOperands,
    llzkPoly_ApplyMapOpGetSymbolOperands, llzkPoly_TypeVarTypeGetFromAttr,
    llzkPoly_TypeVarTypeGetFromStringRef, llzkPoly_TypeVarTypeGetNameRef,
    llzkPoly_TypeVarTypeGetRefName, llzkTypeIsA_Poly_TypeVarType,
    mlirGetDialectHandle__llzk__polymorphic__, mlirOpBuilderCreate, mlirOpBuilderDestroy,
    sanity_tests::{TestContext, context, str_ref},
};

#[test]
fn test_mlir_get_dialect_handle_llzk_polymorphic() {
    unsafe {
        mlirGetDialectHandle__llzk__polymorphic__();
    }
}

#[rstest]
fn test_llzk_type_var_type_get(context: TestContext) {
    unsafe {
        let t = llzkPoly_TypeVarTypeGetFromStringRef(context.ctx, str_ref("T"));
        assert_ne!(t.ptr, null());
    }
}

#[rstest]
fn test_llzk_type_is_a_type_var_type(context: TestContext) {
    unsafe {
        let t = llzkPoly_TypeVarTypeGetFromStringRef(context.ctx, str_ref("T"));
        assert!(llzkTypeIsA_Poly_TypeVarType(t));
    }
}

#[rstest]
fn test_llzk_type_var_type_get_from_attr(context: TestContext) {
    unsafe {
        let s = mlirStringAttrGet(context.ctx, str_ref("T"));
        let t = llzkPoly_TypeVarTypeGetFromAttr(s);
        assert_ne!(t.ptr, null());
    }
}

#[rstest]
fn test_llzk_type_var_type_get_name_ref(context: TestContext) {
    unsafe {
        let s = str_ref("T");
        let t = llzkPoly_TypeVarTypeGetFromStringRef(context.ctx, s);
        assert_ne!(t.ptr, null());
        assert!(mlirStringRefEqual(s, llzkPoly_TypeVarTypeGetRefName(t)));
    }
}

#[rstest]
fn test_llzk_type_var_type_get_name(context: TestContext) {
    unsafe {
        let s = str_ref("T");
        let t = llzkPoly_TypeVarTypeGetFromStringRef(context.ctx, s);
        let s = mlirFlatSymbolRefAttrGet(context.ctx, s);
        assert_ne!(t.ptr, null());
        assert!(mlirAttributeEqual(s, llzkPoly_TypeVarTypeGetNameRef(t)));
    }
}

#[rstest]
fn test_llzk_apply_map_op_build(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let mut exprs = [mlirAffineConstantExprGet(context.ctx, 1)];
        let affine_map =
            mlirAffineMapGet(context.ctx, 0, 0, exprs.len() as isize, exprs.as_mut_ptr());
        let affine_map = mlirAffineMapAttrGet(affine_map);
        let op = llzkPoly_ApplyMapOpBuild(
            builder,
            location,
            affine_map,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_apply_map_op_build_with_affine_map(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let mut exprs = [mlirAffineConstantExprGet(context.ctx, 1)];
        let affine_map =
            mlirAffineMapGet(context.ctx, 0, 0, exprs.len() as isize, exprs.as_mut_ptr());
        let op = llzkPoly_ApplyMapOpBuildWithAffineMap(
            builder,
            location,
            affine_map,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_apply_map_op_build_with_affine_expr(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let expr = mlirAffineConstantExprGet(context.ctx, 1);
        let op = llzkPoly_ApplyMapOpBuildWithAffineExpr(
            builder,
            location,
            expr,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_op_is_a_apply_map_op(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let expr = mlirAffineConstantExprGet(context.ctx, 1);
        let op = llzkPoly_ApplyMapOpBuildWithAffineExpr(
            builder,
            location,
            expr,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        assert!(llzkOperationIsA_Poly_ApplyMapOp(op));
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_apply_map_op_get_affine_map(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let mut exprs = [mlirAffineConstantExprGet(context.ctx, 1)];
        let affine_map =
            mlirAffineMapGet(context.ctx, 0, 0, exprs.len() as isize, exprs.as_mut_ptr());
        let op = llzkPoly_ApplyMapOpBuildWithAffineMap(
            builder,
            location,
            affine_map,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        let out_affine_map = llzkPoly_ApplyMapOpGetAffineMap(op);
        assert!(mlirAffineMapEqual(affine_map, out_affine_map));
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

fn boxed_value_range(size: isize) -> Box<[MlirValue]> {
    vec![MlirValue { ptr: null() }; size as usize].into_boxed_slice()
}

#[rstest]
fn test_llzk_apply_map_op_get_dim_operands(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let mut exprs = [mlirAffineConstantExprGet(context.ctx, 1)];
        let affine_map =
            mlirAffineMapGet(context.ctx, 0, 0, exprs.len() as isize, exprs.as_mut_ptr());
        let op = llzkPoly_ApplyMapOpBuildWithAffineMap(
            builder,
            location,
            affine_map,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        let n_dims = llzkPoly_ApplyMapOpGetNumDimOperands(op);
        let mut dims = boxed_value_range(n_dims);
        llzkPoly_ApplyMapOpGetDimOperands(op, dims.as_mut_ptr());
        assert_eq!(dims.len(), 0);
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}

#[rstest]
fn test_llzk_apply_map_op_get_symbol_operands(context: TestContext) {
    unsafe {
        let builder = mlirOpBuilderCreate(context.ctx);
        let location = mlirLocationUnknownGet(context.ctx);
        let mut exprs = [mlirAffineConstantExprGet(context.ctx, 1)];
        let affine_map =
            mlirAffineMapGet(context.ctx, 0, 0, exprs.len() as isize, exprs.as_mut_ptr());
        let op = llzkPoly_ApplyMapOpBuildWithAffineMap(
            builder,
            location,
            affine_map,
            MlirValueRange {
                values: null(),
                size: 0,
            },
        );
        assert_ne!(op.ptr, null_mut());
        assert!(mlirOperationVerify(op));
        let n_syms = llzkPoly_ApplyMapOpGetNumSymbolOperands(op);
        let mut syms = boxed_value_range(n_syms);
        llzkPoly_ApplyMapOpGetSymbolOperands(op, syms.as_mut_ptr());
        assert_eq!(syms.len(), 0);
        mlirOperationDestroy(op);
        mlirOpBuilderDestroy(builder);
    }
}
