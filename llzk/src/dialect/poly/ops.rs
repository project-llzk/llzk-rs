//! `poly` dialect operations and helper functions.

use crate::{
    builder::{OpBuilder, OpBuilderLike},
    ident,
    macros::llzk_op_type,
    value_ext::{OwningValueRange, ValueRange},
};
use llzk_sys::{
    llzkOperationIsA_Poly_TemplateExprOp, llzkOperationIsA_Poly_TemplateOp,
    llzkOperationIsA_Poly_TemplateParamOp, llzkOperationIsA_Poly_YieldOp,
    llzkPoly_ApplyMapOpBuildWithAffineMap, llzkPoly_TemplateExprOpBuild,
    llzkPoly_TemplateExprOpGetInitializerRegion, llzkPoly_TemplateExprOpGetType,
    llzkPoly_TemplateOpBuild, llzkPoly_TemplateOpGetBodyRegion,
    llzkPoly_TemplateOpGetConstExprNames, llzkPoly_TemplateOpGetConstParamNames,
    llzkPoly_TemplateOpHasConstExprNamed, llzkPoly_TemplateOpHasConstExprOps,
    llzkPoly_TemplateOpHasConstParamNamed, llzkPoly_TemplateOpHasConstParamOps,
    llzkPoly_TemplateOpNumConstExprOps, llzkPoly_TemplateOpNumConstParamOps,
    llzkPoly_TemplateParamOpBuild, llzkPoly_TemplateParamOpGetTypeOpt, llzkPoly_YieldOpBuild,
};
use melior::ir::{
    Attribute, AttributeLike, Block, BlockLike as _, Identifier, Location, Operation,
    RegionLike as _, RegionRef, Type, Value, ValueLike as _,
    attribute::{FlatSymbolRefAttribute, TypeAttribute},
    operation::{OperationBuilder, OperationLike},
};
use mlir_sys::MlirAttribute;

use crate::error::Error;

//===----------------------------------------------------------------------===//
// TemplateOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `poly.template` op.
pub trait TemplateOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the template body region.
    fn body_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkPoly_TemplateOpGetBodyRegion(self.to_raw())) }
    }

    /// Returns `true` if the template defines any `poly.param` children.
    fn has_const_param_ops(&self) -> bool {
        unsafe { llzkPoly_TemplateOpHasConstParamOps(self.to_raw()) }
    }

    /// Returns `true` if the template defines any `poly.expr` children.
    fn has_const_expr_ops(&self) -> bool {
        unsafe { llzkPoly_TemplateOpHasConstExprOps(self.to_raw()) }
    }

    /// Returns the names of all `poly.param` children in definition order.
    fn const_param_names(&self) -> Vec<Attribute<'c>> {
        let num_attrs =
            usize::try_from(unsafe { llzkPoly_TemplateOpNumConstParamOps(self.to_raw()) }).unwrap();
        let mut raw_attrs: Vec<MlirAttribute> = Vec::with_capacity(num_attrs);
        unsafe {
            llzkPoly_TemplateOpGetConstParamNames(self.to_raw(), raw_attrs.as_mut_ptr());
            raw_attrs.set_len(num_attrs);
        }
        raw_attrs
            .into_iter()
            .map(|attr| unsafe { Attribute::from_raw(attr) })
            .collect()
    }

    /// Returns the names of all `poly.expr` children in definition order.
    fn const_expr_names(&self) -> Vec<Attribute<'c>> {
        let num_attrs =
            usize::try_from(unsafe { llzkPoly_TemplateOpNumConstExprOps(self.to_raw()) }).unwrap();
        let mut raw_attrs: Vec<MlirAttribute> = Vec::with_capacity(num_attrs);
        unsafe {
            llzkPoly_TemplateOpGetConstExprNames(self.to_raw(), raw_attrs.as_mut_ptr());
            raw_attrs.set_len(num_attrs);
        }
        raw_attrs
            .into_iter()
            .map(|attr| unsafe { Attribute::from_raw(attr) })
            .collect()
    }

    /// Returns `true` if the template has a `poly.param` with the given name.
    fn has_const_param_named(&self, find: &str) -> bool {
        unsafe {
            let find = melior::StringRef::new(find);
            llzkPoly_TemplateOpHasConstParamNamed(self.to_raw(), find.to_raw())
        }
    }

    /// Returns `true` if the template has a `poly.expr` with the given name.
    fn has_const_expr_named(&self, find: &str) -> bool {
        unsafe {
            let find = melior::StringRef::new(find);
            llzkPoly_TemplateOpHasConstExprNamed(self.to_raw(), find.to_raw())
        }
    }
}

//===----------------------------------------------------------------------===//
// TemplateExprOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `poly.expr` op.
pub trait TemplateExprOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the initializer region.
    fn initializer_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkPoly_TemplateExprOpGetInitializerRegion(self.to_raw())) }
    }

    /// Returns the type yielded from the initializer region.
    fn expr_type(&self) -> Type<'c> {
        unsafe { Type::from_raw(llzkPoly_TemplateExprOpGetType(self.to_raw())) }
    }
}

//===----------------------------------------------------------------------===//
// TemplateParamOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `poly.param` op.
pub trait TemplateParamOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the optional declared type restriction on the parameter.
    fn type_opt(&self) -> Option<Type<'c>> {
        let raw_attr = unsafe { llzkPoly_TemplateParamOpGetTypeOpt(self.to_raw()) };
        if raw_attr.ptr.is_null() {
            None
        } else {
            let attr = unsafe { Attribute::from_raw(raw_attr) };
            let type_attr = TypeAttribute::try_from(attr).expect("malformed poly.param type_opt");
            Some(type_attr.value())
        }
    }
}

//===----------------------------------------------------------------------===//
// Typed op wrappers
//===----------------------------------------------------------------------===//

llzk_op_type!(
    TemplateOp,
    llzkOperationIsA_Poly_TemplateOp,
    "poly.template"
);
llzk_op_type!(
    TemplateExprOp,
    llzkOperationIsA_Poly_TemplateExprOp,
    "poly.expr"
);
llzk_op_type!(
    TemplateParamOp,
    llzkOperationIsA_Poly_TemplateParamOp,
    "poly.param"
);
llzk_op_type!(YieldOp, llzkOperationIsA_Poly_YieldOp, "poly.yield");

impl<'a, 'c: 'a> TemplateOpLike<'c, 'a> for TemplateOp<'c> {}
impl<'a, 'c: 'a> TemplateOpLike<'c, 'a> for TemplateOpRef<'c, 'a> {}
impl<'a, 'c: 'a> TemplateOpLike<'c, 'a> for TemplateOpRefMut<'c, 'a> {}

impl<'a, 'c: 'a> TemplateExprOpLike<'c, 'a> for TemplateExprOp<'c> {}
impl<'a, 'c: 'a> TemplateExprOpLike<'c, 'a> for TemplateExprOpRef<'c, 'a> {}
impl<'a, 'c: 'a> TemplateExprOpLike<'c, 'a> for TemplateExprOpRefMut<'c, 'a> {}

impl<'a, 'c: 'a> TemplateParamOpLike<'c, 'a> for TemplateParamOp<'c> {}
impl<'a, 'c: 'a> TemplateParamOpLike<'c, 'a> for TemplateParamOpRef<'c, 'a> {}
impl<'a, 'c: 'a> TemplateParamOpLike<'c, 'a> for TemplateParamOpRefMut<'c, 'a> {}

/// Constructs a 'poly.applymap' operation.
pub fn applymap<'c>(
    location: Location<'c>,
    map: Attribute<'c>,
    map_operands: &[Value<'c, '_>],
) -> Operation<'c> {
    let ctx = location.context();
    let builder = OpBuilder::new(unsafe { ctx.to_ref() });
    let value_range = OwningValueRange::from(map_operands);
    assert!(unsafe { mlir_sys::mlirAttributeIsAAffineMap(map.to_raw()) });
    unsafe {
        Operation::from_raw(llzkPoly_ApplyMapOpBuildWithAffineMap(
            builder.to_raw(),
            location.to_raw(),
            mlir_sys::mlirAffineMapAttrGetValue(map.to_raw()),
            ValueRange::try_from(&value_range).unwrap().to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `poly.applymap`.
#[inline]
pub fn is_applymap_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.applymap")
}

/// Creates a `poly.template` op and fills its body with the given operations.
pub fn template<'c, I>(
    location: Location<'c>,
    name: &str,
    region_ops: I,
) -> Result<TemplateOp<'c>, Error>
where
    I: IntoIterator<Item = Result<Operation<'c>, Error>>,
{
    let ctx = location.context();
    let builder = OpBuilder::new(unsafe { ctx.to_ref() });
    let op = unsafe {
        Operation::from_raw(llzkPoly_TemplateOpBuild(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(ctx.to_ref(), name).to_raw(),
        ))
    };
    let op: TemplateOp<'c> = op.try_into()?;
    let region = op.body_region();
    let block = region
        .first_block()
        .unwrap_or_else(|| region.append_block(Block::new(&[])));
    region_ops
        .into_iter()
        .try_for_each(|inner_op| -> Result<(), Error> {
            block.append_operation(inner_op?);
            Ok(())
        })?;
    Ok(op)
}

/// Return `true` iff the given op is `poly.template`.
#[inline]
pub fn is_template_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.template")
}

/// Constructs a 'poly.read_const' operation.
pub fn read_const<'c>(location: Location<'c>, symbol: &str, result: Type<'c>) -> Operation<'c> {
    let ctx = location.context();
    OperationBuilder::new("poly.read_const", location)
        .add_attributes(&[(
            ident!(ctx, "const_name"),
            FlatSymbolRefAttribute::new(unsafe { ctx.to_ref() }, symbol).into(),
        )])
        .add_results(&[result])
        .build()
        .expect("valid operation")
}

/// Return `true` iff the given op is `poly.read_const`.
#[inline]
pub fn is_read_const_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.read_const")
}

/// Creates a `poly.param` op.
pub fn param<'c>(
    location: Location<'c>,
    name: &str,
    type_opt: Option<Type<'c>>,
) -> Result<TemplateParamOp<'c>, Error> {
    let ctx = location.context();
    let builder = OpBuilder::new(unsafe { ctx.to_ref() });
    let raw_type = type_opt
        .map(|t| TypeAttribute::new(t).into())
        .unwrap_or_else(|| unsafe {
            Attribute::from_raw(MlirAttribute {
                ptr: std::ptr::null_mut(),
            })
        })
        .to_raw();
    unsafe {
        Operation::from_raw(llzkPoly_TemplateParamOpBuild(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(ctx.to_ref(), name).to_raw(),
            raw_type,
        ))
    }
    .try_into()
}

/// Return `true` iff the given op is `poly.param`.
#[inline]
pub fn is_param_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.param")
}

/// Creates a `poly.expr` op and fills its initializer region with the given operations.
pub fn expr<'c, I>(
    location: Location<'c>,
    name: &str,
    region_ops: I,
) -> Result<TemplateExprOp<'c>, Error>
where
    I: IntoIterator<Item = Result<Operation<'c>, Error>>,
{
    let ctx = location.context();
    let builder = OpBuilder::new(unsafe { ctx.to_ref() });
    let op = unsafe {
        Operation::from_raw(llzkPoly_TemplateExprOpBuild(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(ctx.to_ref(), name).to_raw(),
        ))
    };
    let op: TemplateExprOp<'c> = op.try_into()?;
    let region = op.initializer_region();
    let block = region
        .first_block()
        .unwrap_or_else(|| region.append_block(Block::new(&[])));
    region_ops
        .into_iter()
        .try_for_each(|inner_op| -> Result<(), Error> {
            block.append_operation(inner_op?);
            Ok(())
        })?;
    Ok(op)
}

/// Return `true` iff the given op is `poly.expr`.
#[inline]
pub fn is_expr_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.expr")
}

/// Constructs a 'poly.unifiable_cast' operation.
pub fn unifiable_cast<'c>(
    location: Location<'c>,
    input: Value<'c, '_>,
    result: Type<'c>,
) -> Operation<'c> {
    OperationBuilder::new("poly.unifiable_cast", location)
        .add_operands(&[input])
        .add_results(&[result])
        .build()
        .expect("valid operation")
}

/// Return `true` iff the given op is `poly.unifiable_cast`.
#[inline]
pub fn is_unifiable_cast_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.unifiable_cast")
}

/// Creates a `poly.yield` op.
pub fn r#yield<'c>(location: Location<'c>, val: Value<'c, '_>) -> Result<YieldOp<'c>, Error> {
    let ctx = location.context();
    let builder = OpBuilder::new(unsafe { ctx.to_ref() });
    unsafe {
        Operation::from_raw(llzkPoly_YieldOpBuild(
            builder.to_raw(),
            location.to_raw(),
            val.to_raw(),
        ))
    }
    .try_into()
}

/// Return `true` iff the given op is `poly.yield`.
#[inline]
pub fn is_yield_op<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "poly.yield")
}
