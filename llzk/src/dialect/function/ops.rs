use crate::{
    attributes::{NamedAttribute, array::ArrayAttribute},
    builder::{OpBuilder, OpBuilderLike as _},
    dialect::r#struct::StructType,
    error::Error,
    macros::llzk_op_type,
    map_operands::MapOperandsBuilder,
    symbol_ref::{SymbolRefAttrLike, SymbolRefAttribute},
};

use llzk_sys::{
    llzkFunction_CallOpBuild, llzkFunction_CallOpBuildWithMapOperands,
    llzkFunction_CallOpBuildWithTemplateParams, llzkFunction_CallOpCalleeIsCompute,
    llzkFunction_CallOpCalleeIsConstrain, llzkFunction_CallOpCalleeIsStructCompute,
    llzkFunction_CallOpCalleeIsStructConstrain, llzkFunction_CallOpGetArgOperandsAt,
    llzkFunction_CallOpGetArgOperandsCount, llzkFunction_CallOpGetCallee,
    llzkFunction_CallOpGetMapOperandsAt, llzkFunction_CallOpGetMapOperandsCount,
    llzkFunction_CallOpGetSelfValueFromCompute, llzkFunction_CallOpGetSelfValueFromConstrain,
    llzkFunction_CallOpGetTemplateParams, llzkFunction_CallOpSetArgOperands,
    llzkFunction_CallOpSetCallee, llzkFunction_CallOpSetMapOperands,
    llzkFunction_CallOpSetTemplateParams, llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs,
    llzkFunction_FuncDefOpGetArgAttrs, llzkFunction_FuncDefOpGetBody,
    llzkFunction_FuncDefOpGetFullyQualifiedName, llzkFunction_FuncDefOpGetFunctionType,
    llzkFunction_FuncDefOpGetSelfValueFromCompute, llzkFunction_FuncDefOpGetSelfValueFromConstrain,
    llzkFunction_FuncDefOpGetSingleResultTypeOfCompute, llzkFunction_FuncDefOpGetSymName,
    llzkFunction_FuncDefOpHasAllowConstraintAttr,
    llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr, llzkFunction_FuncDefOpHasAllowWitnessAttr,
    llzkFunction_FuncDefOpHasArgPublicAttr, llzkFunction_FuncDefOpIsDeclaration,
    llzkFunction_FuncDefOpIsInStruct, llzkFunction_FuncDefOpIsStructCompute,
    llzkFunction_FuncDefOpIsStructConstrain, llzkFunction_FuncDefOpNameIsCompute,
    llzkFunction_FuncDefOpNameIsConstrain, llzkFunction_FuncDefOpSetAllowConstraintAttr,
    llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr, llzkFunction_FuncDefOpSetAllowWitnessAttr,
    llzkFunction_FuncDefOpSetFunctionType, llzkFunction_FuncDefOpSetSymName,
    llzkOperationIsA_Function_CallOp, llzkOperationIsA_Function_FuncDefOp,
};
use melior::{
    Context, StringRef,
    ir::{
        Attribute, AttributeLike, BlockLike as _, Location, Operation, RegionLike as _, RegionRef,
        Type, TypeLike, Value,
        attribute::{StringAttribute, TypeAttribute},
        block::BlockArgument,
        operation::{OperationBuilder, OperationLike, OperationMutLike},
        r#type::FunctionType,
    },
};
use mlir_sys::{MlirAttribute, MlirNamedAttribute, mlirDictionaryAttrGet, mlirNamedAttributeGet};

use std::ptr::null;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

fn create_out_of_bounds_error<'c: 'a, 'a>(
    func: &(impl FuncDefOpLike<'c, 'a> + ?Sized),
    idx: usize,
) -> Error {
    let fqn = func.fully_qualified_name();
    Error::OutOfBoundsArgument(Some(fqn.to_string()), idx)
}

//===----------------------------------------------------------------------===//
// FuncDefOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the 'function.def' op.
pub trait FuncDefOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns true if the FuncDefOp has the allow_constraint attribute.
    fn has_allow_constraint_attr(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpHasAllowConstraintAttr(self.to_raw()) }
    }

    /// Sets the allow_constraint attribute in the FuncDefOp operation.
    fn set_allow_constraint_attr(&self, value: bool) {
        unsafe { llzkFunction_FuncDefOpSetAllowConstraintAttr(self.to_raw(), value) }
    }

    /// Returns true if the FuncDefOp has the allow_witness attribute.
    fn has_allow_witness_attr(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpHasAllowWitnessAttr(self.to_raw()) }
    }

    /// Sets the allow_witness attribute in the FuncDefOp operation.
    fn set_allow_witness_attr(&self, value: bool) {
        unsafe { llzkFunction_FuncDefOpSetAllowWitnessAttr(self.to_raw(), value) }
    }

    /// Returns true if the FuncDefOp has the allow_non_native_field_ops attribute.
    fn has_allow_non_native_field_ops_attr(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr(self.to_raw()) }
    }

    /// Sets the allow_non_native_field_ops attribute in the FuncDefOp operation.
    fn set_allow_non_native_field_ops_attr(&self, value: bool) {
        unsafe { llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(self.to_raw(), value) }
    }

    /// Returns true if the `idx`-th argument has the Pub attribute.
    fn arg_is_pub(&self, idx: u32) -> bool {
        unsafe { llzkFunction_FuncDefOpHasArgPublicAttr(self.to_raw(), idx) }
    }

    /// Returns the fully qualified name of the function.
    fn fully_qualified_name(&self) -> SymbolRefAttribute<'c> {
        unsafe {
            Attribute::from_raw(llzkFunction_FuncDefOpGetFullyQualifiedName(
                self.to_raw(),
                false,
            ))
        }
        .try_into()
        .expect("symbol ref attribute")
    }

    /// Returns true if the function's name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE).
    fn name_is_compute(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpNameIsCompute(self.to_raw()) }
    }

    /// Returns true if the function's name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN).
    fn name_is_constrain(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpNameIsConstrain(self.to_raw()) }
    }

    /// Returns true if the function's defined inside a struct.
    fn is_in_struct(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpIsInStruct(self.to_raw()) }
    }

    /// Returns true if the function is the struct's witness computation.
    fn is_struct_compute(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpIsStructCompute(self.to_raw()) }
    }

    /// Returns true if the function is the struct's constrain definition.
    fn is_struct_constrain(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpIsStructConstrain(self.to_raw()) }
    }

    /// If the function name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE), return the "self"
    /// value (i.e. the return value) from the function. Otherwise, return Err(ExpectedFunctionName).
    fn self_value_of_compute(&self) -> Result<Value<'c, 'a>, Error> {
        if self.name_is_compute() {
            Ok(unsafe {
                Value::from_raw(llzkFunction_FuncDefOpGetSelfValueFromCompute(self.to_raw()))
            })
        } else {
            Err(Error::ExpectedFunctionName(&llzk_sys::FUNC_NAME_COMPUTE))
        }
    }

    /// If the function name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN), return the "self"
    /// value (i.e. the first parameter) from the function. Otherwise, return Err(ExpectedFunctionName).
    fn self_value_of_constrain(&self) -> Result<Value<'c, 'a>, Error> {
        if self.name_is_constrain() {
            Ok(unsafe {
                Value::from_raw(llzkFunction_FuncDefOpGetSelfValueFromConstrain(
                    self.to_raw(),
                ))
            })
        } else {
            Err(Error::ExpectedFunctionName(&llzk_sys::FUNC_NAME_CONSTRAIN))
        }
    }

    /// Assuming the function is the compute function returns its StructType result.
    fn result_type_of_compute(&self) -> StructType<'c> {
        unsafe {
            Type::from_raw(llzkFunction_FuncDefOpGetSingleResultTypeOfCompute(
                self.to_raw(),
            ))
        }
        .try_into()
        .expect("struct type")
    }

    /// Returns the n-th argument of the function.
    fn argument(&self, idx: usize) -> Result<BlockArgument<'c, 'a>, Error> {
        self.body()
            .and_then(|region| {
                region
                    .first_block()
                    .ok_or(create_out_of_bounds_error(self, idx))
            })
            .and_then(|block| block.argument(idx).map_err(Into::into))
    }

    /// Looks for an attribute in the n-th argument of the function.
    fn argument_attr(&self, idx: usize, name: &str) -> Result<Attribute<'c>, Error> {
        let arg_attrs = melior::ir::attribute::ArrayAttribute::try_from(unsafe {
            Attribute::from_raw(llzkFunction_FuncDefOpGetArgAttrs(self.to_raw()))
        })
        .map_err(Error::Melior)?;
        let arg = arg_attrs.element(idx)?;
        let name_ref = StringRef::new(name);
        unsafe {
            Attribute::from_option_raw(mlir_sys::mlirDictionaryAttrGetElementByName(
                arg.to_raw(),
                name_ref.to_raw(),
            ))
        }
        .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    ///  Gets the [FunctionType] attribute.
    fn function_type(&self) -> Result<FunctionType<'c>, Error> {
        let attr =
            unsafe { Attribute::from_raw(llzkFunction_FuncDefOpGetFunctionType(self.to_raw())) };
        let type_attr: TypeAttribute<'c> = attr.try_into()?;
        let func_type: FunctionType<'c> = type_attr.value().try_into()?;
        Ok(func_type)
    }

    /// Sets the [FunctionType] attribute.
    fn set_function_type(&self, ty: FunctionType<'c>) {
        let type_attr = TypeAttribute::new(ty.into());
        unsafe { llzkFunction_FuncDefOpSetFunctionType(self.to_raw(), type_attr.to_raw()) }
    }

    /// Returns the sym_name attribute.
    fn sym_name(&self) -> Result<StringAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkFunction_FuncDefOpGetSymName(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the sym_name attribute.
    fn set_sym_name(&self, attr: StringAttribute<'c>) {
        unsafe { llzkFunction_FuncDefOpSetSymName(self.to_raw(), attr.to_raw()) }
    }

    /// Returns true if the function is a declaration (has no body).
    fn is_declaration(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpIsDeclaration(self.to_raw()) }
    }

    /// Returns the body region of the function.
    fn body(&self) -> Result<RegionRef<'c, 'a>, Error> {
        let raw = unsafe { llzkFunction_FuncDefOpGetBody(self.to_raw()) };
        if raw.ptr.is_null() {
            Err(Error::GeneralError(
                "no body in a declaration-only function",
            ))
        } else {
            Ok(unsafe { RegionRef::from_raw(raw) })
        }
    }
}

/// Mutable operations for the `function.def` op.
pub trait FuncDefOpMutLike<'c: 'a, 'a>: FuncDefOpLike<'c, 'a> + OperationMutLike<'c, 'a> {}

//===----------------------------------------------------------------------===//
// FuncDefOp, FuncDefOpRef, and FuncDefOpRefMut
//===----------------------------------------------------------------------===//

llzk_op_type!(
    FuncDefOp,
    llzkOperationIsA_Function_FuncDefOp,
    "function.def"
);

impl<'a, 'c: 'a> FuncDefOpLike<'c, 'a> for FuncDefOp<'c> {}

impl<'a, 'c: 'a> FuncDefOpLike<'c, 'a> for FuncDefOpRef<'c, 'a> {}

impl<'a, 'c: 'a> FuncDefOpLike<'c, 'a> for FuncDefOpRefMut<'c, 'a> {}

impl<'a, 'c: 'a> FuncDefOpMutLike<'c, 'a> for FuncDefOp<'c> {}

impl<'a, 'c: 'a> FuncDefOpMutLike<'c, 'a> for FuncDefOpRefMut<'c, 'a> {}

//===----------------------------------------------------------------------===//
// CallOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the 'function.call' op.
pub trait CallOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns true if the call target name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE).
    fn callee_is_compute(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsCompute(self.to_raw()) }
    }

    /// Returns true if the call target name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN).
    fn callee_is_constrain(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsConstrain(self.to_raw()) }
    }

    /// Return `true` iff the callee function name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE) within a StructDefOp.
    fn callee_is_struct_compute(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsStructCompute(self.to_raw()) }
    }

    /// Return `true` iff the callee function name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN) within a StructDefOp.
    fn callee_is_struct_constrain(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsStructConstrain(self.to_raw()) }
    }

    /// If the function name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE), return the "self"
    /// value (i.e. the return value) from the callee function. Otherwise, return Err(ExpectedFunctionName).
    fn self_value_of_compute(&self) -> Result<Value<'c, 'a>, Error> {
        if self.callee_is_compute() {
            Ok(unsafe {
                Value::from_raw(llzkFunction_CallOpGetSelfValueFromCompute(self.to_raw()))
            })
        } else {
            Err(Error::ExpectedFunctionName(&llzk_sys::FUNC_NAME_COMPUTE))
        }
    }

    /// If the function name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN), return the "self"
    /// value (i.e. the first parameter) from the callee function. Otherwise, return Err(ExpectedFunctionName).
    fn self_value_of_constrain(&self) -> Result<Value<'c, 'a>, Error> {
        if self.callee_is_constrain() {
            Ok(unsafe {
                Value::from_raw(llzkFunction_CallOpGetSelfValueFromConstrain(self.to_raw()))
            })
        } else {
            Err(Error::ExpectedFunctionName(&llzk_sys::FUNC_NAME_CONSTRAIN))
        }
    }

    /// Returns the number of arg operands in the call op.
    fn arg_operand_count(&self) -> usize {
        let n = unsafe { llzkFunction_CallOpGetArgOperandsCount(self.to_raw()) };
        usize::try_from(n).expect("size is negative or too large")
    }

    /// Returns the arg operand at the given index.
    fn arg_operand_at(&self, index: usize) -> Value<'c, 'a> {
        let index = isize::try_from(index).expect("index too large");
        unsafe { Value::from_raw(llzkFunction_CallOpGetArgOperandsAt(self.to_raw(), index)) }
    }

    /// Sets the arg operands of the call op.
    fn set_arg_operands(&self, values: &[Value<'c, '_>]) {
        unsafe {
            llzkFunction_CallOpSetArgOperands(
                self.to_raw(),
                isize::try_from(values.len()).expect("values too large"),
                values.as_ptr() as *const _,
            )
        }
    }

    /// Returns the number of map operands in the call op.
    fn map_operand_count(&self) -> usize {
        let n = unsafe { llzkFunction_CallOpGetMapOperandsCount(self.to_raw()) };
        usize::try_from(n).expect("size is negative or too large")
    }

    /// Returns the map operand at the given index.
    fn map_operand_at(&self, index: usize) -> Value<'c, 'a> {
        let index = isize::try_from(index).expect("index too large");
        unsafe { Value::from_raw(llzkFunction_CallOpGetMapOperandsAt(self.to_raw(), index)) }
    }

    /// Sets the map operands of the call op.
    fn set_map_operands(&self, values: &[Value<'c, '_>]) {
        unsafe {
            llzkFunction_CallOpSetMapOperands(
                self.to_raw(),
                isize::try_from(values.len()).expect("values too large"),
                values.as_ptr() as *const _,
            )
        }
    }

    /// Returns the callee attribute.
    fn callee(&self) -> Result<SymbolRefAttribute<'c>, Error> {
        let a: Attribute<'_> =
            unsafe { Attribute::from_raw(llzkFunction_CallOpGetCallee(self.to_raw())) };
        a.try_into().map_err(Error::Melior)
    }

    /// Sets the callee attribute.
    fn set_callee(&self, attr: SymbolRefAttribute<'c>) {
        unsafe { llzkFunction_CallOpSetCallee(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the template params attribute.
    fn template_params(&self) -> Result<Option<ArrayAttribute<'c>>, Error> {
        let raw = unsafe { llzkFunction_CallOpGetTemplateParams(self.to_raw()) };
        if raw.ptr.is_null() {
            Ok(None)
        } else {
            ArrayAttribute::try_from(unsafe { Attribute::from_raw(raw) }).map(Some)
        }
    }

    /// Sets the template params attribute.
    fn set_template_params(&self, attr: Option<ArrayAttribute<'c>>) {
        match attr {
            Some(arr) => unsafe {
                llzkFunction_CallOpSetTemplateParams(self.to_raw(), arr.to_raw())
            },
            None => unsafe {
                llzkFunction_CallOpSetTemplateParams(
                    self.to_raw(),
                    mlir_sys::MlirAttribute {
                        ptr: std::ptr::null_mut(),
                    },
                )
            },
        }
    }
}

//===----------------------------------------------------------------------===//
// CallOp, CallOpRef, CallOpRefMut
//===----------------------------------------------------------------------===//

llzk_op_type!(CallOp, llzkOperationIsA_Function_CallOp, "function.call");

impl<'a, 'c: 'a> CallOpLike<'c, 'a> for CallOp<'c> {}

impl<'a, 'c: 'a> CallOpLike<'c, 'a> for CallOpRef<'c, 'a> {}

impl<'a, 'c: 'a> CallOpLike<'c, 'a> for CallOpRefMut<'c, 'a> {}

//===----------------------------------------------------------------------===//
// Operation factories
//===----------------------------------------------------------------------===//

fn tuple_to_named_attr((name, attr): &NamedAttribute) -> MlirNamedAttribute {
    unsafe { mlirNamedAttributeGet(name.to_raw(), attr.to_raw()) }
}

fn prepare_arg_attrs<'c>(
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
    input_count: usize,
    ctx: &'c Context,
) -> Vec<MlirAttribute> {
    log::debug!("prepare_arg_attrs(\n{arg_attrs:?},\n{input_count},\n{ctx:?})");
    let Some(arg_attrs) = arg_attrs else {
        return vec![unsafe { mlirDictionaryAttrGet(ctx.to_raw(), 0, null()) }; input_count];
    };

    assert_eq!(arg_attrs.len(), input_count);
    arg_attrs
        .iter()
        .map(|arg_attr| {
            let named_attrs = Vec::from_iter(arg_attr.iter().map(tuple_to_named_attr));
            unsafe {
                mlirDictionaryAttrGet(
                    ctx.to_raw(),
                    isize::try_from(named_attrs.len()).expect("named_attrs too large"),
                    named_attrs.as_ptr(),
                )
            }
        })
        .collect()
}

/// Creates a 'function.def' operation. If the arg_attrs parameter is None creates as many empty
/// argument attributes as input arguments there are to satisfy the requirement of one
/// DictionaryAttr per argument.
pub fn def<'c>(
    location: Location<'c>,
    name: &str,
    r#type: FunctionType<'c>,
    attrs: &[NamedAttribute<'c>],
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> Result<FuncDefOp<'c>, Error> {
    let ctx = location.context();
    let name = StringRef::new(name);
    let attrs: Vec<_> = attrs.iter().map(tuple_to_named_attr).collect();
    let arg_attrs = prepare_arg_attrs(arg_attrs, r#type.input_count(), unsafe { ctx.to_ref() });
    unsafe {
        Operation::from_raw(llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
            location.to_raw(),
            name.to_raw(),
            r#type.to_raw(),
            isize::try_from(attrs.len()).expect("attrs too large"),
            attrs.as_ptr(),
            isize::try_from(arg_attrs.len()).expect("arg_attrs too large"),
            arg_attrs.as_ptr(),
        ))
    }
    .try_into()
}

/// Return `true` iff the given op is `function.def`.
#[inline]
pub fn is_func_def<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "function.def")
}

/// Creates a new `function.call` operation.
pub fn call<'c>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[Value<'c, '_>],
    return_types: &[impl TypeLike<'c>],
) -> Result<CallOp<'c>, Error> {
    unsafe {
        Operation::from_raw(llzkFunction_CallOpBuild(
            builder.to_raw(),
            location.to_raw(),
            isize::try_from(return_types.len()).expect("return_types too large"),
            return_types.as_ptr() as *const _,
            callee.to_raw(),
            isize::try_from(args.len()).expect("args too large"),
            args.as_ptr() as *const _,
        ))
    }
    .try_into()
}

/// Creates a new `function.call` operation with map operands.
pub fn call_with_map_operands<'c>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[Value<'c, '_>],
    return_types: &[impl TypeLike<'c>],
    map_operands: MapOperandsBuilder,
) -> Result<CallOp<'c>, Error> {
    unsafe {
        Operation::from_raw(llzkFunction_CallOpBuildWithMapOperands(
            builder.to_raw(),
            location.to_raw(),
            isize::try_from(return_types.len()).expect("return_types too large"),
            return_types.as_ptr() as *const _,
            callee.to_raw(),
            map_operands.to_raw(),
            isize::try_from(args.len()).expect("args too large"),
            args.as_ptr() as *const _,
        ))
    }
    .try_into()
}

/// Creates a new `function.call` operation with the optional `templateParams` attribute for
/// calling functions inside `poly.template` regions when template parameters are not bound
/// by the call's argument or result types.
pub fn call_with_template_params<'c>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[Value<'c, '_>],
    return_types: &[impl TypeLike<'c>],
    template_params: &[impl AttributeLike<'c>],
) -> Result<CallOp<'c>, Error> {
    unsafe {
        Operation::from_raw(llzkFunction_CallOpBuildWithTemplateParams(
            builder.to_raw(),
            location.to_raw(),
            isize::try_from(return_types.len()).expect("return_types too large"),
            return_types.as_ptr() as *const _,
            callee.to_raw(),
            isize::try_from(template_params.len()).expect("template_params too large"),
            template_params.as_ptr() as *const _,
            isize::try_from(args.len()).expect("args too large"),
            args.as_ptr() as *const _,
        ))
    }
    .try_into()
}

/// Return `true` iff the given op is `function.call`.
#[inline]
pub fn is_func_call<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "function.call")
}

/// Creates a new `function.return` operation.
///
/// This operation is the terminator op for `function.def` and must be the last operation of the
/// last block in it. The values array must match the number of outputs, and their types, of the
/// parent function.
pub fn r#return<'c>(location: Location<'c>, values: &[Value<'c, '_>]) -> Operation<'c> {
    OperationBuilder::new("function.return", location)
        .add_operands(values)
        .build()
        .unwrap()
}

/// Return `true` iff the given op is `function.return`.
#[inline]
pub fn is_func_return<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "function.return")
}
