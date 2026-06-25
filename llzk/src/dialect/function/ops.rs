use crate::{
    attributes::{NamedAttribute, array::ArrayAttribute},
    builder::OpBuilderLike,
    dialect::r#struct::StructType,
    error::Error,
    macros::llzk_op_type,
    map_operands::MapOperandsBuilder,
    symbol_ref::{SymbolRefAttrLike, SymbolRefAttribute},
};

use llzk_sys::{
    llzkFunction_CallOpBuild, llzkFunction_CallOpBuildWithMapOperands,
    llzkFunction_CallOpBuildWithTemplateParams, llzkFunction_CallOpCalleeIsCompute,
    llzkFunction_CallOpCalleeIsConstrain, llzkFunction_CallOpCalleeIsProduct,
    llzkFunction_CallOpCalleeIsStructCompute, llzkFunction_CallOpCalleeIsStructConstrain,
    llzkFunction_CallOpCalleeIsStructProduct, llzkFunction_CallOpGetArgOperandsAt,
    llzkFunction_CallOpGetArgOperandsCount, llzkFunction_CallOpGetCallee,
    llzkFunction_CallOpGetMapOperandsAt, llzkFunction_CallOpGetMapOperandsCount,
    llzkFunction_CallOpGetSelfValueFromCompute, llzkFunction_CallOpGetSelfValueFromConstrain,
    llzkFunction_CallOpGetTemplateParams, llzkFunction_CallOpSetArgOperands,
    llzkFunction_CallOpSetCallee, llzkFunction_CallOpSetMapOperands,
    llzkFunction_CallOpSetTemplateParams, llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs,
    llzkFunction_FuncDefOpGetArgAttrs, llzkFunction_FuncDefOpGetBody,
    llzkFunction_FuncDefOpGetFullyQualifiedName, llzkFunction_FuncDefOpGetFunctionType,
    llzkFunction_FuncDefOpGetResAttrs,
    llzkFunction_FuncDefOpGetSelfValueFromCompute, llzkFunction_FuncDefOpGetSelfValueFromConstrain,
    llzkFunction_FuncDefOpGetSingleResultTypeOfCompute, llzkFunction_FuncDefOpGetSymName,
    llzkFunction_FuncDefOpHasAllowConstraintAttr,
    llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr, llzkFunction_FuncDefOpHasAllowWitnessAttr,
    llzkFunction_FuncDefOpHasArgPublicAttr, llzkFunction_FuncDefOpIsDeclaration,
    llzkFunction_FuncDefOpIsInStruct, llzkFunction_FuncDefOpIsStructCompute,
    llzkFunction_FuncDefOpIsStructConstrain, llzkFunction_FuncDefOpIsStructProduct,
    llzkFunction_FuncDefOpNameIsCompute, llzkFunction_FuncDefOpNameIsConstrain,
    llzkFunction_FuncDefOpNameIsProduct, llzkFunction_FuncDefOpSetAllowConstraintAttr,
    llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr, llzkFunction_FuncDefOpSetAllowWitnessAttr,
    llzkFunction_FuncDefOpSetArgAttrs, llzkFunction_FuncDefOpSetFunctionType,
    llzkFunction_FuncDefOpSetResAttrs, llzkFunction_FuncDefOpSetSymName,
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
        Identifier,
    },
};
use mlir_sys::{
    MlirAttribute, MlirNamedAttribute, mlirDictionaryAttrGet, mlirDictionaryAttrGetElement,
    mlirDictionaryAttrGetElementByName, mlirDictionaryAttrGetNumElements, mlirNamedAttributeGet,
};

use std::ptr::null;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Builds the standard out-of-bounds error used by `function.def` helpers.
fn create_out_of_bounds_error<'c: 'a, 'a>(
    func: &(impl FuncDefOpLike<'c, 'a> + ?Sized),
    idx: usize,
) -> Error {
    let fqn = func.fully_qualified_name();
    Error::OutOfBoundsArgument(Some(fqn.to_string()), idx)
}

/// Selects whether helper methods operate on function arguments or results.
#[derive(Debug, Copy, Clone)]
pub enum SignaturePosition {
    /// Select the function inputs.
    Argument,
    /// Select the function results.
    Result,
}

impl SignaturePosition {
    fn label(self) -> &'static str {
        match self {
            SignaturePosition::Argument => "argument",
            SignaturePosition::Result => "result",
        }
    }
}

/// Creates an empty dictionary attribute in the provided context.
fn empty_dictionary_attr(context: &Context) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirDictionaryAttrGet(context.to_raw(), 0, null())) }
}

/// Converts a slice of named attributes into a dictionary attribute.
fn named_attributes_to_dictionary_attr<'c>(
    context: &'c Context,
    attrs: &[NamedAttribute<'c>],
) -> Attribute<'c> {
    let named_attrs: Vec<_> = attrs.iter().map(tuple_to_named_attr).collect();
    unsafe {
        Attribute::from_raw(mlirDictionaryAttrGet(
            context.to_raw(),
            isize::try_from(named_attrs.len()).expect("named_attrs too large"),
            named_attrs.as_ptr(),
        ))
    }
}

/// Expands a dictionary attribute back into its `(Identifier, Attribute)` pairs.
fn dictionary_attr_entries<'c>(attr: Attribute<'c>) -> Vec<NamedAttribute<'c>> {
    (0..unsafe { mlirDictionaryAttrGetNumElements(attr.to_raw()) })
        .map(|idx| unsafe { mlirDictionaryAttrGetElement(attr.to_raw(), idx) })
        .map(|attr| unsafe {
            (
                Identifier::from_raw(attr.name),
                Attribute::from_raw(attr.attribute),
            )
        })
        .collect()
}

//===----------------------------------------------------------------------===//
// FuncDefOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the 'function.def' op.
pub trait FuncDefOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the raw attribute array attached to either the function arguments or results.
    fn get_signature_attrs(
        &self,
        position: SignaturePosition,
    ) -> Result<ArrayAttribute<'c>, Error> {
        let raw = match position {
            SignaturePosition::Argument => unsafe { llzkFunction_FuncDefOpGetArgAttrs(self.to_raw()) },
            SignaturePosition::Result => unsafe { llzkFunction_FuncDefOpGetResAttrs(self.to_raw()) },
        };
        let attr = unsafe { Attribute::from_option_raw(raw) }.ok_or_else(|| {
            Error::AttributeNotFound(format!("function.def {} attributes", position.label()))
        })?;
        ArrayAttribute::try_from(attr)
    }

    /// Replaces the raw attribute array attached to either the function arguments or results.
    fn set_signature_attrs(&self, position: SignaturePosition, attr: ArrayAttribute<'c>) {
        match position {
            SignaturePosition::Argument => unsafe {
                llzkFunction_FuncDefOpSetArgAttrs(self.to_raw(), attr.to_raw())
            },
            SignaturePosition::Result => unsafe {
                llzkFunction_FuncDefOpSetResAttrs(self.to_raw(), attr.to_raw())
            },
        }
    }

    /// Returns the number of entries in the selected portion of the function signature.
    fn signature_count(&self, position: SignaturePosition) -> Result<usize, Error> {
        let ty = self.function_type()?;
        Ok(match position {
            SignaturePosition::Argument => ty.input_count(),
            SignaturePosition::Result => ty.result_count(),
        })
    }

    /// Looks up a named attribute on the selected argument or result position.
    fn get_signature_named_attr(
        &self,
        position: SignaturePosition,
        idx: usize,
        name: &str,
    ) -> Result<Option<Attribute<'c>>, Error> {
        let count = self.signature_count(position)?;
        if idx >= count {
            return Err(create_out_of_bounds_error(self, idx));
        }

        let attrs = match self.get_signature_attrs(position) {
            Ok(attrs) => attrs,
            Err(Error::AttributeNotFound(_)) => return Ok(None),
            Err(err) => return Err(err),
        };
        let dict = attrs.get(idx).expect("signature attrs length should match arity");
        let raw = unsafe {
            mlirDictionaryAttrGetElementByName(dict.to_raw(), StringRef::new(name).to_raw())
        };
        Ok(unsafe { Attribute::from_option_raw(raw) })
    }

    /// Sets a named attribute on the selected argument or result position.
    fn set_signature_named_attr(
        &self,
        position: SignaturePosition,
        idx: usize,
        name: Identifier<'c>,
        attr: Attribute<'c>,
    ) -> Result<(), Error> {
        let count = self.signature_count(position)?;
        if idx >= count {
            return Err(create_out_of_bounds_error(self, idx));
        }

        let context = unsafe { self.context().to_ref() };
        let mut dicts: Vec<_> = match self.get_signature_attrs(position) {
            Ok(attrs) => attrs.into_iter().collect(),
            Err(Error::AttributeNotFound(_)) => vec![empty_dictionary_attr(context); count],
            Err(err) => return Err(err),
        };
        if dicts.len() < count {
            dicts.resize(count, empty_dictionary_attr(context));
        }

        let mut entries = dictionary_attr_entries(dicts[idx]);
        if let Some(existing) = entries.iter_mut().find(|(existing_name, _)| *existing_name == name)
        {
            existing.1 = attr;
        } else {
            entries.push((name, attr));
        }
        dicts[idx] = named_attributes_to_dictionary_attr(context, &entries);
        self.set_signature_attrs(position, ArrayAttribute::new(context, &dicts));
        Ok(())
    }

    /// Returns the argument attribute array.
    fn arg_attrs(&self) -> Result<ArrayAttribute<'c>, Error> {
        self.get_signature_attrs(SignaturePosition::Argument)
    }

    /// Sets the argument attribute array.
    fn set_arg_attrs(&self, attr: ArrayAttribute<'c>) {
        self.set_signature_attrs(SignaturePosition::Argument, attr)
    }

    /// Returns the result attribute array.
    fn res_attrs(&self) -> Result<ArrayAttribute<'c>, Error> {
        self.get_signature_attrs(SignaturePosition::Result)
    }

    /// Sets the result attribute array.
    fn set_res_attrs(&self, attr: ArrayAttribute<'c>) {
        self.set_signature_attrs(SignaturePosition::Result, attr)
    }

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

    /// Returns true if the `idx`-th argument has a `function.arg_name` attribute.
    fn has_arg_name(&self, idx: u32) -> bool {
        matches!(self.arg_name_attr(idx as usize), Ok(Some(_)))
    }

    /// Returns the `function.arg_name` attribute for the `idx`-th argument.
    fn arg_name_attr(&self, idx: usize) -> Result<Option<StringAttribute<'c>>, Error> {
        self.get_signature_named_attr(SignaturePosition::Argument, idx, "function.arg_name")?
            .map(StringAttribute::try_from)
            .transpose()
            .map_err(Error::Melior)
    }

    /// Sets the `function.arg_name` attribute for the `idx`-th argument.
    fn set_arg_name_attr(&self, idx: usize, attr: StringAttribute<'c>) -> Result<(), Error> {
        self.set_signature_named_attr(
            SignaturePosition::Argument,
            idx,
            Identifier::new(unsafe { self.context().to_ref() }, "function.arg_name"),
            attr.into(),
        )
    }

    /// Sets the `function.arg_name` attribute for the `idx`-th argument from a string.
    fn set_arg_name(&self, idx: usize, name: &str) -> Result<(), Error> {
        self.set_arg_name_attr(idx, StringAttribute::new(unsafe { self.context().to_ref() }, name))
    }

    /// Returns true if the `idx`-th result has a `function.res_name` attribute.
    fn has_res_name(&self, idx: u32) -> bool {
        matches!(self.res_name_attr(idx as usize), Ok(Some(_)))
    }

    /// Returns the `function.res_name` attribute for the `idx`-th result.
    fn res_name_attr(&self, idx: usize) -> Result<Option<StringAttribute<'c>>, Error> {
        self.get_signature_named_attr(SignaturePosition::Result, idx, "function.res_name")?
            .map(StringAttribute::try_from)
            .transpose()
            .map_err(Error::Melior)
    }

    /// Sets the `function.res_name` attribute for the `idx`-th result.
    fn set_res_name_attr(&self, idx: usize, attr: StringAttribute<'c>) -> Result<(), Error> {
        self.set_signature_named_attr(
            SignaturePosition::Result,
            idx,
            Identifier::new(unsafe { self.context().to_ref() }, "function.res_name"),
            attr.into(),
        )
    }

    /// Sets the `function.res_name` attribute for the `idx`-th result from a string.
    fn set_res_name(&self, idx: usize, name: &str) -> Result<(), Error> {
        self.set_res_name_attr(idx, StringAttribute::new(unsafe { self.context().to_ref() }, name))
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

    /// Returns true if the function's name is [`FUNC_NAME_PRODUCT`](llzk_sys::FUNC_NAME_PRODUCT).
    fn name_is_product(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpNameIsProduct(self.to_raw()) }
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

    /// Returns true if the function is the struct's product function.
    fn is_struct_product(&self) -> bool {
        unsafe { llzkFunction_FuncDefOpIsStructProduct(self.to_raw()) }
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
        self.get_signature_named_attr(SignaturePosition::Argument, idx, name)?
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

    /// Returns true if the call target name is [`FUNC_NAME_PRODUCT`](llzk_sys::FUNC_NAME_PRODUCT).
    fn callee_is_product(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsProduct(self.to_raw()) }
    }

    /// Return `true` iff the callee function name is [`FUNC_NAME_COMPUTE`](llzk_sys::FUNC_NAME_COMPUTE) within a StructDefOp.
    fn callee_is_struct_compute(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsStructCompute(self.to_raw()) }
    }

    /// Return `true` iff the callee function name is [`FUNC_NAME_CONSTRAIN`](llzk_sys::FUNC_NAME_CONSTRAIN) within a StructDefOp.
    fn callee_is_struct_constrain(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsStructConstrain(self.to_raw()) }
    }

    /// Return `true` iff the callee function name is [`FUNC_NAME_PRODUCT`](llzk_sys::FUNC_NAME_PRODUCT) within a StructDefOp.
    fn callee_is_struct_product(&self) -> bool {
        unsafe { llzkFunction_CallOpCalleeIsStructProduct(self.to_raw()) }
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

/// Converts a Rust named-attribute tuple to the raw C API representation.
fn tuple_to_named_attr((name, attr): &NamedAttribute) -> MlirNamedAttribute {
    unsafe { mlirNamedAttributeGet(name.to_raw(), attr.to_raw()) }
}

/// Builds one dictionary attribute per function argument.
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
        .map(|arg_attr| named_attributes_to_dictionary_attr(ctx, arg_attr).to_raw())
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

/// Creates a `function.def` operation and optionally sets both argument and result attributes.
pub fn def_with_signature_attrs<'c>(
    location: Location<'c>,
    name: &str,
    r#type: FunctionType<'c>,
    attrs: &[NamedAttribute<'c>],
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
    res_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> Result<FuncDefOp<'c>, Error> {
    let context = unsafe { location.context().to_ref() };
    let op = def(location, name, r#type, attrs, arg_attrs)?;
    if let Some(arg_attrs) = arg_attrs {
        let attr = ArrayAttribute::new(
            context,
            &prepare_arg_attrs(Some(arg_attrs), r#type.input_count(), context)
                .into_iter()
                .map(|attr| unsafe { Attribute::from_raw(attr) })
                .collect::<Vec<_>>(),
        );
        op.set_arg_attrs(attr);
    }
    if let Some(res_attrs) = res_attrs {
        let attr = ArrayAttribute::new(
            context,
            &prepare_arg_attrs(Some(res_attrs), r#type.result_count(), context)
                .into_iter()
                .map(|attr| unsafe { Attribute::from_raw(attr) })
                .collect::<Vec<_>>(),
        );
        op.set_res_attrs(attr);
    }
    Ok(op)
}

/// Return `true` iff the given op is `function.def`.
#[inline]
pub fn is_func_def<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "function.def")
}

/// Creates a new `function.call` operation.
pub fn call<'c>(
    builder: &impl OpBuilderLike<'c>,
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
    builder: &impl OpBuilderLike<'c>,
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
    builder: &impl OpBuilderLike<'c>,
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
