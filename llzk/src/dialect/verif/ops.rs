//! `verif` dialect ops.

use llzk_sys::{
    llzkOperationIsA_Verif_ContractOp, llzkOperationIsA_Verif_EnsureComputeOp,
    llzkOperationIsA_Verif_EnsureConstrainOp, llzkOperationIsA_Verif_IncludeOp,
    llzkOperationIsA_Verif_RequireComputeOp, llzkOperationIsA_Verif_RequireConstrainOp,
    llzkVerif_ContractOpBuildFromTarget, llzkVerif_ContractOpGetArgAttrs,
    llzkVerif_ContractOpGetBody, llzkVerif_ContractOpGetCallableRegion,
    llzkVerif_ContractOpGetFullyQualifiedName, llzkVerif_ContractOpGetFunctionType,
    llzkVerif_ContractOpGetSymName, llzkVerif_ContractOpGetTarget, llzkVerif_ContractOpHasArgName,
    llzkVerif_ContractOpHasArgPublicAttr, llzkVerif_ContractOpHasFuncTarget,
    llzkVerif_ContractOpHasStructTarget, llzkVerif_ContractOpIsDeclaration,
    llzkVerif_ContractOpSetArgAttrs, llzkVerif_ContractOpSetFunctionType,
    llzkVerif_ContractOpSetSymName, llzkVerif_ContractOpSetTarget, llzkVerif_EnsureComputeOpBuild,
    llzkVerif_EnsureComputeOpGetCondition, llzkVerif_EnsureComputeOpSetCondition,
    llzkVerif_EnsureConstrainOpBuild, llzkVerif_EnsureConstrainOpGetCondition,
    llzkVerif_EnsureConstrainOpSetCondition, llzkVerif_IncludeOpBuild,
    llzkVerif_IncludeOpContractTargetsStruct, llzkVerif_IncludeOpGetArgOperandsAt,
    llzkVerif_IncludeOpGetArgOperandsCount, llzkVerif_IncludeOpGetCallee,
    llzkVerif_IncludeOpGetMapOpGroupSizes, llzkVerif_IncludeOpGetMapOperandsAt,
    llzkVerif_IncludeOpGetMapOperandsCount, llzkVerif_IncludeOpGetNumDimsPerMap,
    llzkVerif_IncludeOpGetSelfValue, llzkVerif_IncludeOpGetTemplateParams,
    llzkVerif_IncludeOpGetTypeSignature, llzkVerif_IncludeOpResolveCallable,
    llzkVerif_IncludeOpSetArgOperands, llzkVerif_IncludeOpSetCallee,
    llzkVerif_IncludeOpSetMapOpGroupSizes, llzkVerif_IncludeOpSetMapOperands,
    llzkVerif_IncludeOpSetNumDimsPerMap, llzkVerif_IncludeOpSetTemplateParams,
    llzkVerif_RequireComputeOpBuild, llzkVerif_RequireComputeOpGetCondition,
    llzkVerif_RequireComputeOpSetCondition, llzkVerif_RequireConstrainOpBuild,
    llzkVerif_RequireConstrainOpGetCondition, llzkVerif_RequireConstrainOpSetCondition,
};
use melior::{
    Context,
    ir::{
        Attribute, AttributeLike, BlockLike as _, Identifier, Location, Operation, OperationRef,
        RegionLike as _, RegionRef, Type, ValueLike as _,
        attribute::{DenseI32ArrayAttribute, StringAttribute, TypeAttribute},
        block::{Block, BlockArgument},
        operation::OperationLike,
        r#type::FunctionType,
    },
};

use crate::{
    attributes::{array::ArrayAttribute, null_attr, rebuild_array_attr},
    builder::{OpBuilder, OpBuilderLike},
    error::Error,
    macros::llzk_op_type,
    symbol_ref::{SymbolRefAttrLike, SymbolRefAttribute},
    type_ext::FunctionTypeExt as _,
    value_ext::ValueRange,
};

fn create_out_of_bounds_error<'c: 'a, 'a>(
    contract: &(impl ContractOpLike<'c, 'a> + ?Sized),
    idx: usize,
) -> Error {
    match SymbolRefAttribute::try_from(contract.fully_qualified_name()) {
        Ok(fqn) => Error::OutOfBoundsArgument(Some(fqn.to_string()), idx),
        Err(err) => err.into(),
    }
}

//===----------------------------------------------------------------------===//
// ContractOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `verif.contract` op.
pub trait ContractOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the name of the contract.
    ///
    /// # Panics
    ///
    /// If the `verif.contract` op does not have a `sym_name` attribute.
    fn name(&'a self) -> &'c str {
        self.get_sym_name().map(|attr| attr.value()).unwrap()
    }

    /// Returns the sym_name attribute.
    fn get_sym_name(&self) -> Result<StringAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_ContractOpGetSymName(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the sym_name attribute.
    fn set_sym_name(&self, attr: StringAttribute<'c>) {
        unsafe { llzkVerif_ContractOpSetSymName(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the target attribute.
    fn target(&self) -> Result<SymbolRefAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_ContractOpGetTarget(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the target attribute.
    fn set_target(&self, attr: SymbolRefAttribute<'c>) {
        unsafe { llzkVerif_ContractOpSetTarget(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the function type of the contract.
    fn get_function_type(&self) -> Result<FunctionType<'c>, Error> {
        let attr =
            unsafe { Attribute::from_raw(llzkVerif_ContractOpGetFunctionType(self.to_raw())) };
        let type_attr: TypeAttribute<'c> = attr.try_into()?;
        type_attr.value().try_into().map_err(Error::Melior)
    }

    /// Sets the function type.
    fn set_function_type(&self, ty: FunctionType<'c>) {
        let type_attr = TypeAttribute::new(ty.into());
        unsafe { llzkVerif_ContractOpSetFunctionType(self.to_raw(), type_attr.to_raw()) }
    }

    /// Returns the argument attribute array.
    fn get_arg_attrs(&self) -> Result<ArrayAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_ContractOpGetArgAttrs(self.to_raw())) };
        Ok(rebuild_array_attr(unsafe { self.context().to_ref() }, attr))
    }

    /// Sets the argument attribute array.
    fn set_arg_attrs(&self, attr: ArrayAttribute<'c>) {
        unsafe { llzkVerif_ContractOpSetArgAttrs(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the body region of the contract.
    fn get_body(&self) -> Result<RegionRef<'c, 'a>, Error> {
        let raw = unsafe { llzkVerif_ContractOpGetBody(self.to_raw()) };
        if raw.ptr.is_null() {
            Err(Error::GeneralError(
                "no body in a declaration-only contract",
            ))
        } else {
            Ok(unsafe { RegionRef::from_raw(raw) })
        }
    }

    /// Returns true if the argument at `idx` has the `pub` attribute.
    fn arg_is_pub(&self, idx: u32) -> bool {
        unsafe { llzkVerif_ContractOpHasArgPublicAttr(self.to_raw(), idx) }
    }

    /// Returns true if the argument at `idx` has the `function.arg_name` attribute.
    fn has_arg_name(&self, idx: u32) -> bool {
        unsafe { llzkVerif_ContractOpHasArgName(self.to_raw(), idx) }
    }

    /// Returns true if the contract targets a function.
    fn has_func_target(&self) -> bool {
        unsafe { llzkVerif_ContractOpHasFuncTarget(self.to_raw()) }
    }

    /// Returns true if the contract targets a struct.
    fn has_struct_target(&self) -> bool {
        unsafe { llzkVerif_ContractOpHasStructTarget(self.to_raw()) }
    }

    /// Returns true if the contract has no body.
    fn is_declaration(&self) -> bool {
        unsafe { llzkVerif_ContractOpIsDeclaration(self.to_raw()) }
    }

    /// Returns the callable region for the contract.
    fn get_callable_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkVerif_ContractOpGetCallableRegion(self.to_raw())) }
    }

    /// Returns the fully qualified name of the contract.
    fn fully_qualified_name(&self) -> Attribute<'c> {
        unsafe {
            Attribute::from_raw(llzkVerif_ContractOpGetFullyQualifiedName(
                self.to_raw(),
                false,
            ))
        }
    }

    /// Returns the n-th argument of the contract.
    fn argument(&self, idx: usize) -> Result<BlockArgument<'c, 'a>, Error> {
        self.get_body()
            .and_then(|region| {
                region
                    .first_block()
                    .ok_or(create_out_of_bounds_error(self, idx))
            })
            .and_then(|block| block.argument(idx).map_err(Into::into))
    }
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

llzk_op_type!(
    ContractOp,
    llzkOperationIsA_Verif_ContractOp,
    "verif.contract"
);

impl<'a, 'c: 'a> ContractOpLike<'c, 'a> for ContractOp<'c> {}
impl<'a, 'c: 'a> ContractOpLike<'c, 'a> for ContractOpRef<'c, 'a> {}
impl<'a, 'c: 'a> ContractOpLike<'c, 'a> for ContractOpRefMut<'c, 'a> {}

/// Creates a `verif.contract` op using the builder that infers argument and type
/// information from the target symbol. Also inserts an entry block into the
/// contract region for convenience.
pub fn contract<'c, 'a>(
    builder: &'a impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: &str,
    target: &str,
) -> Result<ContractOp<'c>, Error> {
    let context = location.context();
    let op = unsafe {
        ContractOp::from_raw(llzkVerif_ContractOpBuildFromTarget(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(context.to_ref(), name).to_raw(),
            Identifier::new(context.to_ref(), target).to_raw(),
        ))
    };
    if let Ok(region) = op.get_body()
        && region.first_block().is_none()
    {
        let args = op
            .get_function_type()?
            .inputs()
            .map(|ty| (ty, location))
            .collect::<Vec<(Type<'c>, Location<'c>)>>();
        region.append_block(Block::new(&args));
    }
    Ok(op)
}

/// Returns `true` iff the given op is `verif.contract`.
#[inline]
pub fn is_contract<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "verif.contract")
}

//===----------------------------------------------------------------------===//
// IncludeOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `verif.include` op.
pub trait IncludeOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the number of call operands.
    fn arg_operand_count(&self) -> usize {
        let n = unsafe { llzkVerif_IncludeOpGetArgOperandsCount(self.to_raw()) };
        usize::try_from(n).expect("size is negative or too large")
    }

    /// Returns the call operand at `index`.
    fn arg_operand_at(&self, index: usize) -> melior::ir::Value<'c, 'a> {
        let index = isize::try_from(index).expect("index too large");
        unsafe {
            melior::ir::Value::from_raw(llzkVerif_IncludeOpGetArgOperandsAt(self.to_raw(), index))
        }
    }

    /// Sets the call operands.
    fn set_arg_operands(&self, values: &[melior::ir::Value<'c, '_>]) {
        unsafe {
            llzkVerif_IncludeOpSetArgOperands(
                self.to_raw(),
                isize::try_from(values.len()).expect("values too large"),
                values.as_ptr() as *const _,
            )
        }
    }

    /// Returns the number of flattened map operands.
    fn map_operand_count(&self) -> usize {
        let n = unsafe { llzkVerif_IncludeOpGetMapOperandsCount(self.to_raw()) };
        usize::try_from(n).expect("size is negative or too large")
    }

    /// Returns the flattened map operand at `index`.
    fn map_operand_at(&self, index: usize) -> melior::ir::Value<'c, 'a> {
        let index = isize::try_from(index).expect("index too large");
        unsafe {
            melior::ir::Value::from_raw(llzkVerif_IncludeOpGetMapOperandsAt(self.to_raw(), index))
        }
    }

    /// Sets grouped map operands. Each [`ValueRange`] corresponds to one group.
    fn set_map_operands<'g, 'v>(&self, groups: &'g [ValueRange<'c, 'v, '_>]) {
        let raw_groups: Vec<_> = groups.iter().map(ValueRange::to_raw).collect();
        unsafe {
            llzkVerif_IncludeOpSetMapOperands(
                self.to_raw(),
                isize::try_from(raw_groups.len()).expect("group count too large"),
                raw_groups.as_ptr(),
            )
        }
    }

    /// Returns the callee attribute.
    fn get_callee(&self) -> Result<SymbolRefAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_IncludeOpGetCallee(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the callee attribute.
    fn set_callee(&self, attr: impl SymbolRefAttrLike<'c>) {
        unsafe { llzkVerif_IncludeOpSetCallee(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the template parameter attribute, if present.
    fn get_template_params(&self) -> Result<Option<ArrayAttribute<'c>>, Error> {
        let raw = unsafe { llzkVerif_IncludeOpGetTemplateParams(self.to_raw()) };
        if raw.ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(rebuild_array_attr(
                unsafe { self.context().to_ref() },
                unsafe { Attribute::from_raw(raw) },
            )))
        }
    }

    /// Sets the template parameter attribute.
    fn set_template_params(&self, attr: Option<ArrayAttribute<'c>>) {
        let raw = attr.map_or_else(null_attr, |attr| attr.to_raw());
        unsafe { llzkVerif_IncludeOpSetTemplateParams(self.to_raw(), raw) }
    }

    /// Returns the `numDimsPerMap` attribute.
    fn get_num_dims_per_map(&self) -> Result<DenseI32ArrayAttribute<'c>, Error> {
        let attr =
            unsafe { Attribute::from_raw(llzkVerif_IncludeOpGetNumDimsPerMap(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the `numDimsPerMap` attribute.
    fn set_num_dims_per_map(&self, attr: DenseI32ArrayAttribute<'c>) {
        unsafe { llzkVerif_IncludeOpSetNumDimsPerMap(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the `mapOpGroupSizes` attribute.
    fn get_map_op_group_sizes(&self) -> Result<DenseI32ArrayAttribute<'c>, Error> {
        let attr =
            unsafe { Attribute::from_raw(llzkVerif_IncludeOpGetMapOpGroupSizes(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the `mapOpGroupSizes` attribute.
    fn set_map_op_group_sizes(&self, attr: DenseI32ArrayAttribute<'c>) {
        unsafe { llzkVerif_IncludeOpSetMapOpGroupSizes(self.to_raw(), attr.to_raw()) }
    }

    /// Returns true iff the callee contract targets a struct.
    fn contract_targets_struct(&self) -> bool {
        unsafe { llzkVerif_IncludeOpContractTargetsStruct(self.to_raw()) }
    }

    /// Returns the `self` value when the callee contract targets a struct.
    fn self_value(&self) -> Result<melior::ir::Value<'c, 'a>, Error> {
        if self.contract_targets_struct() {
            Ok(unsafe {
                melior::ir::Value::from_raw(llzkVerif_IncludeOpGetSelfValue(self.to_raw()))
            })
        } else {
            Err(Error::GeneralError(
                "include op callee does not target a struct contract",
            ))
        }
    }

    /// Returns the inferred type signature for the include op.
    fn type_signature(&self) -> Result<FunctionType<'c>, Error> {
        let ty = unsafe { Type::from_raw(llzkVerif_IncludeOpGetTypeSignature(self.to_raw())) };
        ty.try_into().map_err(Error::Melior)
    }

    /// Resolves the callable target, if it can be found.
    fn resolve_callable(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(llzkVerif_IncludeOpResolveCallable(self.to_raw())) }
    }
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

llzk_op_type!(IncludeOp, llzkOperationIsA_Verif_IncludeOp, "verif.include");

impl<'a, 'c: 'a> IncludeOpLike<'c, 'a> for IncludeOp<'c> {}
impl<'a, 'c: 'a> IncludeOpLike<'c, 'a> for IncludeOpRef<'c, 'a> {}
impl<'a, 'c: 'a> IncludeOpLike<'c, 'a> for IncludeOpRefMut<'c, 'a> {}

/// Creates a `verif.include` op with only direct call operands.
pub fn include<'c>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
) -> Result<IncludeOp<'c>, Error> {
    let template_params = template_params.map_or_else(null_attr, |attr| attr.to_raw());
    unsafe {
        Operation::from_raw(llzkVerif_IncludeOpBuild(
            builder.to_raw(),
            location.to_raw(),
            callee.to_raw(),
            llzk_sys::MlirValueRange {
                values: args.as_ptr() as *const _,
                size: isize::try_from(args.len()).expect("args too large"),
            },
            template_params,
        ))
    }
    .try_into()
}

/// Creates a `verif.include` op with grouped map operands and explicit
/// `numDimsPerMap` metadata.
pub fn include_with_map_operands<'c, 'g, 'v>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
    map_operands: &'g [ValueRange<'c, 'v, '_>],
    num_dims_per_map: DenseI32ArrayAttribute<'c>,
) -> Result<IncludeOp<'c>, Error> {
    let op = include(builder, location, callee, args, template_params)?;
    op.set_map_operands(map_operands);
    op.set_num_dims_per_map(num_dims_per_map);
    Ok(op)
}

/// Creates a `verif.include` op with grouped map operands and dimension counts
/// provided as a Rust slice.
///
/// TODO: There's a weird segfault that occurs when you use the `unsafe { location.context().to_ref() }`
/// pattern specifically in conjunction with `DenseI32ArrayAttribute::new`. To bypass
/// this, we take the `context` parameter explicitly from the caller.
pub fn include_with_map_operands_slice<'c, 'g, 'v>(
    context: &'c Context,
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
    map_operands: &'g [ValueRange<'c, 'v, '_>],
    num_dims_per_map: &[i32],
) -> Result<IncludeOp<'c>, Error> {
    include_with_map_operands(
        builder,
        location,
        callee,
        args,
        template_params,
        map_operands,
        DenseI32ArrayAttribute::new(context, num_dims_per_map),
    )
}

/// Returns `true` iff the given op is `verif.include`.
#[inline]
pub fn is_include<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "verif.include")
}

//===----------------------------------------------------------------------===//
// ConditionOpLike
//===----------------------------------------------------------------------===//

/// Shared accessors for `verif.require_*` and `verif.ensure_*` ops.
pub trait ConditionOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the raw condition operand.
    fn condition_raw(&self) -> melior::ir::Value<'c, 'a>;
    /// Sets the raw condition operand.
    fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>);

    /// Returns the boolean condition operand.
    fn condition(&self) -> melior::ir::Value<'c, 'a> {
        self.condition_raw()
    }

    /// Sets the boolean condition operand.
    fn set_condition(&self, value: melior::ir::Value<'c, '_>) {
        self.set_condition_raw(value)
    }
}

macro_rules! impl_condition_op {
    ($type:ident, $isa:ident, $name:literal, $build:path, $get:path, $set:path, $pred:ident, $ctor:ident) => {
        llzk_op_type!($type, $isa, $name);

        impl<'a, 'c: 'a> ConditionOpLike<'c, 'a> for $type<'c> {
            fn condition_raw(&self) -> melior::ir::Value<'c, 'a> {
                unsafe { melior::ir::Value::from_raw($get(self.to_raw())) }
            }

            fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>) {
                unsafe { $set(self.to_raw(), value.to_raw()) }
            }
        }

        paste::paste! {
            impl<'a, 'c: 'a> ConditionOpLike<'c, 'a> for [<$type Ref>]<'c, 'a> {
                fn condition_raw(&self) -> melior::ir::Value<'c, 'a> {
                    unsafe { melior::ir::Value::from_raw($get(self.to_raw())) }
                }

                fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>) {
                    unsafe { $set(self.to_raw(), value.to_raw()) }
                }
            }

            impl<'a, 'c: 'a> ConditionOpLike<'c, 'a> for [<$type RefMut>]<'c, 'a> {
                fn condition_raw(&self) -> melior::ir::Value<'c, 'a> {
                    unsafe { melior::ir::Value::from_raw($get(self.to_raw())) }
                }

                fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>) {
                    unsafe { $set(self.to_raw(), value.to_raw()) }
                }
            }
        }

        #[doc = concat!("Creates a `", $name, "` op.")]
        pub fn $ctor<'c>(
            builder: &OpBuilder<'c>,
            location: Location<'c>,
            condition: impl melior::ir::ValueLike<'c>,
        ) -> Result<$type<'c>, Error> {
            unsafe {
                Operation::from_raw($build(
                    builder.to_raw(),
                    location.to_raw(),
                    condition.to_raw(),
                ))
            }
            .try_into()
        }

        #[inline]
        #[doc = concat!("Returns `true` iff the given op is `", $name, "`.")]
        pub fn $pred<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
            crate::operation::isa(op, $name)
        }
    };
}

//===----------------------------------------------------------------------===//
// Condition Ops
//===----------------------------------------------------------------------===//

impl_condition_op!(
    EnsureComputeOp,
    llzkOperationIsA_Verif_EnsureComputeOp,
    "verif.ensure_compute",
    llzkVerif_EnsureComputeOpBuild,
    llzkVerif_EnsureComputeOpGetCondition,
    llzkVerif_EnsureComputeOpSetCondition,
    is_ensure_compute,
    ensure_compute
);

impl_condition_op!(
    EnsureConstrainOp,
    llzkOperationIsA_Verif_EnsureConstrainOp,
    "verif.ensure_constrain",
    llzkVerif_EnsureConstrainOpBuild,
    llzkVerif_EnsureConstrainOpGetCondition,
    llzkVerif_EnsureConstrainOpSetCondition,
    is_ensure_constrain,
    ensure_constrain
);

impl_condition_op!(
    RequireComputeOp,
    llzkOperationIsA_Verif_RequireComputeOp,
    "verif.require_compute",
    llzkVerif_RequireComputeOpBuild,
    llzkVerif_RequireComputeOpGetCondition,
    llzkVerif_RequireComputeOpSetCondition,
    is_require_compute,
    require_compute
);

impl_condition_op!(
    RequireConstrainOp,
    llzkOperationIsA_Verif_RequireConstrainOp,
    "verif.require_constrain",
    llzkVerif_RequireConstrainOpBuild,
    llzkVerif_RequireConstrainOpGetCondition,
    llzkVerif_RequireConstrainOpSetCondition,
    is_require_constrain,
    require_constrain
);
