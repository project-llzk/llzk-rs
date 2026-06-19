//! `verif` dialect ops.

use llzk_sys::{
    llzkOperationIsA_Verif_ContractOp, llzkOperationIsA_Verif_EnsureComputeOp,
    llzkOperationIsA_Verif_EnsureConstrainOp, llzkOperationIsA_Verif_IncludeOp,
    llzkOperationIsA_Verif_InvariantOp, llzkOperationIsA_Verif_RequireComputeOp,
    llzkOperationIsA_Verif_RequireConstrainOp, llzkVerif_ContractOpBuildFromTargetAttr,
    llzkVerif_ContractOpGetArgAttrs, llzkVerif_ContractOpGetBody,
    llzkVerif_ContractOpGetCallableRegion, llzkVerif_ContractOpGetFullyQualifiedName,
    llzkVerif_ContractOpGetFunctionType, llzkVerif_ContractOpGetSymName,
    llzkVerif_ContractOpGetTarget, llzkVerif_ContractOpHasArgName,
    llzkVerif_ContractOpHasArgPublicAttr, llzkVerif_ContractOpHasFuncTarget,
    llzkVerif_ContractOpHasStructTarget, llzkVerif_ContractOpIsDeclaration,
    llzkVerif_ContractOpSetArgAttrs, llzkVerif_ContractOpSetFunctionType,
    llzkVerif_ContractOpSetSymName, llzkVerif_ContractOpSetTarget, llzkVerif_DecreasesOpBuild,
    llzkVerif_EnsureComputeOpBuild, llzkVerif_EnsureComputeOpGetCondition,
    llzkVerif_EnsureComputeOpSetCondition, llzkVerif_EnsureConstrainOpBuild,
    llzkVerif_EnsureConstrainOpGetCondition, llzkVerif_EnsureConstrainOpSetCondition,
    llzkVerif_IncludeOpBuild, llzkVerif_IncludeOpContractTargetsStruct,
    llzkVerif_IncludeOpGetArgOperandsAt, llzkVerif_IncludeOpGetArgOperandsCount,
    llzkVerif_IncludeOpGetCallee, llzkVerif_IncludeOpGetMapOpGroupSizes,
    llzkVerif_IncludeOpGetMapOperandsAt, llzkVerif_IncludeOpGetMapOperandsCount,
    llzkVerif_IncludeOpGetNumDimsPerMap, llzkVerif_IncludeOpGetSelfValue,
    llzkVerif_IncludeOpGetTemplateParams, llzkVerif_IncludeOpGetTypeSignature,
    llzkVerif_IncludeOpResolveCallable, llzkVerif_IncludeOpSetArgOperands,
    llzkVerif_IncludeOpSetCallee, llzkVerif_IncludeOpSetMapOpGroupSizes,
    llzkVerif_IncludeOpSetMapOperands, llzkVerif_IncludeOpSetNumDimsPerMap,
    llzkVerif_IncludeOpSetTemplateParams, llzkVerif_IncreasesOpBuild, llzkVerif_InvariantOpBuild,
    llzkVerif_InvariantOpGetBody, llzkVerif_InvariantOpGetLoopArgTypes,
    llzkVerif_InvariantOpGetLoopName, llzkVerif_InvariantOpGetParentContract,
    llzkVerif_InvariantOpSetLoopArgTypes, llzkVerif_InvariantOpSetLoopName, llzkVerif_OldOpBuild,
    llzkVerif_RequireComputeOpBuild, llzkVerif_RequireComputeOpGetCondition,
    llzkVerif_RequireComputeOpSetCondition, llzkVerif_RequireConstrainOpBuild,
    llzkVerif_RequireConstrainOpGetCondition, llzkVerif_RequireConstrainOpSetCondition,
    llzkVerif_StepOpBuild, llzkVerif_StepOpGetRegion, llzkVerif_StepYieldOpBuild,
};
use melior::{
    StringRef,
    ir::{
        Attribute, AttributeLike, BlockLike as _, BlockRef, Identifier, Location, OperationRef,
        Region, RegionLike as _, RegionRef, Type, TypeLike, Value, ValueLike,
        attribute::{DenseI32ArrayAttribute, StringAttribute, TypeAttribute},
        block::{Block, BlockArgument},
        operation::OperationLike,
        r#type::FunctionType,
    },
};

use crate::{
    attributes::{array::ArrayAttribute, null_attr, rebuild_array_attr},
    builder::OpBuilderLike,
    error::Error,
    macros::llzk_op_type,
    symbol_ref::{SymbolRefAttrLike, SymbolRefAttribute},
    type_ext::FunctionTypeExt as _,
    value_ext::ValueRange,
};

mod contract_op_ext {
    use std::iter::FusedIterator;

    use melior::ir::{BlockLike as _, RegionLike as _, block::BlockArgument};

    use super::{ContractOpLike, ContractOpRef};

    /// Iterator over the input arguments of a contract op.
    #[derive(Debug)]
    pub struct ContractInputsIter<'c: 'a, 'a> {
        contract: ContractOpRef<'c, 'a>,
        start: usize,
        end: usize,
    }

    impl<'c: 'a, 'a> ContractInputsIter<'c, 'a> {
        pub(super) fn new(contract: &'a (impl ContractOpLike<'c, 'a> + ?Sized)) -> Self {
            let contract = unsafe { ContractOpRef::from_raw(contract.to_raw()) };
            let end = contract
                .body()
                .ok()
                .and_then(|region| region.first_block())
                .map(|block| block.argument_count())
                .unwrap_or(0);
            Self {
                contract,
                start: 0,
                end,
            }
        }
    }

    impl<'c: 'a, 'a> Iterator for ContractInputsIter<'c, 'a> {
        type Item = BlockArgument<'c, 'a>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let idx = self.start;
            self.start += 1;
            Some(self.contract.argument(idx).unwrap())
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            if self.start >= self.end {
                return (0, Some(0));
            }
            let size = self.end - self.start;
            (size, Some(size))
        }
    }

    impl ExactSizeIterator for ContractInputsIter<'_, '_> {}

    impl DoubleEndedIterator for ContractInputsIter<'_, '_> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            self.end -= 1;
            Some(self.contract.argument(self.end).unwrap())
        }
    }

    impl FusedIterator for ContractInputsIter<'_, '_> {}
}

pub use contract_op_ext::ContractInputsIter;

mod include_op_ext {
    use std::iter::FusedIterator;

    use melior::ir::Value;

    use super::{IncludeOpLike, IncludeOpRef};

    /// Iterator over the direct arg operands of an include op.
    #[derive(Debug)]
    pub struct IncludeArgOperandsIter<'c: 'a, 'a> {
        include: IncludeOpRef<'c, 'a>,
        start: usize,
        end: usize,
    }

    impl<'c: 'a, 'a> IncludeArgOperandsIter<'c, 'a> {
        pub(super) fn new(include: &'a (impl IncludeOpLike<'c, 'a> + ?Sized)) -> Self {
            let include = unsafe { IncludeOpRef::from_raw(include.to_raw()) };
            let end = include.arg_operand_count();
            Self {
                include,
                start: 0,
                end,
            }
        }
    }

    impl<'c: 'a, 'a> Iterator for IncludeArgOperandsIter<'c, 'a> {
        type Item = Value<'c, 'a>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let idx = self.start;
            self.start += 1;
            Some(self.include.arg_operand_at(idx))
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            if self.start >= self.end {
                return (0, Some(0));
            }
            let size = self.end - self.start;
            (size, Some(size))
        }
    }

    impl ExactSizeIterator for IncludeArgOperandsIter<'_, '_> {}

    impl DoubleEndedIterator for IncludeArgOperandsIter<'_, '_> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            self.end -= 1;
            Some(self.include.arg_operand_at(self.end))
        }
    }

    impl FusedIterator for IncludeArgOperandsIter<'_, '_> {}
}

pub use include_op_ext::IncludeArgOperandsIter;

/// Defines a `is_$name` operation that checks if the given operation matches the expected
/// operation type.
macro_rules! isa_fn {
    ($name:ident, $op_name:expr) => {
        paste::paste! {
            #[doc = concat!("Returns `true` iff the given op is `verif.", $op_name, "`.")]
            #[inline]
            pub fn [<is_ $name>]<'c: 'a, 'a>(op: &impl ::melior::ir::operation::OperationLike<'c, 'a>) -> bool {
                crate::operation::isa(op, concat!("verif.", $op_name))
            }
        }
    };
    ($name:ident) => {
        isa_fn!($name, stringify!($name));
    };
}

#[inline]
fn create_out_of_bounds_error<'c: 'a, 'a>(
    contract: &(impl ContractOpLike<'c, 'a> + ?Sized),
    idx: usize,
) -> Error {
    let fqn = contract.fully_qualified_name();
    Error::OutOfBoundsArgument(Some(fqn.to_string()), idx)
}

//===----------------------------------------------------------------------===//
// ContractOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the `verif.contract` op.
pub trait ContractOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the sym_name attribute.
    fn sym_name(&self) -> Result<StringAttribute<'c>, Error> {
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
    fn function_type(&self) -> Result<FunctionType<'c>, Error> {
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
    fn arg_attrs(&self) -> Result<ArrayAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_ContractOpGetArgAttrs(self.to_raw())) };
        Ok(rebuild_array_attr(unsafe { self.context().to_ref() }, attr))
    }

    /// Sets the argument attribute array.
    fn set_arg_attrs(&self, attr: ArrayAttribute<'c>) {
        unsafe { llzkVerif_ContractOpSetArgAttrs(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the body region of the contract.
    fn body(&self) -> Result<RegionRef<'c, 'a>, Error> {
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
    fn callable_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkVerif_ContractOpGetCallableRegion(self.to_raw())) }
    }

    /// Returns the fully qualified name of the contract.
    fn fully_qualified_name(&self) -> SymbolRefAttribute<'c> {
        unsafe {
            Attribute::from_raw(llzkVerif_ContractOpGetFullyQualifiedName(
                self.to_raw(),
                false,
            ))
        }
        .try_into()
        .expect("symbol ref attribute")
    }

    /// Returns the n-th argument of the contract.
    fn argument(&self, idx: usize) -> Result<BlockArgument<'c, 'a>, Error> {
        self.body()
            .and_then(|region| {
                region
                    .first_block()
                    .ok_or_else(|| create_out_of_bounds_error(self, idx))
            })
            .and_then(|block| block.argument(idx).map_err(Into::into))
    }

    /// Returns an iterator over the input arguments of the contract.
    fn inputs(&'a self) -> ContractInputsIter<'c, 'a> {
        ContractInputsIter::new(self)
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
///
/// The op is inserted at the builder's insertion point, returning a reference to it.
///
/// # Panics
///
/// If the insertion point is not set.
pub fn contract<'c, 'a, 'b>(
    builder: &'a impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: &str,
    target: impl SymbolRefAttrLike<'c>,
) -> Result<ContractOpRef<'c, 'b>, Error> {
    let context = location.context();
    let op = unsafe {
        ContractOpRef::from_raw(llzkVerif_ContractOpBuildFromTargetAttr(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(context.to_ref(), name).to_raw(),
            target.to_raw(),
        ))
    };
    if let Ok(region) = op.body() {
        if region.first_block().is_none() {
            let args = op
                .function_type()?
                .inputs()
                .map(|ty| (ty, location))
                .collect::<Vec<(Type<'c>, Location<'c>)>>();
            region.append_block(Block::new(&args));
        }
        Ok(op)
    } else {
        Err(Error::GeneralError("expected non-empty body"))
    }
}

isa_fn!(contract);

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

    /// Returns an iterator over the direct arg operands of the include op.
    fn arg_operands(&'a self) -> IncludeArgOperandsIter<'c, 'a> {
        IncludeArgOperandsIter::new(self)
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
    fn callee(&self) -> Result<SymbolRefAttribute<'c>, Error> {
        let attr = unsafe { Attribute::from_raw(llzkVerif_IncludeOpGetCallee(self.to_raw())) };
        Ok(attr.try_into()?)
    }

    /// Sets the callee attribute.
    fn set_callee(&self, attr: impl SymbolRefAttrLike<'c>) {
        unsafe { llzkVerif_IncludeOpSetCallee(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the template parameter attribute, if present.
    fn template_params(&self) -> Result<Option<ArrayAttribute<'c>>, Error> {
        let raw = unsafe { llzkVerif_IncludeOpGetTemplateParams(self.to_raw()) };
        if raw.ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(unsafe { Attribute::from_raw(raw) }.try_into()?))
        }
    }

    /// Sets the template parameter attribute.
    fn set_template_params(&self, attr: Option<ArrayAttribute<'c>>) {
        let raw = attr.map_or_else(null_attr, |attr| attr.to_raw());
        unsafe { llzkVerif_IncludeOpSetTemplateParams(self.to_raw(), raw) }
    }

    /// Returns the `numDimsPerMap` attribute.
    fn num_dims_per_map(&self) -> Result<DenseI32ArrayAttribute<'c>, Error> {
        let attr =
            unsafe { Attribute::from_raw(llzkVerif_IncludeOpGetNumDimsPerMap(self.to_raw())) };
        attr.try_into().map_err(Error::Melior)
    }

    /// Sets the `numDimsPerMap` attribute.
    fn set_num_dims_per_map(&self, attr: DenseI32ArrayAttribute<'c>) {
        unsafe { llzkVerif_IncludeOpSetNumDimsPerMap(self.to_raw(), attr.to_raw()) }
    }

    /// Returns the `mapOpGroupSizes` attribute.
    fn map_op_group_sizes(&self) -> Result<DenseI32ArrayAttribute<'c>, Error> {
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
pub fn include<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
) -> Result<IncludeOpRef<'c, 'a>, Error> {
    let template_params = template_params.map_or_else(null_attr, |attr| attr.to_raw());
    unsafe {
        OperationRef::from_raw(llzkVerif_IncludeOpBuild(
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
pub fn include_with_map_operands<'c, 'g, 'v, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
    map_operands: &'g [ValueRange<'c, 'v, '_>],
    num_dims_per_map: DenseI32ArrayAttribute<'c>,
) -> Result<IncludeOpRef<'c, 'a>, Error> {
    let op = include(builder, location, callee, args, template_params)?;
    op.set_map_operands(map_operands);
    op.set_num_dims_per_map(num_dims_per_map);
    Ok(op)
}

/// Creates a `verif.include` op with grouped map operands and dimension counts
/// provided as a Rust slice.
pub fn include_with_map_operands_slice<'c, 'g, 'v, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    callee: impl SymbolRefAttrLike<'c>,
    args: &[melior::ir::Value<'c, '_>],
    template_params: Option<ArrayAttribute<'c>>,
    map_operands: &'g [ValueRange<'c, 'v, '_>],
    num_dims_per_map: &[i32],
) -> Result<IncludeOpRef<'c, 'a>, Error> {
    let ctx = location.context();
    include_with_map_operands(
        builder,
        location,
        callee,
        args,
        template_params,
        map_operands,
        DenseI32ArrayAttribute::new(unsafe { ctx.to_ref() }, num_dims_per_map),
    )
}

isa_fn!(include);

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
                unsafe { $set(self.to_raw(), ::melior::ir::ValueLike::to_raw(&value)) }
            }
        }

        paste::paste! {
            impl<'a, 'c: 'a> ConditionOpLike<'c, 'a> for [<$type Ref>]<'c, 'a> {
                fn condition_raw(&self) -> melior::ir::Value<'c, 'a> {
                    unsafe { melior::ir::Value::from_raw($get(self.to_raw())) }
                }

                fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>) {
                    unsafe { $set(self.to_raw(), ::melior::ir::ValueLike::to_raw(&value)) }
                }
            }

            impl<'a, 'c: 'a> ConditionOpLike<'c, 'a> for [<$type RefMut>]<'c, 'a> {
                fn condition_raw(&self) -> melior::ir::Value<'c, 'a> {
                    unsafe { melior::ir::Value::from_raw($get(self.to_raw())) }
                }

                fn set_condition_raw(&self, value: melior::ir::Value<'c, '_>) {
                    unsafe { $set(self.to_raw(), ::melior::ir::ValueLike::to_raw(&value)) }
                }
            }


            #[doc = concat!("Creates a `", $name, "` op.")]
            pub fn $ctor<'c, 'a>(
                builder: &impl OpBuilderLike<'c>,
                location: Location<'c>,
                condition: impl melior::ir::ValueLike<'c>,
            ) -> Result<[<$type Ref>]<'c, 'a>, Error> {
                unsafe {
                    OperationRef::from_raw($build(
                        builder.to_raw(),
                        location.to_raw(),
                        condition.to_raw(),
                    ))
                }
                .try_into()
            }
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

//===----------------------------------------------------------------------===//
// InvariantOpLike
//===----------------------------------------------------------------------===//

/// Methods for `verif.invariant` ops.
pub trait InvariantOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the body of the operation.
    fn body(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(llzkVerif_InvariantOpGetBody(self.to_raw())) }
    }

    /// Returns the loop label.
    fn loop_name(&self) -> &'c str {
        StringAttribute::try_from(unsafe {
            Attribute::from_raw(llzkVerif_InvariantOpGetLoopName(self.to_raw()))
        })
        .unwrap()
        .value()
    }

    /// Returns the loop argument's types.
    fn loop_arg_types(&self) -> Vec<Type<'c>> {
        let arr = ArrayAttribute::try_from(unsafe {
            Attribute::from_raw(llzkVerif_InvariantOpGetLoopArgTypes(self.to_raw()))
        })
        .unwrap();
        arr.into_iter()
            .map(|a| TypeAttribute::try_from(a).unwrap().value())
            .collect()
    }

    /// Returns the parent `verif.contract` operation.
    fn parent_contract(&self) -> Option<ContractOpRef<'c, 'a>> {
        ContractOpRef::from_option_raw(unsafe {
            llzkVerif_InvariantOpGetParentContract(self.to_raw())
        })
    }
}

/// Mutable methods for `verif.invariant` ops.
pub trait InvariantOpMutLike<'c: 'a, 'a>: InvariantOpLike<'c, 'a> {
    /// Sets the loop label.
    fn set_loop_name(&mut self, name: &str) {
        let context = self.context();
        let name = StringAttribute::new(unsafe { context.to_ref() }, name);
        unsafe {
            llzkVerif_InvariantOpSetLoopName(self.to_raw(), Attribute::from(name).to_raw());
        }
    }

    /// Sets the loop argument's types.
    ///
    /// The types need to be consistent with the targeted loop's arguments and
    /// the block arguments of this op's body.
    fn set_loop_arg_types(&mut self, types: &[Type<'c>]) {
        let attrs = types
            .iter()
            .copied()
            .map(TypeAttribute::new)
            .map(Attribute::from)
            .collect::<Vec<_>>();
        let context = self.context();
        let arr = ArrayAttribute::new(unsafe { context.to_ref() }, &attrs);
        unsafe {
            llzkVerif_InvariantOpSetLoopArgTypes(self.to_raw(), arr.to_raw());
        }
    }
}

//===----------------------------------------------------------------------===//
// InvariantOp
//===----------------------------------------------------------------------===//

llzk_op_type!(
    InvariantOp,
    llzkOperationIsA_Verif_InvariantOp,
    "verif.invariant"
);

impl<'a, 'c: 'a> InvariantOpLike<'c, 'a> for InvariantOp<'c> {}
impl<'a, 'c: 'a> InvariantOpMutLike<'c, 'a> for InvariantOp<'c> {}
impl<'a, 'c: 'a> InvariantOpLike<'c, 'a> for InvariantOpRef<'c, 'a> {}
impl<'a, 'c: 'a> InvariantOpLike<'c, 'a> for InvariantOpRefMut<'c, 'a> {}
impl<'a, 'c: 'a> InvariantOpMutLike<'c, 'a> for InvariantOpRefMut<'c, 'a> {}

/// Creates a new invariant operation.
pub fn invariant<'c, 'o, B>(
    builder: &B,
    location: Location<'c>,
    loop_label: &str,
    args: &[(Type<'c>, Location<'c>)],
    build: impl FnOnce(&B, &[Value<'c, 'o>]) -> Result<(), Error>,
) -> Result<InvariantOpRef<'c, 'o>, Error>
where
    B: OpBuilderLike<'c>,
{
    let (types, locations): (Vec<_>, Vec<_>) =
        args.iter().map(|(t, l)| (t.to_raw(), l.to_raw())).unzip();
    let op = unsafe {
        InvariantOpRef::from_raw(llzkVerif_InvariantOpBuild(
            builder.to_raw(),
            location.to_raw(),
            StringRef::new(loop_label).to_raw(),
            args.len() as isize,
            types.as_ptr(),
            locations.as_ptr(),
        ))
    };
    let body = op.body();
    let saved = builder.save_insertion_point();
    builder.set_insertion_point_at_end(body);
    let arguments =
        Vec::from_iter((0..body.argument_count()).map(|n| Value::from(body.argument(n).unwrap())));
    let res = build(builder, &arguments);
    builder.restore_insertion_point(saved);
    res.map(|_| op)
}

isa_fn!(invariant);

/// Creates an `verif.increases` operation.
pub fn increases<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkVerif_IncreasesOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(increases);

/// Creates a `verif.decreases` operation.
pub fn decreases<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkVerif_DecreasesOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(decreases);

/// Creates a `verif.step` operation.
///
/// Accepts an optional builder callback for filling the body of the operation.
/// The callback must return a value that is used for creating the `verif.step.yield`
/// terminator op required by `verif.step`.
///
/// If the callback is not passed, then the operation is constructed as is and the caller is
/// responsible of manually adding the body.
pub fn step<'c, 'a, B>(
    builder: &B,
    location: Location<'c>,
    build: Option<impl FnOnce(&B) -> Result<Value<'c, 'a>, Error>>,
) -> Result<OperationRef<'c, 'a>, Error>
where
    B: OpBuilderLike<'c>,
{
    let op = unsafe {
        OperationRef::from_raw(llzkVerif_StepOpBuild(builder.to_raw(), location.to_raw()))
    };
    let Some(build) = build else {
        return Ok(op);
    };
    let region = unsafe { RegionRef::from_raw(llzkVerif_StepOpGetRegion(op.to_raw())) };
    let block = region.append_block(Block::new(&[]));
    let saved = builder.save_insertion_point();
    builder.set_insertion_point_at_end(block);
    let res = build(builder);
    if let Ok(value) = &res {
        step_yield(builder, location, *value);
    }
    builder.restore_insertion_point(saved);
    res.map(|_| op)
}

isa_fn!(step);

/// Creates a `verif.step.yield` operation.
///
/// When creating a [`step`] operation with a builder callback is not necessary to call this
/// factory.
pub fn step_yield<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkVerif_StepYieldOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(step_yield, "step.yield");

/// Creates a `verif.old` operation.
pub fn old<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkVerif_OldOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(old);
