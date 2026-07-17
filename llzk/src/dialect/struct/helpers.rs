//! Convenience functions for creating common operation patterns.

use super::r#type::StructType;
use crate::{
    attributes::NamedAttribute,
    builder::{OpBuilder, OpBuilderLike},
    dialect,
    error::Error,
    prelude::{
        FUNC_NAME_COMPUTE, FUNC_NAME_CONSTRAIN, FUNC_NAME_PRODUCT, FuncDefOpLike as _, FuncDefOpRef,
    },
};
use melior::ir::{Attribute, Identifier, Location, RegionLike as _, Type, r#type::FunctionType};

/// Creates an empty `@compute` function with the configuration expected by `struct.def`.
pub fn compute_fn<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    loc: Location<'c>,
    struct_type: StructType<'c>,
    inputs: &[(Type<'c>, Location<'c>)],
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> Result<FuncDefOpRef<'c, 'a>, Error> {
    let context = loc.context();
    let input_types: Vec<Type<'c>> = inputs.iter().map(|(t, _)| *t).collect();
    dialect::function::def(
        builder,
        loc,
        FUNC_NAME_COMPUTE.as_ref(),
        FunctionType::new(
            unsafe { context.to_ref() },
            &input_types,
            &[struct_type.into()],
        ),
        &[],
        arg_attrs,
        dialect::empty_region,
    )
    .and_then(|f| {
        let block = f.body()?.first_block().ok_or(Error::EmptyBlock)?;
        let inner_bldr = OpBuilder::at_block_begin(unsafe { context.to_ref() }, block);
        let new_struct = super::new(&inner_bldr, loc, struct_type);
        dialect::function::r#return(&inner_bldr, loc, &[new_struct.result(0)?.into()]);
        f.set_allow_witness_attr(true);
        f.set_allow_non_native_field_ops_attr(true);
        Ok(f)
    })
}

/// Creates an empty `@constrain` function with the configuration expected by `struct.def`.
///
/// If `arg_attrs` is `Some` it must have `inputs.len() + 1` elements and element #0 is the
/// argument attributes of the self argument.
pub fn constrain_fn<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    loc: Location<'c>,
    struct_type: StructType<'c>,
    inputs: &[(Type<'c>, Location<'c>)],
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> Result<FuncDefOpRef<'c, 'a>, Error> {
    let context = loc.context();
    let mut input_types: Vec<Type<'c>> = Vec::with_capacity(inputs.len() + 1);
    input_types.push(struct_type.into());
    input_types.extend(inputs.iter().map(|(t, _)| *t));
    let all_arg_attrs = arg_attrs.map(|original| {
        let mut result: Vec<Vec<(Identifier<'_>, Attribute<'_>)>> = vec![vec![]];
        result.extend(original.iter().cloned());
        result
    });
    dialect::function::def(
        builder,
        loc,
        FUNC_NAME_CONSTRAIN.as_ref(),
        FunctionType::new(unsafe { context.to_ref() }, &input_types, &[]),
        &[],
        all_arg_attrs.as_deref(),
        dialect::empty_region,
    )
    .and_then(|f| {
        let block = f.body()?.first_block().ok_or(Error::EmptyBlock)?;
        let inner_bldr = OpBuilder::at_block_begin(unsafe { context.to_ref() }, block);
        dialect::function::r#return(&inner_bldr, loc, &[]);
        f.set_allow_constraint_attr(true);
        f.set_allow_non_native_field_ops_attr(true);
        Ok(f)
    })
}

/// Creates an empty `@product` function with the configuration expected by `struct.def`.
pub fn product_fn<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    loc: Location<'c>,
    struct_type: StructType<'c>,
    inputs: &[(Type<'c>, Location<'c>)],
    arg_attrs: Option<&[Vec<NamedAttribute<'c>>]>,
) -> Result<FuncDefOpRef<'c, 'a>, Error> {
    let context = loc.context();
    let input_types: Vec<Type<'c>> = inputs.iter().map(|(t, _)| *t).collect();
    dialect::function::def(
        builder,
        loc,
        FUNC_NAME_PRODUCT.as_ref(),
        FunctionType::new(
            unsafe { context.to_ref() },
            &input_types,
            &[struct_type.into()],
        ),
        &[],
        arg_attrs,
        dialect::empty_region,
    )
    .and_then(|f| {
        let block = f.body()?.first_block().ok_or(Error::EmptyBlock)?;
        let inner_bldr = OpBuilder::at_block_begin(unsafe { context.to_ref() }, block);
        let new_struct = super::new(&inner_bldr, loc, struct_type);
        dialect::function::r#return(&inner_bldr, loc, &[new_struct.result(0)?.into()]);
        f.set_allow_constraint_attr(true);
        f.set_allow_witness_attr(true);
        f.set_allow_non_native_field_ops_attr(true);
        Ok(f)
    })
}
