//! Convenience functions for creating common operation patterns.

use super::r#type::StructType;
use crate::{
    attributes::NamedAttribute,
    builder::{EntryPoint, OpBuilder, OpBuilderLike},
    dialect,
    error::Error,
    prelude::{
        FUNC_NAME_COMPUTE, FUNC_NAME_CONSTRAIN, FUNC_NAME_PRODUCT, FeltType, FuncDefOpLike as _,
        FuncDefOpRef, StructDefOp,
    },
};
use melior::{
    Context,
    ir::{
        Attribute, Block, BlockLike as _, Identifier, Location, RegionLike as _, Type,
        operation::OperationLike as _, r#type::FunctionType,
    },
};

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

/// Creates the `@Signal` struct.
///
/// The `@Main` struct's inputs must be of this type or arrays of this type.
#[deprecated]
pub fn define_signal_struct<'c>(context: &'c Context) -> Result<StructDefOp<'c>, Error> {
    let loc = Location::new(context, "Signal struct", 0, 0);
    let typ = StructType::from_str(context, "Signal");
    let reg = "reg";
    let block = Block::new(&[]);
    let builder = OpBuilder::at_block_begin(context, &block);
    let op = super::def(&builder, loc, "Signal", |builder| {
        super::member(builder, loc, reg, FeltType::new(context), true, false, true)?;
        compute_fn(
            builder,
            loc,
            typ,
            &[(FeltType::new(context).into(), loc)],
            None,
        )
        .and_then(|compute| {
            let block = compute
                .body()?
                .first_block()
                .ok_or(Error::BlockExpected(0))?;
            let fst = block.first_operation().ok_or(Error::EmptyBlock)?;
            if fst.name() != Identifier::new(context, "struct.new") {
                return Err(Error::OperationExpected(
                    "struct.new",
                    fst.name().as_string_ref().as_str()?.to_owned(),
                ));
            }
            let builder = OpBuilder::new(context, EntryPoint::After(fst));
            super::writem(
                &builder,
                loc,
                fst.result(0)?.into(),
                reg,
                block.argument(0)?.into(),
            )?;
            Ok(compute)
        })?;
        constrain_fn(
            builder,
            loc,
            typ,
            &[(FeltType::new(context).into(), loc)],
            None,
        )
        .and_then(|constrain| {
            let block = constrain
                .body()?
                .first_block()
                .ok_or(Error::BlockExpected(0))?;
            let fst = block.first_operation().ok_or(Error::EmptyBlock)?;
            if fst.name() != Identifier::new(context, "function.return") {
                return Err(Error::OperationExpected(
                    "function.return",
                    fst.name().as_string_ref().as_str()?.to_owned(),
                ));
            }
            let builder = OpBuilder::new(context, EntryPoint::Before(fst));
            let reg = super::readm(
                &builder,
                loc,
                FeltType::new(context).into(),
                block.argument(0)?.into(),
                "reg",
            )?;
            builder.set_insertion_point_after(reg);
            dialect::constrain::eq(
                &builder,
                loc,
                reg.result(0)?.into(),
                block.argument(1)?.into(),
            );
            Ok(constrain)
        })?;
        Ok(())
    })?;
    Ok(unsafe { op.to_ref() }.clone())
}
