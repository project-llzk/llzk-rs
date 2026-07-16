#![allow(unused_crate_dependencies)]
//! Heavily commented example of creating IR representing a circuit for a division gadget.
//!
//! The gadget performs the division and constrains the dividend to be equal to the quotient times
//! the divisor.
//!
//! Creates a single struct with two inputs and one output.
//!
//! Run with: `cargo run --package llzk --example division`

use std::result::Result as StdResult;

// Commonly used types are re-exported in the prelude.
use llzk::{
    builder::{OpBuilder, OpBuilderLike},
    prelude::*,
};

type Result<T> = StdResult<T, LlzkError>;

const MAIN_STRUCT_NAME: &str = "Entry";

fn main() -> Result<()> {
    // The context preloads the LLZK dialects for convenience.
    let context = LlzkContext::new();
    // IR objects have a location associated to them. Usually a source location
    // but we won't bother with that in this case.
    let location = Location::unknown(&context);
    // LLZK top-level modules require some additional attributes.
    // This function creates a module preconfigured with these attributes.
    let mut module = llzk_module(location, None);
    module.as_operation_mut().set_attribute(
        MAIN_ATTR_NAME.as_ref(),
        TypeAttribute::new(
            StructType::new(FlatSymbolRefAttribute::new(&context, MAIN_STRUCT_NAME), &[]).into(),
        )
        .into(),
    );

    // Operations can be created with factory methods with the same name as the op they create,
    // mimicking its mnemonic (struct.def in this case).
    let builder = OpBuilder::at_block_begin(&context, module.body());

    // The inputs of the main struct must be of type !felt.type (or array thereof).
    let felt_type = FeltType::new(&context);
    let out_field_name = "c";

    // We store the output of the division in a data field.
    // Members can have three extra annotations: signal, column, and public.
    // The signal annotation makes the field a witness-stored constraint variable.
    // The public annotation makes the field an output of the circuit.
    dialect::r#struct::def(&builder, location, MAIN_STRUCT_NAME, |builder| {
        let is_signal = true;
        let is_column = false;
        let is_public = true;
        dialect::r#struct::member(
            builder,
            location,
            out_field_name,
            felt_type,
            is_signal,
            is_column,
            is_public,
        )?;
        gen_witness(
            builder,
            &context,
            location,
            felt_type.into(),
            out_field_name,
        )?;
        gen_constrain(
            builder,
            &context,
            location,
            felt_type.into(),
            out_field_name,
        )?;
        Ok(())
    })?;

    // Now that we have filled out the struct we can verify it and print it.
    // For verifying and printing we need get a reference to the `builtin.module` op
    // representing the module.
    let module_op = module.as_operation();
    if module_op.verify() {
        println!("{module_op}")
    } else {
        eprintln!("Module failed to verify");
    }

    Ok(())
}

fn gen_witness<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    // Context is the type used in melior to represent the MLIRContext.
    // A reference to a LlzkContext can be used as a reference to a Context.
    context: &'c Context,
    location: Location<'c>,
    input_type: Type<'c>,
    out_field_name: &str,
) -> Result<OperationRef<'c, 'a>> {
    let ip = builder.save_insertion_point();

    // The inputs to the functions are public circuit inputs.
    let inputs = vec![(input_type, location); 2];
    let pub_attr = [PublicAttribute::new_named_attr(context)];
    let main_ty = StructType::from_str(context, MAIN_STRUCT_NAME);

    // The functions inside a struct need to have a particular structure. This helper creates the
    // `@compute` function with its proper structure.
    let compute_fn = dialect::r#struct::helpers::compute_fn(
        builder,
        location,
        main_ty,
        &inputs,
        Some(&[pub_attr.to_vec(), pub_attr.to_vec()]),
    )?;

    // Witness generation is represented by creating an instance of the containing struct, filling
    // its fields, and returning the value of the struct. The `compute_fn` helper
    // inserts a `struct.new` operation followed by a `function.return` operation to represent this.
    // The specific IR for our circuit needs to go in between these two operations.
    // We will insert it using the return op as reference so we need to get ahold of it and the
    // block that contains it.
    let (block, ret_op) = compute_fn
        .body()?
        .first_block()
        .and_then(|b| Some((b, b.terminator()?)))
        .unwrap();

    // Get the two inputs from the block arguments.
    let a = block.argument(0)?;
    let b = block.argument(1)?;

    // The witness computes c = a / b
    builder.set_insertion_point(ret_op);
    let c = dialect::felt::div(builder, location, a.into(), b.into())?.result(0)?;
    // The result needs to be written into the output field. For that we need to get the value
    // created by `struct.new` first.
    let self_value = block.first_operation().unwrap().result(0)?;
    // Then use the `struct.writem` operation to commit the value into the field.
    dialect::r#struct::writem(
        builder,
        location,
        self_value.into(),
        out_field_name,
        c.into(),
    )?;

    builder.restore_insertion_point(ip);
    Ok(compute_fn.into())
}

fn gen_constrain<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    context: &'c Context,
    location: Location<'c>,
    input_type: Type<'c>,
    out_field_name: &str,
) -> Result<OperationRef<'c, 'a>> {
    let ip = builder.save_insertion_point();

    // The inputs to the functions are public circuit inputs.
    let inputs = vec![(input_type, location); 2];
    let pub_attr = [PublicAttribute::new_named_attr(context)];
    let main_ty = StructType::from_str(context, MAIN_STRUCT_NAME);

    // The functions inside a struct need to have a particular structure. This helper creates the
    // `@constrain` function with its proper structure.
    let constrain_fn = dialect::r#struct::helpers::constrain_fn(
        builder,
        location,
        main_ty,
        &inputs,
        Some(&[pub_attr.to_vec(), pub_attr.to_vec()]),
    )?;

    // The constraint system is represented by a function that takes as argument an instance of
    // the parent struct as well as the same inputs the `@compute` function takes.
    // This function returns no values.
    // The `constrain_fn` helper inserts an empty `function.return` operation.
    //
    // Similar to how we generated the IR for `@compute` we need to put the IR before the
    // `function.return` operation.
    let (block, ret_op) = constrain_fn
        .body()?
        .first_block()
        .and_then(|b| Some((b, b.terminator()?)))
        .unwrap();

    // Obtain the inputs same as before but with the offsets increased by 1.
    let a = block.argument(1)?;
    let b = block.argument(2)?;

    // The instance that we are constraining is passed as the first argument.
    let self_value = block.argument(0)?;
    // And then read the witness output from the instance.
    builder.set_insertion_point(ret_op);
    let c = dialect::r#struct::readm(
        builder,
        location,
        FeltType::new(context).into(),
        self_value.into(),
        out_field_name,
    )?
    .result(0)?;

    // The constraint is  c * b = a
    // We can use the `constrain.eq` operation for emitting equality constraints.
    let t = dialect::felt::mul(builder, location, c.into(), b.into())?.result(0)?;
    dialect::constrain::eq(builder, location, t.into(), a.into());

    builder.restore_insertion_point(ip);
    Ok(constrain_fn.into())
}
