//! Heavily commented example of creating IR representing a circuit for a division gadget.
//!
//! The gadget performs the division and constrains the dividend to be equal to the quotient times
//! the divisor.
//!
//! Creates a single struct with two inputs and one output.
//!
//! Run with: `cargo run --package llzk --example division`

use std::error::Error as StdError;
use std::result::Result as StdResult;

// Commonly used types are re-exported in the prelude.
use llzk::{builder::OpBuilder, prelude::*};

type Result<T> = StdResult<T, Box<dyn StdError>>;

const MAIN_STRUCT_NAME: &'static str = "Entry";

fn main() -> Result<()> {
    // The context preloads the LLZK dialects for convenience.
    let context = LlzkContext::new();
    // IR objects have a location associated to them. Usually a source location
    // but we won't bother with that in this case.
    let location = Location::unknown(&context);
    // LLZK top-level modules require some additional attributes.
    // This function creates a module preconfigured with these attributes.
    let mut module = llzk_module(location);
    module.as_operation_mut().set_attribute(
        MAIN_ATTR_NAME.as_ref(),
        TypeAttribute::new(
            StructType::new(FlatSymbolRefAttribute::new(&context, MAIN_STRUCT_NAME), &[]).into(),
        )
        .into(),
    );

    // Operations can be created with factory methods with the same name as the op they create,
    // mimicking its mnemonic (struct.def in this case).
    let main_st = dialect::r#struct::def(location, MAIN_STRUCT_NAME, &[], [])?;

    // The inputs of the main struct must be of type !felt.type (or array thereof).
    let felt_type = FeltType::new(&context);

    // We store the output of the division in a data field.
    // Members can have two extra annotations; column and public.
    // The public annotation makes the field an output of the circuit.
    let out_field = {
        let is_column = false;
        let is_public = true;
        dialect::r#struct::member(location, "c", felt_type, is_column, is_public)?
    };
    let compute_fn = witness(&context, location, felt_type.into(), &out_field)?;
    let constrain_fn = constraints(&context, location, felt_type.into(), &out_field)?;

    main_st.body().append_operation(out_field.into());
    main_st.body().append_operation(compute_fn.into());
    main_st.body().append_operation(constrain_fn.into());

    // Now that we have filled out the struct we can add it to the module, verify it, and print it.
    module.body().append_operation(main_st.into());
    // For verifying and printing we need get a reference to the `builtin.module` op representing
    // the module.
    let module_op = module.as_operation();

    if module_op.verify() {
        println!("{module_op}")
    } else {
        eprintln!("Module failed to verify");
    }

    Ok(())
}

fn witness<'c>(
    // Context is the type used in melior to represent the MLIRContext.
    // A reference to a LlzkContext can be used as a reference to a Context.
    context: &'c Context,
    location: Location<'c>,
    input_type: Type<'c>,
    out_field: &MemberDefOp<'c>,
) -> Result<Operation<'c>> {
    // The inputs to the functions are public circuit inputs.
    let inputs = vec![(input_type, location); 2];
    let pub_attr = [PublicAttribute::new_named_attr(context)];
    let main_ty = StructType::from_str(context, MAIN_STRUCT_NAME);

    // The functions inside a struct need to have a particular structure. This helper creates the
    // `@compute` function with its proper structure.
    let compute_fn = dialect::r#struct::helpers::compute_fn(
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
        .region(0)?
        .first_block()
        .and_then(|b| Some((b, b.terminator()?)))
        .unwrap();

    // Get the two inputs from the block arguments.
    let a = block.argument(0)?;
    let b = block.argument(1)?;

    // The witness computes c = a / b
    let c = block
        .insert_operation_before(ret_op, dialect::felt::div(location, a.into(), b.into())?)
        .result(0)?;
    // The result needs to be written into the output field. For that we need to get the value
    // created by `struct.new` first.
    let self_value = block.first_operation().unwrap().result(0)?;
    // Then use the `struct.writem` operation to commit the value into the field.
    block.insert_operation_before(
        ret_op,
        dialect::r#struct::writem(
            location,
            self_value.into(),
            out_field.member_name(),
            c.into(),
        )?,
    );

    Ok(compute_fn.into())
}

fn constraints<'c>(
    context: &'c Context,
    location: Location<'c>,
    input_type: Type<'c>,
    out_field: &MemberDefOp<'c>,
) -> Result<Operation<'c>> {
    // The inputs to the functions are public circuit inputs.
    let inputs = vec![(input_type, location); 2];
    let pub_attr = [PublicAttribute::new_named_attr(context)];
    let main_ty = StructType::from_str(context, MAIN_STRUCT_NAME);

    // The functions inside a struct need to have a particular structure. This helper creates the
    // `@constrain` function with its proper structure.
    let constrain_fn = dialect::r#struct::helpers::constrain_fn(
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
        .region(0)?
        .first_block()
        .and_then(|b| Some((b, b.terminator()?)))
        .unwrap();

    let builder = OpBuilder::new(context);

    // Obtain the inputs same as before but with the offsets increased by 1.
    let a = block.argument(1)?;
    let b = block.argument(2)?;

    // The instance that we are constraining is passed as the first argument.
    let self_value = block.argument(0)?;
    // And then read the witness output from the instance.
    let c = block
        .insert_operation_before(
            ret_op,
            dialect::r#struct::readm(
                &builder,
                location,
                FeltType::new(context).into(),
                self_value.into(),
                out_field.member_name(),
            )?,
        )
        .result(0)?;

    // The constraint is  c * b = a
    // We can use the `constrain.eq` operation for emitting equality constraints.
    let t = block
        .insert_operation_before(ret_op, dialect::felt::mul(location, c.into(), b.into())?)
        .result(0)?;
    block.insert_operation_before(ret_op, dialect::constrain::eq(location, t.into(), a.into()));

    Ok(constrain_fn.into())
}
