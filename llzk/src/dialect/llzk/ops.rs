use crate::builder::OpBuilderLike;
use llzk_sys::{llzkLlzk_NonDetOpBuild, llzkOperationIsA_Llzk_NonDetOp};
use melior::ir::{Location, OperationRef, Type, TypeLike};

/// Creates a new `llzk.nondet` operation.
pub fn nondet<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result_type: Type<'c>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkLlzk_NonDetOpBuild(
            builder.to_raw(),
            location.to_raw(),
            result_type.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(llzk, nondet, llzkOperationIsA_Llzk_NonDetOp);
