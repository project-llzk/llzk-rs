use crate::builder::OpBuilderLike;
use llzk_sys::llzkLlzk_NonDetOpBuild;
use melior::ir::{Location, OperationRef, Type, TypeLike, operation::OperationLike};

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

/// Return `true` iff the given op is `llzk.nondet`.
#[inline]
pub fn is_nondet<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "llzk.nondet")
}
