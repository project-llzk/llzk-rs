use melior::ir::{
    Location, Operation, Type,
    operation::{OperationBuilder, OperationLike},
};

/// Creates a new `llzk.nondet` operation.
pub fn nondet<'c>(location: Location<'c>, result_type: Type<'c>) -> Operation<'c> {
    OperationBuilder::new("llzk.nondet", location)
        .add_results(&[result_type])
        .build()
        .unwrap()
}

/// Return `true` iff the given op is `llzk.nondet`.
#[inline]
pub fn is_nondet<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "llzk.nondet")
}
