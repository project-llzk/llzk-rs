use super::FeltConstAttribute;
use crate::{builder::OpBuilderLike, error::Error, macros::isa_fn};
use llzk_sys::{
    llzkFelt_AddFeltOpBuild, llzkFelt_AndFeltOpBuild, llzkFelt_DivFeltOpBuild,
    llzkFelt_FeltConstantOpBuild, llzkFelt_InvFeltOpBuild, llzkFelt_MulFeltOpBuild,
    llzkFelt_NegFeltOpBuild, llzkFelt_NotFeltOpBuild, llzkFelt_OrFeltOpBuild,
    llzkFelt_PowFeltOpBuild, llzkFelt_ShlFeltOpBuild, llzkFelt_ShrFeltOpBuild,
    llzkFelt_SignedIntDivFeltOpBuild, llzkFelt_SignedModFeltOpBuild, llzkFelt_SubFeltOpBuild,
    llzkFelt_UnsignedIntDivFeltOpBuild, llzkFelt_UnsignedModFeltOpBuild, llzkFelt_XorFeltOpBuild,
};
use melior::ir::{
    AttributeLike as _, Location, OperationRef, TypeLike as _, operation::OperationLike,
};

macro_rules! op {
    ($arity:ident, $($args:tt)*) => {
        crate::macros::dialect_op!($arity typed felt, $($args)*);
    };
}

op!(binop, add);
op!(binop, sub);
op!(binop, div);
op!(binop, mul);
op!(binop, pow);
op!(binop, shl);
op!(binop, shr);
op!(binop, sintdiv, llzkFelt_SignedIntDivFeltOpBuild);
op!(binop, smod, llzkFelt_SignedModFeltOpBuild);
op!(binop, uintdiv, llzkFelt_UnsignedIntDivFeltOpBuild);
op!(binop, umod, llzkFelt_UnsignedModFeltOpBuild);
op!(binop, bit_and, llzkFelt_AndFeltOpBuild);
op!(binop, bit_or, llzkFelt_OrFeltOpBuild);
op!(binop, bit_xor, llzkFelt_XorFeltOpBuild);
op!(unop, inv);
op!(unop, neg);
op!(unop, bit_not, llzkFelt_NotFeltOpBuild);

/// Creates a `felt.const` operation.
pub fn constant<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: FeltConstAttribute<'c>,
) -> Result<OperationRef<'c, 'a>, Error> {
    Ok(unsafe {
        OperationRef::from_raw(llzkFelt_FeltConstantOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.r#type().to_raw(),
            value.to_raw(),
        ))
    })
}

/// Return `true` iff the given op is `felt.const`.
#[inline]
pub fn is_felt_const<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "felt.const")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn felt_const_op(value: u64) {
        // ensure value fits in i64, which is what's internally used by FeltConstAttribute
        let value = value % (i64::MAX as u64 + 1);
        let ctx = LlzkContext::new();
        let module = Module::new(Location::unknown(&ctx));
        let builder = crate::builder::OpBuilder::at_block_begin(&ctx, module.body());
        let op = constant(
            &builder,
            Location::unknown(&ctx),
            FeltConstAttribute::new(&ctx, value, None),
        )
        .unwrap();
        assert!(op.verify(), "operation {op:?} failed verification");
    }

    #[quickcheck]
    fn felt_const_op_isa(value: u64) {
        // ensure value fits in i64, which is what's internally used by FeltConstAttribute
        let value = value % (i64::MAX as u64 + 1);
        let ctx = LlzkContext::new();
        let module = Module::new(Location::unknown(&ctx));
        let builder = crate::builder::OpBuilder::at_block_begin(&ctx, module.body());
        let op = constant(
            &builder,
            Location::unknown(&ctx),
            FeltConstAttribute::new(&ctx, value, None),
        )
        .unwrap();
        assert!(is_felt_const(&op), "operation {op:?} failed isa test");
    }
}
