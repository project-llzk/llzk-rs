use super::FeltConstAttribute;
use crate::{builder::OpBuilderLike, error::Error};
use llzk_sys::{
    llzkFelt_AddFeltOpBuild, llzkFelt_AndFeltOpBuild, llzkFelt_DivFeltOpBuild,
    llzkFelt_FeltConstantOpBuild, llzkFelt_InvFeltOpBuild, llzkFelt_MulFeltOpBuild,
    llzkFelt_NegFeltOpBuild, llzkFelt_NotFeltOpBuild, llzkFelt_OrFeltOpBuild,
    llzkFelt_PowFeltOpBuild, llzkFelt_ShlFeltOpBuild, llzkFelt_ShrFeltOpBuild,
    llzkFelt_SignedIntDivFeltOpBuild, llzkFelt_SignedModFeltOpBuild, llzkFelt_SubFeltOpBuild,
    llzkFelt_UnsignedIntDivFeltOpBuild, llzkFelt_UnsignedModFeltOpBuild, llzkFelt_XorFeltOpBuild,
    llzkOperationIsA_Felt_AddFeltOp, llzkOperationIsA_Felt_AndFeltOp,
    llzkOperationIsA_Felt_DivFeltOp, llzkOperationIsA_Felt_FeltConstantOp,
    llzkOperationIsA_Felt_InvFeltOp, llzkOperationIsA_Felt_MulFeltOp,
    llzkOperationIsA_Felt_NegFeltOp, llzkOperationIsA_Felt_NotFeltOp,
    llzkOperationIsA_Felt_OrFeltOp, llzkOperationIsA_Felt_PowFeltOp,
    llzkOperationIsA_Felt_ShlFeltOp, llzkOperationIsA_Felt_ShrFeltOp,
    llzkOperationIsA_Felt_SignedIntDivFeltOp, llzkOperationIsA_Felt_SignedModFeltOp,
    llzkOperationIsA_Felt_SubFeltOp, llzkOperationIsA_Felt_UnsignedIntDivFeltOp,
    llzkOperationIsA_Felt_UnsignedModFeltOp, llzkOperationIsA_Felt_XorFeltOp,
};
use melior::ir::{AttributeLike as _, Location, OperationRef, TypeLike as _};

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
op!(
    binop,
    sintdiv,
    llzkFelt_SignedIntDivFeltOpBuild,
    llzkOperationIsA_Felt_SignedIntDivFeltOp
);
op!(
    binop,
    smod,
    llzkFelt_SignedModFeltOpBuild,
    llzkOperationIsA_Felt_SignedModFeltOp
);
op!(
    binop,
    uintdiv,
    llzkFelt_UnsignedIntDivFeltOpBuild,
    llzkOperationIsA_Felt_UnsignedIntDivFeltOp
);
op!(
    binop,
    umod,
    llzkFelt_UnsignedModFeltOpBuild,
    llzkOperationIsA_Felt_UnsignedModFeltOp
);
op!(
    binop,
    bit_and,
    llzkFelt_AndFeltOpBuild,
    llzkOperationIsA_Felt_AndFeltOp
);
op!(
    binop,
    bit_or,
    llzkFelt_OrFeltOpBuild,
    llzkOperationIsA_Felt_OrFeltOp
);
op!(
    binop,
    bit_xor,
    llzkFelt_XorFeltOpBuild,
    llzkOperationIsA_Felt_XorFeltOp
);
op!(unop, inv);
op!(unop, neg);
op!(
    unop,
    bit_not,
    llzkFelt_NotFeltOpBuild,
    llzkOperationIsA_Felt_NotFeltOp
);

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

crate::macros::isa_fn!(felt, const, llzkOperationIsA_Felt_FeltConstantOp);

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
        assert!(is_const_op(&op), "operation {op:?} failed isa test");
    }
}
