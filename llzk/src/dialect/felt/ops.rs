use super::FeltConstAttribute;
use crate::builder::OpBuilderLike;
use crate::error::Error;
use llzk_sys::{
    llzkFelt_AddFeltOpBuild, llzkFelt_AndFeltOpBuild, llzkFelt_DivFeltOpBuild,
    llzkFelt_FeltConstantOpBuild, llzkFelt_InvFeltOpBuild, llzkFelt_MulFeltOpBuild,
    llzkFelt_NegFeltOpBuild, llzkFelt_NotFeltOpBuild, llzkFelt_OrFeltOpBuild,
    llzkFelt_PowFeltOpBuild, llzkFelt_ShlFeltOpBuild, llzkFelt_ShrFeltOpBuild,
    llzkFelt_SignedIntDivFeltOpBuild, llzkFelt_SignedModFeltOpBuild, llzkFelt_SubFeltOpBuild,
    llzkFelt_UnsignedIntDivFeltOpBuild, llzkFelt_UnsignedModFeltOpBuild, llzkFelt_XorFeltOpBuild,
};
use melior::ir::{
    AttributeLike, Location, OperationRef, Type, TypeLike, Value, ValueLike as _,
    operation::OperationLike,
};

#[inline]
fn build_binop<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    result: Type<'c>,
    build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirType,
        mlir_sys::MlirValue,
        mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> Result<OperationRef<'c, 'a>, Error> {
    Ok(unsafe {
        OperationRef::from_raw(build(
            builder.to_raw(),
            location.to_raw(),
            result.to_raw(),
            lhs.to_raw(),
            rhs.to_raw(),
        ))
    })
}

#[inline]
fn build_unop<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    value: Value<'c, '_>,
    result: Type<'c>,
    build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirType,
        mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> Result<OperationRef<'c, 'a>, Error> {
    Ok(unsafe {
        OperationRef::from_raw(build(
            builder.to_raw(),
            location.to_raw(),
            result.to_raw(),
            value.to_raw(),
        ))
    })
}

macro_rules! binop {
    ($name:ident) => {
        binop!($name, stringify!($name));
    };
    ($name:ident, $opname:expr, $build:ident) => {
        #[doc = concat!("Creates a `felt.", $opname ,"` operation.")]
        pub fn $name<'c, 'a>(
            builder: &impl OpBuilderLike<'c>,
            location: Location<'c>,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
        ) -> Result<OperationRef<'c, 'a>, Error> {
            build_binop(builder, location, lhs, rhs, lhs.r#type(), $build)
        }

        paste::paste! {
            #[doc = concat!("Return `true` iff the given op is `felt.", $opname ,"`.")]
            #[inline]
            pub fn [<is_felt_ $name>]<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
                crate::operation::isa(op, concat!("felt.", $opname))
            }
        }
    };
}

macro_rules! unop {
    ($name:ident) => {
        unop!($name, stringify!($name));
    };
    ($name:ident, $opname:expr, $build:ident) => {
        #[doc = concat!("Creates a `felt.", $opname ,"` operation.")]
        pub fn $name<'c, 'a>(
            builder: &impl OpBuilderLike<'c>,
            location: Location<'c>,
            value: Value<'c, '_>,
        ) -> Result<OperationRef<'c, 'a>, Error> {
            build_unop(builder, location, value, value.r#type(), $build)
        }

        paste::paste! {
            #[doc = concat!("Return `true` iff the given op is `felt.", $opname ,"`.")]
            #[inline]
            pub fn [<is_felt_ $name>]<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
                crate::operation::isa(op, concat!("felt.", $opname))
            }
        }
    };
}

binop!(add, "add", llzkFelt_AddFeltOpBuild);
binop!(bit_and, "bit_and", llzkFelt_AndFeltOpBuild);
binop!(bit_or, "bit_or", llzkFelt_OrFeltOpBuild);
binop!(bit_xor, "bit_xor", llzkFelt_XorFeltOpBuild);
binop!(div, "div", llzkFelt_DivFeltOpBuild);
binop!(mul, "mul", llzkFelt_MulFeltOpBuild);
binop!(pow, "pow", llzkFelt_PowFeltOpBuild);
binop!(shl, "shl", llzkFelt_ShlFeltOpBuild);
binop!(shr, "shr", llzkFelt_ShrFeltOpBuild);
binop!(sintdiv, "sintdiv", llzkFelt_SignedIntDivFeltOpBuild);
binop!(smod, "smod", llzkFelt_SignedModFeltOpBuild);
binop!(sub, "sub", llzkFelt_SubFeltOpBuild);
binop!(uintdiv, "uintdiv", llzkFelt_UnsignedIntDivFeltOpBuild);
binop!(umod, "umod", llzkFelt_UnsignedModFeltOpBuild);
unop!(bit_not, "bit_not", llzkFelt_NotFeltOpBuild);
unop!(inv, "inv", llzkFelt_InvFeltOpBuild);
unop!(neg, "neg", llzkFelt_NegFeltOpBuild);

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
