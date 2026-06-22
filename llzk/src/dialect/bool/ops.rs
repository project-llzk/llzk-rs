use crate::{
    builder::OpBuilderLike,
    dialect::bool::{CmpPredicate, CmpPredicateAttribute},
    error::Error,
    ident,
};

use llzk_sys::MlirOpBuilder;
use melior::ir::{
    Block, BlockLike, Location, Operation, OperationRef, RegionLike, Type, TypeLike, Value,
    ValueLike,
    attribute::StringAttribute,
    operation::{OperationBuilder, OperationLike},
    r#type::IntegerType,
};
use mlir_sys::{MlirLocation, MlirValue};

/// Defines a `is_$name` operation that checks if the given operation matches the expected
/// operation type.
macro_rules! isa_fn {
    ($name:ident, $op_name:expr) => {
        paste::paste! {
            #[doc = concat!("Returns `true` iff the given op is `bool.", $op_name, "`.")]
            #[inline]
            pub fn [<is_bool_ $name>]<'c: 'a, 'a>(op: &impl ::melior::ir::operation::OperationLike<'c, 'a>) -> bool {
                crate::operation::isa(op, concat!("bool.", $op_name))
            }
        }
    };
    ($name:ident) => {
        isa_fn!($name, stringify!($name));
    };
}

fn build_cmp_op<'c>(
    pred: CmpPredicate,
    location: Location<'c>,
    operands: &[Value<'c, '_>],
) -> Result<Operation<'c>, Error> {
    let ctx = location.context();
    OperationBuilder::new("bool.cmp", location)
        .add_results(&[IntegerType::new(unsafe { ctx.to_ref() }, 1).into()])
        .add_operands(operands)
        .add_attributes(&[(
            ident!(ctx, "predicate"),
            CmpPredicateAttribute::new(unsafe { ctx.to_ref() }, pred).into(),
        )])
        .build()
        .map_err(Into::into)
}

macro_rules! cmp_binop {
    ($name:ident, $pred:expr) => {
        #[doc = concat!("Creates a `bool.cmp ", stringify!($name) ,"` operation.")]
        pub fn $name<'c>(
            location: Location<'c>,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
        ) -> Result<Operation<'c>, Error> {
            build_cmp_op($pred, location, &[lhs, rhs])
        }
    };
}

cmp_binop!(eq, CmpPredicate::Eq);
cmp_binop!(ge, CmpPredicate::Ge);
cmp_binop!(gt, CmpPredicate::Gt);
cmp_binop!(le, CmpPredicate::Le);
cmp_binop!(lt, CmpPredicate::Lt);
cmp_binop!(ne, CmpPredicate::Ne);

isa_fn!(cmp);

fn build_op<'c>(
    name: &str,
    location: Location<'c>,
    operands: &[Value<'c, '_>],
) -> Result<Operation<'c>, Error> {
    let ctx = location.context();
    OperationBuilder::new(format!("bool.{name}").as_str(), location)
        .add_results(&[IntegerType::new(unsafe { ctx.to_ref() }, 1).into()])
        .add_operands(operands)
        .build()
        .map_err(Into::into)
}

macro_rules! binop {
    ($name:ident) => {
        binop!($name, stringify!($name));
    };
    ($name:ident, $opname:expr) => {
        #[doc = concat!("Creates a `bool.", $opname ,"` operation.")]
        pub fn $name<'c>(
            location: Location<'c>,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
        ) -> Result<Operation<'c>, Error> {
            build_op($opname, location, &[lhs, rhs])
        }

        isa_fn!($name);
    };
}

macro_rules! unop {
    ($name:ident) => {
        unop!($name, stringify!($name));
    };
    ($name:ident, $opname:expr) => {
        #[doc = concat!("Creates a `bool.", $opname ,"` operation.")]
        pub fn $name<'c>(
            location: Location<'c>,
            value: Value<'c, '_>,
        ) -> Result<Operation<'c>, Error> {
            build_op($opname, location, &[value])
        }

        isa_fn!($name);
    };
}

binop!(and);
binop!(or);
binop!(xor);
unop!(not);

/// Creates a `bool.assert` operation.
pub fn assert<'c>(
    location: Location<'c>,
    cond: Value<'c, '_>,
    msg: Option<&str>,
) -> Result<Operation<'c>, Error> {
    let ctx = location.context();
    let mut builder = OperationBuilder::new("bool.assert", location).add_operands(&[cond]);
    if let Some(msg) = msg {
        builder = builder.add_attributes(&[(
            ident!(ctx, "msg"),
            StringAttribute::new(unsafe { ctx.to_ref() }, msg).into(),
        )]);
    }
    builder.build().map_err(Into::into)
}

isa_fn!(assert);

/// Helper for creating a quantifier op.
fn create_quantifier_body<'c, 'a, B>(
    builder: &B,
    location: Location<'c>,
    domain: Value<'c, '_>,
    op_build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> Result<OperationRef<'c, 'a>, Error>
where
    B: OpBuilderLike<'c>,
{
    let op = unsafe {
        OperationRef::from_raw(op_build(
            builder.to_raw(),
            location.to_raw(),
            domain.to_raw(),
        ))
    };

    let region = op.region(0).map_err(Error::Melior)?;
    let iter_type = super::quantifier_iter_type(domain.r#type())?;
    region.append_block(Block::new(&[(iter_type, location)]));
    Ok(op)
}

/// Creates a `bool.forall` operation.
///
/// Adds an empty block with the correct iteration type based on the domain's type.
pub fn forall<'c, 'a, B>(
    builder: &B,
    location: Location<'c>,
    domain: Value<'c, '_>,
) -> Result<OperationRef<'c, 'a>, Error>
where
    B: OpBuilderLike<'c>,
{
    create_quantifier_body(builder, location, domain, llzk_sys::llzkBool_ForAllOpBuild)
}

isa_fn!(forall);

/// Creates a `bool.exists` operation.
///
/// Adds an empty block with the correct iteration type based on the domain's type.
pub fn exists<'c, 'a, B>(
    builder: &B,
    location: Location<'c>,
    domain: Value<'c, '_>,
) -> Result<OperationRef<'c, 'a>, Error>
where
    B: OpBuilderLike<'c>,
{
    create_quantifier_body(builder, location, domain, llzk_sys::llzkBool_ExistsOpBuild)
}

isa_fn!(exists);

/// Creates a `bool.yield` operation.
pub fn r#yield<'c, 'a, B: OpBuilderLike<'c>>(
    builder: &B,
    location: Location<'c>,
    value: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzk_sys::llzkBool_YieldOpBuild(
            builder.to_raw(),
            location.to_raw(),
            value.to_raw(),
        ))
    }
}

isa_fn!(r#yield);
