use llzk_sys::{
    llzkMemberDefOpGetHasPublicAttr, llzkMemberDefOpSetPublicAttr, llzkMemberReadOpBuild,
    llzkOperationIsAMemberDefOp, llzkOperationIsAStructDefOp, llzkStructDefOpGetBody,
    llzkStructDefOpGetBodyRegion, llzkStructDefOpGetComputeFuncOp,
    llzkStructDefOpGetConstrainFuncOp, llzkStructDefOpGetHasColumns,
    llzkStructDefOpGetHasParamName, llzkStructDefOpGetIsMainComponent, llzkStructDefOpGetMemberDef,
    llzkStructDefOpGetMemberDefs, llzkStructDefOpGetNumMemberDefs, llzkStructDefOpGetType,
    llzkStructDefOpGetTypeWithParams,
};
use melior::{
    StringRef,
    ir::{
        Attribute, AttributeLike, Block, BlockLike as _, BlockRef, Location, Operation,
        OperationRef, Region, RegionLike as _, RegionRef, Type, TypeLike, Value, ValueLike,
        attribute::{ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike, OperationMutLike},
    },
};
use mlir_sys::MlirOperation;

use crate::{
    builder::{OpBuilder, OpBuilderLike},
    dialect::function::FuncDefOpRef,
    error::Error,
    ident,
    macros::llzk_op_type,
};

use super::StructType;

//===----------------------------------------------------------------------===//
// StructDefOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the 'struct.def' op.
pub trait StructDefOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns the associated StructType to this op using the const params defined by the op.
    ///
    /// # Panics
    ///
    /// If the 'struct.def' op type is not `!struct.type`.
    fn r#type(&self) -> StructType<'c> {
        unsafe { Type::from_raw(llzkStructDefOpGetType(self.to_raw())) }
            .try_into()
            .expect("StructDefOpLike::type error")
    }

    /// Returns the name of the struct
    ///
    /// # Panics
    ///
    /// If the 'struct.def' op doesn't have an attribute named `sym_name`.
    fn name(&'a self) -> &'c str {
        self.attribute("sym_name")
            .and_then(StringAttribute::try_from)
            .map(|a| a.value())
            .unwrap()
    }

    /// Returns the single body Region of the StructDefOp.
    fn body_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkStructDefOpGetBodyRegion(self.to_raw())) }
    }

    /// Returns the single body Block within the StructDefOp's Region.
    fn body(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(llzkStructDefOpGetBody(self.to_raw())) }
    }

    /// Returns the associated StructType to this op using the given const params instead of the
    /// parameters defined by the op.
    ///
    /// # Panics
    ///
    /// If the 'struct.def' op type is not `!struct.type`.
    fn type_with_params(&self, params: ArrayAttribute<'c>) -> StructType<'c> {
        unsafe {
            Type::from_raw(llzkStructDefOpGetTypeWithParams(
                self.to_raw(),
                params.to_raw(),
            ))
        }
        .try_into()
        .expect("StructDefOpLike::type error")
    }

    /// Returns the operation that defines the member with the given name, if present.
    ///
    /// # Panics
    ///
    /// If the nested symbol operation with the given name is not a `struct.member`.
    fn get_member_def(&self, name: &str) -> Option<MemberDefOpRef<'c, 'a>> {
        let name = StringRef::new(name);
        let raw_op = unsafe { llzkStructDefOpGetMemberDef(self.to_raw(), name.to_raw()) };
        if raw_op.ptr.is_null() {
            return None;
        }
        Some(
            unsafe { OperationRef::from_raw(raw_op) }
                .try_into()
                .expect("op of type 'struct.member'"),
        )
    }

    /// Returns the operation that defines the member with the given name, creating a new operation
    /// if not present.
    fn get_or_create_member_def<F>(&self, name: &str, f: F) -> Result<MemberDefOpRef<'c, 'a>, Error>
    where
        F: FnOnce() -> Result<MemberDefOp<'c>, Error>,
    {
        match self.get_member_def(name) {
            Some(f) => Ok(f),
            None => {
                let op = f()?;
                let region = self.region(0)?;
                let block = region
                    .first_block()
                    .unwrap_or_else(|| region.append_block(Block::new(&[])));

                let member_ref = block.append_operation(op.into());

                Ok(member_ref.try_into()?)
            }
        }
    }

    /// Fills the given array with the MemberDefOp operations inside this struct.
    ///
    /// # Panics
    ///
    /// If any of the result operations is not a `struct.member` op.
    fn get_member_defs(&self) -> Vec<MemberDefOpRef<'c, '_>> {
        let num_members =
            usize::try_from(unsafe { llzkStructDefOpGetNumMemberDefs(self.to_raw()) }).unwrap();
        let mut raw_ops: Vec<MlirOperation> = Vec::with_capacity(num_members);
        unsafe {
            llzkStructDefOpGetMemberDefs(self.to_raw(), raw_ops.as_mut_ptr());
            raw_ops.set_len(num_members);
        };
        raw_ops
            .into_iter()
            .map(|op| {
                unsafe { OperationRef::from_raw(op) }
                    .try_into()
                    .expect("op of type 'struct.member'")
            })
            .collect()
    }

    /// Returns true if the struct has members marked as columns.
    fn has_columns(&self) -> bool {
        unsafe { llzkStructDefOpGetHasColumns(self.to_raw()) }.value != 0
    }

    /// Returns a [`FuncDefOpRef`] reference to the operation that defines the witness computation
    /// of the struct.
    ///
    /// # Panics
    ///
    /// If the result operation is not a `function.def`.
    fn get_compute_func<'b>(&self) -> Option<FuncDefOpRef<'c, 'b>> {
        let raw_op = unsafe { llzkStructDefOpGetComputeFuncOp(self.to_raw()) };
        if raw_op.ptr.is_null() {
            return None;
        }
        Some(
            unsafe { OperationRef::from_raw(raw_op) }
                .try_into()
                .expect("op of type 'function.def'"),
        )
    }

    /// Returns a [`FuncDefOpRef`] reference to the operation that defines the constraints of the
    /// struct.
    ///
    /// # Panics
    ///
    /// If the result operation is not a `function.def`.
    fn get_constrain_func<'b>(&self) -> Option<FuncDefOpRef<'c, 'b>> {
        let raw_op = unsafe { llzkStructDefOpGetConstrainFuncOp(self.to_raw()) };
        if raw_op.ptr.is_null() {
            return None;
        }
        Some(
            unsafe { OperationRef::from_raw(raw_op) }
                .try_into()
                .expect("op of type 'function.def'"),
        )
    }

    /// Returns true if the struct has a parameter with the given name.
    fn has_param_name(&self, name: &str) -> bool {
        let name = StringRef::new(name);
        unsafe { llzkStructDefOpGetHasParamName(self.to_raw(), name.to_raw()) }
    }

    /// Returns a StringAttr with the fully qualified name of the struct.
    fn get_fully_qualified_name(&self) -> Attribute<'_> {
        todo!("melior does not have a SymbolRefAttribute type")
    }

    /// Returns true if the struct is the main entry point of the circuit.
    fn is_main_component(&self) -> bool {
        unsafe { llzkStructDefOpGetIsMainComponent(self.to_raw()) }
    }
}

/// Defines the mutable public API of the 'struct.def' op.
pub trait StructDefOpMutLike<'c: 'a, 'a>:
    StructDefOpLike<'c, 'a> + OperationMutLike<'c, 'a>
{
}

//===----------------------------------------------------------------------===//
// StructDefOp, StructDefOpRef, and StructDefOpRefMut
//===----------------------------------------------------------------------===//

llzk_op_type!(StructDefOp, llzkOperationIsAStructDefOp, "struct.def");

impl<'a, 'c: 'a> StructDefOpLike<'c, 'a> for StructDefOp<'c> {}

impl<'a, 'c: 'a> StructDefOpLike<'c, 'a> for StructDefOpRef<'c, 'a> {}

impl<'a, 'c: 'a> StructDefOpLike<'c, 'a> for StructDefOpRefMut<'c, 'a> {}

impl<'a, 'c: 'a> StructDefOpMutLike<'c, 'a> for StructDefOp<'c> {}

impl<'a, 'c: 'a> StructDefOpMutLike<'c, 'a> for StructDefOpRefMut<'c, 'a> {}

//===----------------------------------------------------------------------===//
// MemberDefOpLike
//===----------------------------------------------------------------------===//

/// Defines the public API of the 'struct.member' op.
pub trait MemberDefOpLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Returns true if the member op has a `llzk.pub` attribute.
    fn has_public_attr(&self) -> bool {
        unsafe { llzkMemberDefOpGetHasPublicAttr(self.to_raw()) }
    }

    /// Sets or unsets the `llzk.pub` attribute.
    fn set_public_attr(&self, value: bool) {
        unsafe {
            llzkMemberDefOpSetPublicAttr(self.to_raw(), value);
        }
    }

    /// Returns the name of the member.
    ///
    /// # Panics
    ///
    /// If the 'struct.member' op doesn't have an attribute named `sym_name`.
    fn member_name(&self) -> &'c str {
        self.attribute("sym_name")
            .and_then(StringAttribute::try_from)
            .expect("malformed 'struct.member' op")
            .value()
    }

    /// Returns the type of the member.
    ///
    /// # Panics
    ///
    /// If the 'struct.member' op doesn't have a attribute named `type`.
    fn member_type(&self) -> Type<'c> {
        self.attribute("type")
            .and_then(TypeAttribute::try_from)
            .expect("malformed 'struct.member' op")
            .value()
    }
}

//===----------------------------------------------------------------------===//
// MemberDefOp, MemberDefOpRef, MemberDefOpRefMut
//===----------------------------------------------------------------------===//

llzk_op_type!(MemberDefOp, llzkOperationIsAMemberDefOp, "struct.member");

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOp<'c> {}

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOpRef<'c, 'a> {}

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOpRefMut<'c, 'a> {}

//===----------------------------------------------------------------------===//
// Operation factories
//===----------------------------------------------------------------------===//

/// Creates a 'struct.def' op
pub fn def<'c, I>(
    location: Location<'c>,
    name: &str,
    params: &[&str],
    region_ops: I,
) -> Result<StructDefOp<'c>, Error>
where
    I: IntoIterator<Item = Result<Operation<'c>, Error>>,
{
    let ctx = location.context();
    let params: Vec<Attribute> = params
        .iter()
        .map(|a| FlatSymbolRefAttribute::new(unsafe { ctx.to_ref() }, a).into())
        .collect();
    let params = ArrayAttribute::new(unsafe { ctx.to_ref() }, &params).into();
    let region = Region::new();
    let block = Block::new(&[]);
    region_ops
        .into_iter()
        .try_for_each(|op| -> Result<(), Error> {
            block.append_operation(op?);
            Ok(())
        })?;
    region.append_block(block);
    let name: Attribute = StringAttribute::new(unsafe { ctx.to_ref() }, name).into();
    let attrs = [
        (ident!(ctx, "sym_name"), name),
        (ident!(ctx, "const_params"), params),
    ];

    OperationBuilder::new("struct.def", location)
        .add_attributes(&attrs)
        .add_regions([region])
        .build()
        .map_err(Into::into)
        .and_then(TryInto::try_into)
}

/// Return `true` iff the given op is `struct.def`.
#[inline]
pub fn is_struct_def<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "struct.def")
}

/// Creates a 'struct.member' op
pub fn member<'c, T>(
    location: Location<'c>,
    name: &str,
    r#type: T,
    is_column: bool,
    is_public: bool,
) -> Result<MemberDefOp<'c>, Error>
where
    T: Into<Type<'c>>,
{
    let ctx = location.context();
    let r#type = TypeAttribute::new(r#type.into());
    let mut builder = OperationBuilder::new("struct.member", location).add_attributes(&[
        (
            ident!(ctx, "sym_name"),
            StringAttribute::new(unsafe { ctx.to_ref() }, name).into(),
        ),
        (ident!(ctx, "type"), r#type.into()),
    ]);

    builder = if is_column {
        builder.add_attributes(&[(
            ident!(ctx, "column"),
            Attribute::unit(unsafe { ctx.to_ref() }),
        )])
    } else {
        builder
    };

    builder
        .build()
        .map_err(Into::into)
        .and_then(TryInto::try_into)
        .inspect(|op: &MemberDefOp<'c>| op.set_public_attr(is_public))
}

/// Return `true` iff the given op is `struct.member`.
#[inline]
pub fn is_struct_member<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "struct.member")
}

/// Creates a 'struct.readm' op
pub fn readm<'c>(
    builder: &OpBuilder<'c>,
    location: Location<'c>,
    result_type: Type<'c>,
    component: Value<'c, '_>,
    member_name: &str,
) -> Result<Operation<'c>, Error> {
    let member_name = StringRef::new(member_name);
    unsafe {
        let raw = llzkMemberReadOpBuild(
            builder.to_raw(),
            location.to_raw(),
            result_type.to_raw(),
            component.to_raw(),
            member_name.to_raw(),
        );
        if raw.ptr.is_null() {
            Err(Error::BuildMethodFailed("readm"))
        } else {
            Ok(Operation::from_raw(raw))
        }
    }
}

/// Creates a 'struct.readm' op.
///
/// This factory method is not implemented yet.
pub fn readm_with_offset<'c>() -> Operation<'c> {
    todo!()
}

/// Return `true` iff the given op is `struct.readm`.
#[inline]
pub fn is_struct_readm<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "struct.readm")
}

/// Creates a 'struct.writem' op.
pub fn writem<'c>(
    location: Location<'c>,
    component: Value<'c, '_>,
    member_name: &str,
    value: Value<'c, '_>,
) -> Result<Operation<'c>, Error> {
    let context = location.context();
    let member_name = FlatSymbolRefAttribute::new(unsafe { context.to_ref() }, member_name);
    let attrs = [(ident!(context, "member_name"), member_name.into())];
    OperationBuilder::new("struct.writem", location)
        .add_operands(&[component, value])
        .add_attributes(&attrs)
        .build()
        .map_err(Into::into)
}

/// Return `true` iff the given op is `struct.writem`.
#[inline]
pub fn is_struct_writem<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "struct.writem")
}

/// Creates a 'struct.new' op
pub fn new<'c>(location: Location<'c>, r#type: StructType<'c>) -> Operation<'c> {
    OperationBuilder::new("struct.new", location)
        .add_results(&[r#type.into()])
        .build()
        .expect("valid operation")
}

/// Return `true` iff the given op is `struct.new`.
#[inline]
pub fn is_struct_new<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "struct.new")
}
