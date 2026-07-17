use super::StructType;
use crate::{
    builder::{OpBuilder, OpBuilderLike},
    dialect::function::FuncDefOpRef,
    error::Error,
    macros::llzk_op_type,
    operation::detach_and_erase_op,
    prelude::SymbolRefAttribute,
};
use llzk_sys::{
    llzkOperationIsA_Struct_CreateStructOp, llzkOperationIsA_Struct_MemberDefOp,
    llzkOperationIsA_Struct_MemberReadOp, llzkOperationIsA_Struct_MemberWriteOp,
    llzkOperationIsA_Struct_StructDefOp, llzkStruct_CreateStructOpBuild,
    llzkStruct_MemberDefOpBuild, llzkStruct_MemberDefOpGetColumnValue,
    llzkStruct_MemberDefOpGetSignalValue, llzkStruct_MemberDefOpHasPublicAttr,
    llzkStruct_MemberDefOpSetColumnValue, llzkStruct_MemberDefOpSetPublicAttr,
    llzkStruct_MemberDefOpSetSignalValue, llzkStruct_MemberReadOpBuild,
    llzkStruct_MemberReadOpBuildWithLiteralDistance, llzkStruct_MemberWriteOpBuild,
    llzkStruct_StructDefOpBuild, llzkStruct_StructDefOpGetBody,
    llzkStruct_StructDefOpGetBodyRegion, llzkStruct_StructDefOpGetComputeFuncOp,
    llzkStruct_StructDefOpGetConstrainFuncOp, llzkStruct_StructDefOpGetFullyQualifiedName,
    llzkStruct_StructDefOpGetMemberDef, llzkStruct_StructDefOpGetMemberDefs,
    llzkStruct_StructDefOpGetNumMemberDefs, llzkStruct_StructDefOpGetNumTemplateExprOpNames,
    llzkStruct_StructDefOpGetNumTemplateParamOpNames, llzkStruct_StructDefOpGetProductFuncOp,
    llzkStruct_StructDefOpGetTemplateExprOpNames, llzkStruct_StructDefOpGetTemplateParamOpNames,
    llzkStruct_StructDefOpGetType, llzkStruct_StructDefOpGetTypeWithParams,
    llzkStruct_StructDefOpHasColumns, llzkStruct_StructDefOpIsMainComponent,
};
use melior::{
    StringRef,
    ir::{
        Attribute, AttributeLike as _, Block, BlockRef, Identifier, Location, OperationRef,
        RegionLike as _, RegionRef, Type, TypeLike as _, Value, ValueLike as _,
        attribute::{ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::{OperationLike, OperationMutLike},
    },
};
use mlir_sys::{MlirAttribute, MlirOperation};

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
        unsafe { Type::from_raw(llzkStruct_StructDefOpGetType(self.to_raw())) }
            .try_into()
            .expect("StructDefOpLike::type error")
    }

    /// Returns the symbol defined by this struct definition.
    ///
    /// # Panics
    ///
    /// If the 'struct.def' op doesn't have an attribute named `sym_name`.
    fn sym_name(&'a self) -> &'c str {
        self.attribute("sym_name")
            .and_then(StringAttribute::try_from)
            .map(|a| a.value())
            .unwrap()
    }

    /// Returns the single body Region of the StructDefOp.
    fn body_region(&self) -> RegionRef<'c, 'a> {
        unsafe { RegionRef::from_raw(llzkStruct_StructDefOpGetBodyRegion(self.to_raw())) }
    }

    /// Returns the single body Block within the StructDefOp's Region.
    fn body(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(llzkStruct_StructDefOpGetBody(self.to_raw())) }
    }

    /// Returns the associated StructType to this op using the given const params instead of the
    /// parameters defined by the op.
    ///
    /// # Panics
    ///
    /// If the 'struct.def' op type is not `!struct.type`.
    fn type_with_params(&self, params: ArrayAttribute<'c>) -> StructType<'c> {
        unsafe {
            Type::from_raw(llzkStruct_StructDefOpGetTypeWithParams(
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
    fn find_member_def(&self, name: &str) -> Option<MemberDefOpRef<'c, 'a>> {
        let raw_op = unsafe {
            llzkStruct_StructDefOpGetMemberDef(
                self.to_raw(),
                Identifier::new(self.context().to_ref(), name).to_raw(),
            )
        };
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
    fn find_or_create_member_def<F>(
        &self,
        name: &str,
        build_member_def: F,
    ) -> Result<MemberDefOpRef<'c, 'a>, Error>
    where
        F: FnOnce(&OpBuilder<'c, '_>) -> Result<MemberDefOpRef<'c, 'a>, Error>,
    {
        match self.find_member_def(name) {
            Some(f) => Ok(f),
            None => {
                let region = self.body_region();
                let block = region
                    .first_block()
                    .unwrap_or_else(|| region.append_block(Block::new(&[])));
                let builder = OpBuilder::at_block_end(unsafe { self.context().to_ref() }, block);
                build_member_def(&builder)
            }
        }
    }

    /// Returns a vector of the member definitions inside this struct.
    ///
    /// # Panics
    ///
    /// If any of the result operations is not a `struct.member` op.
    fn member_defs(&self) -> Vec<MemberDefOpRef<'c, 'a>> {
        let num_members =
            usize::try_from(unsafe { llzkStruct_StructDefOpGetNumMemberDefs(self.to_raw()) })
                .unwrap();
        let mut raw_ops: Vec<MlirOperation> = Vec::with_capacity(num_members);
        unsafe {
            llzkStruct_StructDefOpGetMemberDefs(self.to_raw(), raw_ops.as_mut_ptr());
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
        unsafe { llzkStruct_StructDefOpHasColumns(self.to_raw()) }.value != 0
    }

    /// Returns a [`FuncDefOpRef`] reference to the operation that defines the witness computation
    /// of the struct.
    ///
    /// # Panics
    ///
    /// If the result operation is not a `function.def`.
    fn compute_func<'b>(&self) -> Option<FuncDefOpRef<'c, 'b>> {
        let raw_op = unsafe { llzkStruct_StructDefOpGetComputeFuncOp(self.to_raw()) };
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
    fn constrain_func<'b>(&self) -> Option<FuncDefOpRef<'c, 'b>> {
        let raw_op = unsafe { llzkStruct_StructDefOpGetConstrainFuncOp(self.to_raw()) };
        if raw_op.ptr.is_null() {
            return None;
        }
        Some(
            unsafe { OperationRef::from_raw(raw_op) }
                .try_into()
                .expect("op of type 'function.def'"),
        )
    }

    /// Returns a [`FuncDefOpRef`] reference to the operation that defines the product body of the
    /// struct.
    ///
    /// # Panics
    ///
    /// If the result operation is not a `function.def`.
    fn product_func<'b>(&self) -> Option<FuncDefOpRef<'c, 'b>> {
        let raw_op = unsafe { llzkStruct_StructDefOpGetProductFuncOp(self.to_raw()) };
        if raw_op.ptr.is_null() {
            return None;
        }
        Some(
            unsafe { OperationRef::from_raw(raw_op) }
                .try_into()
                .expect("op of type 'function.def'"),
        )
    }

    /// Returns the names of all template parameters accessible by the struct,
    /// if the struct is within a template op. Otherwise, returns an empty vec.
    fn template_param_op_names(&self) -> Vec<FlatSymbolRefAttribute<'c>> {
        let num_attrs = usize::try_from(unsafe {
            llzkStruct_StructDefOpGetNumTemplateParamOpNames(self.to_raw())
        })
        .unwrap();
        let mut raw_attrs: Vec<MlirAttribute> = Vec::with_capacity(num_attrs);
        unsafe {
            llzkStruct_StructDefOpGetTemplateParamOpNames(self.to_raw(), raw_attrs.as_mut_ptr());
            raw_attrs.set_len(num_attrs);
        };
        raw_attrs
            .into_iter()
            .map(|attr| {
                FlatSymbolRefAttribute::try_from(unsafe { Attribute::from_raw(attr) }).unwrap()
            })
            .collect()
    }

    /// Returns the names of all template expressions accessible by the struct,
    /// if the struct is within a template op. Otherwise, returns an empty vec.
    fn template_expr_op_names(&self) -> Vec<FlatSymbolRefAttribute<'c>> {
        let num_attrs = usize::try_from(unsafe {
            llzkStruct_StructDefOpGetNumTemplateExprOpNames(self.to_raw())
        })
        .unwrap();
        let mut raw_attrs: Vec<MlirAttribute> = Vec::with_capacity(num_attrs);
        unsafe {
            llzkStruct_StructDefOpGetTemplateExprOpNames(self.to_raw(), raw_attrs.as_mut_ptr());
            raw_attrs.set_len(num_attrs);
        };
        raw_attrs
            .into_iter()
            .map(|attr| {
                FlatSymbolRefAttribute::try_from(unsafe { Attribute::from_raw(attr) }).unwrap()
            })
            .collect()
    }

    /// Returns a [`SymbolRefAttribute`] with the fully qualified name of the struct.
    ///
    /// # Panics
    ///
    /// If the fully qualified name is not a [`SymbolRefAttribute`].
    fn fully_qualified_name(&self) -> SymbolRefAttribute<'c> {
        unsafe { Attribute::from_raw(llzkStruct_StructDefOpGetFullyQualifiedName(self.to_raw())) }
            .try_into()
            .expect("symbol ref attribute")
    }

    /// Returns true if the struct is the main entry point of the circuit.
    fn is_main_component(&self) -> bool {
        unsafe { llzkStruct_StructDefOpIsMainComponent(self.to_raw()) }
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

llzk_op_type!(
    StructDefOp,
    llzkOperationIsA_Struct_StructDefOp,
    "struct.def"
);

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
    /// Returns true if the member is stored as a witness signal.
    fn signal(&self) -> bool {
        unsafe { llzkStruct_MemberDefOpGetSignalValue(self.to_raw()) }
    }

    /// Sets or unsets the `signal` attribute.
    fn set_signal(&self, value: bool) {
        unsafe {
            llzkStruct_MemberDefOpSetSignalValue(self.to_raw(), value);
        }
    }

    /// Returns true if the member supports offset table accesses.
    fn column(&self) -> bool {
        unsafe { llzkStruct_MemberDefOpGetColumnValue(self.to_raw()) }
    }

    /// Sets or unsets the `column` attribute.
    fn set_column(&self, value: bool) {
        unsafe {
            llzkStruct_MemberDefOpSetColumnValue(self.to_raw(), value);
        }
    }

    /// Returns true if the member op has a `llzk.pub` attribute.
    fn has_public_attr(&self) -> bool {
        unsafe { llzkStruct_MemberDefOpHasPublicAttr(self.to_raw()) }
    }

    /// Sets or unsets the `llzk.pub` attribute.
    fn set_public_attr(&self, value: bool) {
        unsafe {
            llzkStruct_MemberDefOpSetPublicAttr(self.to_raw(), value);
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

llzk_op_type!(
    MemberDefOp,
    llzkOperationIsA_Struct_MemberDefOp,
    "struct.member"
);

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOp<'c> {}

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOpRef<'c, 'a> {}

impl<'a, 'c: 'a> MemberDefOpLike<'c, 'a> for MemberDefOpRefMut<'c, 'a> {}

//===----------------------------------------------------------------------===//
// Operation factories
//===----------------------------------------------------------------------===//

/// Creates a 'struct.def' op and fills its body with operations produced by the callback.
///
/// Use [`crate::dialect::empty_region`] as the `fill` callback to leave the body empty so contents
/// can be added later.
pub fn def<'c, 'a, B>(
    builder: &B,
    location: Location<'c>,
    name: &str,
    fill: impl FnOnce(&B) -> Result<(), Error>,
) -> Result<StructDefOpRef<'c, 'a>, Error>
where
    B: OpBuilderLike<'c>,
{
    let ctx = location.context();
    let op = unsafe {
        OperationRef::from_raw(llzkStruct_StructDefOpBuild(
            builder.to_raw(),
            location.to_raw(),
            Identifier::new(ctx.to_ref(), name).to_raw(),
        ))
    };
    let op: StructDefOpRef<'c, 'a> = op.try_into()?;

    let region = op.body_region();
    let block = region
        .first_block()
        .unwrap_or_else(|| region.append_block(Block::new(&[])));

    let _guard = builder.insertion_guard();
    builder.set_insertion_point_at_start(block);
    match fill(builder) {
        Ok(()) => Ok(op),
        Err(err) => {
            detach_and_erase_op(op);
            Err(err)
        }
    }
}

crate::macros::isa_fn!(struct, def, llzkOperationIsA_Struct_StructDefOp);

/// Creates a 'struct.member' op
pub fn member<'c, 'a, T>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    name: &str,
    r#type: T,
    is_signal: bool,
    is_column: bool,
    is_public: bool,
) -> Result<MemberDefOpRef<'c, 'a>, Error>
where
    T: Into<Type<'c>>,
{
    let r#type = r#type.into();
    unsafe {
        OperationRef::from_raw(llzkStruct_MemberDefOpBuild(
            builder.to_raw(),
            location.to_raw(),
            StringRef::new(name).to_raw(),
            r#type.to_raw(),
            is_signal,
            is_column,
        ))
    }
    .try_into()
    .inspect(|op: &MemberDefOpRef<'c, 'a>| {
        op.set_public_attr(is_public);
    })
}

crate::macros::isa_fn!(struct, member, llzkOperationIsA_Struct_MemberDefOp);

/// Creates a 'struct.readm' op
pub fn readm<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result_type: Type<'c>,
    component: Value<'c, '_>,
    member_name: &str,
) -> Result<OperationRef<'c, 'a>, Error> {
    unsafe {
        let raw = llzkStruct_MemberReadOpBuild(
            builder.to_raw(),
            location.to_raw(),
            result_type.to_raw(),
            component.to_raw(),
            Identifier::new(result_type.context().to_ref(), member_name).to_raw(),
        );
        if raw.ptr.is_null() {
            Err(Error::BuildMethodFailed("readm"))
        } else {
            Ok(OperationRef::from_raw(raw))
        }
    }
}

/// Creates a 'struct.readm' op with an offset table access. This is only valid if the member is marked as a column.
pub fn readm_with_offset<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result_type: Type<'c>,
    component: Value<'c, '_>,
    member_name: &str,
    distance: i64,
) -> Result<OperationRef<'c, 'a>, Error> {
    unsafe {
        let raw = llzkStruct_MemberReadOpBuildWithLiteralDistance(
            builder.to_raw(),
            location.to_raw(),
            result_type.to_raw(),
            component.to_raw(),
            Identifier::new(result_type.context().to_ref(), member_name).to_raw(),
            distance,
        );
        if raw.ptr.is_null() {
            Err(Error::BuildMethodFailed("readm_with_offset"))
        } else {
            Ok(OperationRef::from_raw(raw))
        }
    }
}

crate::macros::isa_fn!(struct, readm, llzkOperationIsA_Struct_MemberReadOp);

/// Creates a 'struct.writem' op.
pub fn writem<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    component: Value<'c, '_>,
    member_name: &str,
    value: Value<'c, '_>,
) -> Result<OperationRef<'c, 'a>, Error> {
    let context = location.context();
    let member_name = FlatSymbolRefAttribute::new(unsafe { context.to_ref() }, member_name);
    Ok(unsafe {
        OperationRef::from_raw(llzkStruct_MemberWriteOpBuild(
            builder.to_raw(),
            location.to_raw(),
            component.to_raw(),
            value.to_raw(),
            member_name.to_raw(),
        ))
    })
}

crate::macros::isa_fn!(struct, writem, llzkOperationIsA_Struct_MemberWriteOp);

/// Creates a 'struct.new' op
pub fn new<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    r#type: StructType<'c>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkStruct_CreateStructOpBuild(
            builder.to_raw(),
            location.to_raw(),
            r#type.to_raw(),
        ))
    }
}

crate::macros::isa_fn!(struct, new, llzkOperationIsA_Struct_CreateStructOp);
