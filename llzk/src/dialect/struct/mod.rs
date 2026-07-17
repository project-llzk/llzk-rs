//! `struct` dialect.

pub mod helpers;
mod ops;
mod r#type;

use llzk_sys::mlirGetDialectHandle__llzk__component__;
use melior::dialect::DialectHandle;
pub use ops::{
    MemberDefOp, MemberDefOpLike, MemberDefOpRef, StructDefOp, StructDefOpLike, StructDefOpMutLike,
    StructDefOpRef, def, is_def_op, is_member_op, is_new_op, is_readm_op, is_writem_op, member,
    new, readm, readm_with_offset, writem,
};
pub use r#type::{StructType, is_struct_type};

/// Returns a handle to the `struct` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__component__()) }
}

/// Exports the common types of the struct dialect.
pub mod prelude {
    pub use super::{
        ops::{
            MemberDefOp, MemberDefOpLike, MemberDefOpRef, MemberDefOpRefMut, StructDefOp,
            StructDefOpLike, StructDefOpMutLike, StructDefOpRef, StructDefOpRefMut,
        },
        r#type::{StructType, is_struct_type},
    };
}
