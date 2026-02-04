//! `struct` dialect.

pub mod helpers;
mod ops;
mod r#type;

use llzk_sys::mlirGetDialectHandle__llzk__component__;
use melior::dialect::DialectHandle;
pub use ops::{
    MemberDefOp, MemberDefOpLike, MemberDefOpRef, StructDefOp, StructDefOpLike, StructDefOpMutLike,
    StructDefOpRef, def, member, new, readm, readm_with_offset, writem,
};
pub use ops::{is_struct_def, is_struct_member, is_struct_new, is_struct_readm, is_struct_writem};
pub use r#type::{StructType, is_struct_type};

/// Returns a handle to the `struct` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__component__()) }
}

/// Exports the common types of the struct dialect.
pub mod prelude {
    pub use super::ops::{
        MemberDefOp, MemberDefOpLike, MemberDefOpRef, MemberDefOpRefMut, StructDefOp, StructDefOpLike,
        StructDefOpMutLike, StructDefOpRef, StructDefOpRefMut,
    };
    pub use super::r#type::{StructType, is_struct_type};
}
