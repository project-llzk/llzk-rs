//! `poly` dialect.

pub mod ops;
pub mod r#type;
pub use ops::{applymap, read_const, unifiable_cast};
pub use ops::{is_applymap_op, is_read_const_op, is_unifiable_cast_op};
pub use r#type::{TVarType, is_type_variable};

use llzk_sys::mlirGetDialectHandle__llzk__polymorphic__;
use melior::dialect::DialectHandle;

/// Returns a handle to the `poly` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__polymorphic__()) }
}

/// Exports the common types of the poly dialect.
pub mod prelude {
    pub use super::r#type::{TVarType, is_type_variable};
}
