//! `array` dialect.

mod ops;
#[cfg(test)]
mod tests;
mod r#type;

use llzk_sys::mlirGetDialectHandle__llzk__array__;
use melior::dialect::DialectHandle;
pub use ops::{
    ArrayCtor, extract, insert, is_extract_op, is_insert_op, is_len_op, is_new_op, is_read_op,
    is_write_op, len, new, read, write,
};
pub use r#type::{ArrayType, is_array_type};

/// Returns a handle to the `array` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__array__()) }
}

/// Exports the common types of the array dialect.
pub mod prelude {
    pub use super::r#type::{ArrayType, is_array_type};
}
