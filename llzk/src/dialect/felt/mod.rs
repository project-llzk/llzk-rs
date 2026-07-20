//! `felt` dialect.

mod attrs;
mod ops;
mod r#type;

pub use attrs::{FeltConstAttribute, FieldSpecAttribute};
use llzk_sys::mlirGetDialectHandle__llzk__felt__;
use melior::dialect::DialectHandle;
pub use ops::{
    add, bit_and, bit_not, bit_or, bit_xor, constant, div, inv, mul, neg, pow, shl, shr, sintdiv,
    smod, sub, uintdiv, umod,
};
pub use ops::{
    is_add_op, is_bit_and_op, is_bit_not_op, is_bit_or_op, is_bit_xor_op, is_const_op, is_div_op,
    is_inv_op, is_mul_op, is_neg_op, is_pow_op, is_shl_op, is_shr_op, is_sintdiv_op, is_smod_op,
    is_sub_op, is_uintdiv_op, is_umod_op,
};
pub use r#type::{FeltType, is_felt_type};

/// Returns a handle to the `felt` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__felt__()) }
}

/// Exports the common types of the felt dialect.
pub mod prelude {
    pub use super::attrs::{FeltConstAttribute, FieldSpecAttribute};
    pub use super::r#type::{FeltType, is_felt_type};
}
