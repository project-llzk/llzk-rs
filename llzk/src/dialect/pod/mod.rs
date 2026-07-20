//! `pod` dialect.

pub mod attrs;
pub mod ops;
pub mod r#type;
pub use ops::{is_new_op, is_read_op, is_write_op};
pub use ops::{new, new_with_affine_init, read, write};

use llzk_sys::mlirGetDialectHandle__llzk__pod__;
use melior::dialect::DialectHandle;

/// Returns a handle to the `pod` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__pod__()) }
}

/// Exports the common types and records of the pod dialect.
pub mod prelude {
    pub use super::attrs::PodRecordAttribute;
    pub use super::ops::RecordValue;
    pub use super::r#type::{PodType, is_pod_type};
}
