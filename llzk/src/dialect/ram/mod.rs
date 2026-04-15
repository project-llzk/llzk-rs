//! `ram` dialect.

mod ops;
#[cfg(test)]
mod tests;

use llzk_sys::mlirGetDialectHandle__llzk__ram__;
use melior::dialect::DialectHandle;
pub use ops::{is_ram_load, is_ram_store, load, store};

/// Returns a handle to the `ram` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__ram__()) }
}
