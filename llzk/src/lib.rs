#![doc = include_str!("../README.md")]
#![cfg_attr(test, allow(unused_crate_dependencies))]

use llzk_sys::llzkRegisterCoreDialects;
use melior::dialect::DialectRegistry;

pub mod affine;
pub mod attributes;
pub mod builder;
pub mod context;
mod diagnostics;
pub mod dialect;
pub mod error;
mod macros;
pub mod map_operands;
pub mod operation;
pub mod passes;
pub mod prelude;
pub mod symbol_lookup;
pub mod symbol_ref;
pub mod symbol_table;
pub mod targets;
#[cfg(test)]
mod test;
pub mod type_ext;
pub mod typing;
pub mod utils;
pub mod value_ext;

/// Adds all core LLZK dialects into the given registry.
pub fn register_core_llzk_dialects(registry: &DialectRegistry) {
    unsafe { llzkRegisterCoreDialects(registry.to_raw()) }
}

/// Adds R1CS backend dialects into the given registry.
pub fn register_r1cs_dialects(registry: &DialectRegistry) {
    unsafe { llzk_sys::llzkRegisterR1CSDialects(registry.to_raw()) }
}

/// Adds PCL backend dialects into the given registry.
///
/// Does nothing if LLZK was compiled without the PCL backend.
pub fn register_pcl_dialects(registry: &DialectRegistry) {
    unsafe { llzk_sys::llzkRegisterPCLDialects(registry.to_raw()) }
}

/// Adds ZKLean backend dialects into the given registry.
pub fn register_zklean_dialects(registry: &DialectRegistry) {
    unsafe { llzk_sys::llzkRegisterZKLeanDialects(registry.to_raw()) }
}

/// Adds SMT backend dialects into the given registry.
pub fn register_smt_dialects(registry: &DialectRegistry) {
    unsafe { llzk_sys::llzkRegisterSMTDialects(registry.to_raw()) }
}
