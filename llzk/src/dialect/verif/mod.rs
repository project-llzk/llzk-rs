//! `verif` dialect.

mod ops;

use llzk_sys::mlirGetDialectHandle__llzk__verif__;
use melior::dialect::DialectHandle;

pub use ops::{
    ConditionOpLike, ContractInputsIter, ContractOp, ContractOpLike, ContractOpRef,
    EnsureComputeOp, EnsureComputeOpRef, EnsureConstrainOp, EnsureConstrainOpRef,
    IncludeArgOperandsIter, IncludeOp, IncludeOpLike, IncludeOpRef, IncludeOpRefMut,
    RequireComputeOp, RequireComputeOpRef, RequireConstrainOp, RequireConstrainOpRef, contract,
    ensure_compute, ensure_constrain, include, include_with_map_operands,
    include_with_map_operands_slice, is_contract, is_ensure_compute, is_ensure_constrain,
    is_include, is_require_compute, is_require_constrain, require_compute, require_constrain,
};

/// Returns a handle to the `verif` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__verif__()) }
}

/// Exports the common types of the verif dialect.
pub mod prelude {
    pub use super::ops::{
        ConditionOpLike, ContractInputsIter, ContractOp, ContractOpLike, ContractOpRef,
        ContractOpRefMut, EnsureComputeOp, EnsureComputeOpRef, EnsureComputeOpRefMut,
        EnsureConstrainOp, EnsureConstrainOpRef, EnsureConstrainOpRefMut, IncludeArgOperandsIter,
        IncludeOp, IncludeOpLike, IncludeOpRef, IncludeOpRefMut, RequireComputeOp,
        RequireComputeOpRef, RequireComputeOpRefMut, RequireConstrainOp, RequireConstrainOpRef,
        RequireConstrainOpRefMut,
    };
}
