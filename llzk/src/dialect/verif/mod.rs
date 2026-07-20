//! `verif` dialect.

mod ops;

use llzk_sys::mlirGetDialectHandle__llzk__verif__;
use melior::dialect::DialectHandle;

pub use ops::{
    ConditionOpLike, ContractInputsIter, ContractOp, ContractOpLike, ContractOpRef,
    EnsureComputeOp, EnsureComputeOpRef, EnsureConstrainOp, EnsureConstrainOpRef,
    IncludeArgOperandsIter, IncludeOp, IncludeOpLike, IncludeOpRef, IncludeOpRefMut, InvariantOp,
    InvariantOpLike, InvariantOpMutLike, InvariantOpRef, InvariantOpRefMut, RequireComputeOp,
    RequireComputeOpRef, RequireConstrainOp, RequireConstrainOpRef, contract, contract_end,
    decreases, ensure_compute, ensure_constrain, include, include_with_map_operands,
    include_with_map_operands_slice, increases, invariant, invariant_build, is_contract_end_op,
    is_contract_op, is_decreases_op, is_ensure_compute_op, is_ensure_constrain_op, is_include_op,
    is_increases_op, is_invariant_op, is_old_op, is_require_compute_op, is_require_constrain_op,
    is_step_op, is_step_yield_op, old, require_compute, require_constrain, step, step_build,
    step_yield,
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
        IncludeOp, IncludeOpLike, IncludeOpRef, IncludeOpRefMut, InvariantOp, InvariantOpLike,
        InvariantOpMutLike, InvariantOpRef, InvariantOpRefMut, RequireComputeOp,
        RequireComputeOpRef, RequireComputeOpRefMut, RequireConstrainOp, RequireConstrainOpRef,
        RequireConstrainOpRefMut,
    };
}
