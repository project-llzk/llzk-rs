//! Exports the most common types and function in llzk.

pub use crate::context::LlzkContext;
pub use crate::dialect::array::prelude::*;
pub use crate::dialect::bool::prelude::*;
pub use crate::dialect::felt::prelude::*;
pub use crate::dialect::function::prelude::*;
pub use crate::dialect::llzk::prelude::*;
pub use crate::dialect::module::{ModuleExt, llzk_module};
pub use crate::dialect::pod::prelude::*;
pub use crate::dialect::poly::prelude::*;
pub use crate::dialect::r#struct::prelude::*;
pub use crate::dialect::verif::prelude::*;
pub use crate::error::Error as LlzkError;
pub use crate::operation::{replace_uses_of_with, verify_operation, verify_operation_with_diags};
pub use crate::passes as llzk_passes;
pub use crate::symbol_ref::{SymbolRefAttrLike, SymbolRefAttribute};
pub use crate::symbol_table;
pub use crate::type_ext::*;
pub use crate::typing::{types_unify, types_unify_with_prefix};
pub use crate::utils::{IntoRef, print_block, print_operation, print_region};

/// Exports from the various llzk dialects.
pub mod dialect {

    /// Exports functions from the 'array' dialect
    pub mod array {
        pub use crate::dialect::array::{extract, insert, len, new, read, write};
        pub use crate::dialect::array::{
            is_array_type, is_extract_op, is_insert_op, is_len_op, is_new_op, is_read_op,
            is_write_op,
        };
    }

    /// Exports functions from the 'bool' dialect
    pub mod bool {
        pub use crate::dialect::bool::{and, assert, eq, ge, gt, le, lt, ne, not, or, xor};
        pub use crate::dialect::bool::{
            is_and_op, is_assert_op, is_cmp_op, is_not_op, is_or_op, is_xor_op,
        };
    }

    /// Exports functions from the 'cast' dialect
    pub mod cast {
        pub use crate::dialect::cast::{is_tofelt_op, is_toindex_op};
        pub use crate::dialect::cast::{tofelt, toindex};
    }

    /// Exports functions from the 'constrain' dialect
    pub mod constrain {
        pub use crate::dialect::constrain::{eq, r#in};
        pub use crate::dialect::constrain::{is_eq_op, is_in_op};
    }

    /// Exports functions from the 'felt' dialect
    pub mod felt {
        pub use crate::dialect::felt::{
            add, bit_and, bit_not, bit_or, bit_xor, constant, div, inv, mul, neg, pow, shl, shr,
            sintdiv, smod, sub, uintdiv, umod,
        };
        pub use crate::dialect::felt::{
            is_add_op, is_bit_and_op, is_bit_not_op, is_bit_or_op, is_bit_xor_op, is_const_op,
            is_div_op, is_felt_type, is_inv_op, is_mul_op, is_neg_op, is_pow_op, is_shl_op,
            is_shr_op, is_sintdiv_op, is_smod_op, is_sub_op, is_uintdiv_op, is_umod_op,
        };
    }

    /// Exports functions from the 'function' dialect
    pub mod function {
        pub use crate::dialect::function::{
            arg_name_attr, call, call_with_map_operands, call_with_template_params, def,
            def_with_signature_attrs, res_name_attr, r#return,
        };
        pub use crate::dialect::function::{is_call_op, is_def_op, is_return_op};
    }

    /// Exports functions from the 'global' dialect
    pub mod global {
        pub use crate::dialect::global::{def, read, write};
        pub use crate::dialect::global::{is_def_op, is_read_op, is_write_op};
    }

    /// Exports functions from the 'llzk' dialect
    pub mod llzk {
        pub use crate::dialect::llzk::{is_nondet_op, nondet};
    }

    /// Exports functions from the 'pod' dialect
    pub mod pod {
        pub use crate::dialect::pod::ops::{is_new_op, is_read_op, is_write_op};
        pub use crate::dialect::pod::ops::{new, new_with_affine_init, read, write};
    }

    /// Exports functions from the 'poly' dialect
    pub mod poly {
        pub use crate::dialect::poly::ops::{
            expr, is_expr_op, is_param_op, is_read_const_op, is_template_op, is_yield_op, param,
            read_const, template, r#yield,
        };
    }

    /// Exports functions from the 'ram' dialect
    pub mod ram {
        pub use crate::dialect::ram::{is_load_op, is_store_op};
        pub use crate::dialect::ram::{load, store};
    }

    /// Exports functions from the 'struct' dialect
    pub mod r#struct {
        pub use crate::dialect::r#struct::helpers;
        pub use crate::dialect::r#struct::{def, member, new, readm, readm_with_offset, writem};
        pub use crate::dialect::r#struct::{
            is_def_op, is_member_op, is_new_op, is_readm_op, is_struct_type, is_writem_op,
        };
    }

    /// Exports functions from the 'verif' dialect
    pub mod verif {
        pub use crate::dialect::verif::{
            contract, ensure_compute, ensure_constrain, include, include_with_map_operands,
            include_with_map_operands_slice, is_contract_op, is_ensure_compute_op,
            is_ensure_constrain_op, is_include_op, is_require_compute_op, is_require_constrain_op,
            require_compute, require_constrain,
        };
    }
}

/// Exports LLZK constants.
pub use llzk_sys::{
    FUNC_NAME_COMPUTE, FUNC_NAME_CONSTRAIN, FUNC_NAME_PRODUCT, LANG_ATTR_NAME, MAIN_ATTR_NAME,
};

/// melior reexports of commonly used types.
pub use melior::{
    Context, ContextRef, Error as MeliorError, StringRef,
    ir::{
        Location, Module, Region, RegionLike, RegionRef, Value, ValueLike,
        attribute::{
            Attribute, AttributeLike, BoolAttribute, FlatSymbolRefAttribute, IntegerAttribute,
            StringAttribute, TypeAttribute,
        },
        block::{Block, BlockArgument, BlockLike, BlockRef},
        operation::{
            Operation, OperationLike, OperationMutLike, OperationRef, OperationRefMut,
            OperationResult, WalkOrder, WalkResult,
        },
        r#type::{FunctionType, IntegerType, Type, TypeLike},
    },
    pass::{OperationPassManager, Pass, PassManager},
};

/// Reexport of the passes included in melior.
pub mod melior_passes {
    pub use melior::pass::r#async::*;
    pub use melior::pass::conversion::*;
    pub use melior::pass::gpu::*;
    pub use melior::pass::linalg::*;
    pub use melior::pass::sparse_tensor::*;
    pub use melior::pass::transform::*;
}

/// Reexport of the dialects included in melior.
pub mod melior_dialects {
    pub use melior::dialect::arith;
    /// Exports functions from the 'scf' dialect and extensions for LLZK.
    pub mod scf {
        pub use crate::dialect::scf_ext::{
            is_condition_op, is_for_op, is_if_op, is_while_op, is_yield_op,
        };
        pub use melior::dialect::scf::*;
    }
    pub use melior::dialect::index;
}
