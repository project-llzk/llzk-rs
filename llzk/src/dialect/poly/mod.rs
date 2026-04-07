//! `poly` dialect.

pub mod ops;
pub mod r#type;
pub use ops::{
    TemplateExprOp, TemplateExprOpLike, TemplateOp, TemplateOpLike, TemplateParamOp,
    TemplateParamOpLike, YieldOp,
};
pub use ops::{
    applymap, expr, is_applymap_op, is_expr_op, is_param_op, is_read_const_op, is_template_op,
    is_unifiable_cast_op, is_yield_op, param, read_const, template, unifiable_cast, r#yield,
};
pub use r#type::{TVarType, is_type_variable};

use llzk_sys::mlirGetDialectHandle__llzk__polymorphic__;
use melior::dialect::DialectHandle;

/// Returns a handle to the `poly` dialect.
pub fn handle() -> DialectHandle {
    unsafe { DialectHandle::from_raw(mlirGetDialectHandle__llzk__polymorphic__()) }
}

/// Exports the common types of the poly dialect.
pub mod prelude {
    pub use super::ops::{
        TemplateExprOp, TemplateExprOpLike, TemplateOp, TemplateOpLike, TemplateParamOp,
        TemplateParamOpLike, YieldOp,
    };
    pub use super::r#type::{TVarType, is_type_variable};
}
