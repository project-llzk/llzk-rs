use crate::attributes::NamedAttribute;
use llzk_sys::{FUNCTION_ARG_NAME_ATTR_NAME, FUNCTION_RES_NAME_ATTR_NAME};
use melior::{
    Context,
    ir::{Identifier, attribute::StringAttribute},
};

/// Creates a `FUNCTION_ARG_NAME_ATTR_NAME` named attribute.
pub fn arg_name_attr<'c>(context: &'c Context, name: &str) -> NamedAttribute<'c> {
    (
        Identifier::new(context, FUNCTION_ARG_NAME_ATTR_NAME.as_ref()),
        StringAttribute::new(context, name).into(),
    )
}

/// Creates a `FUNCTION_RES_NAME_ATTR_NAME` named attribute.
pub fn res_name_attr<'c>(context: &'c Context, name: &str) -> NamedAttribute<'c> {
    (
        Identifier::new(context, FUNCTION_RES_NAME_ATTR_NAME.as_ref()),
        StringAttribute::new(context, name).into(),
    )
}
