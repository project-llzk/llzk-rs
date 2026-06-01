use crate::attributes::NamedAttribute;
use melior::{
    Context,
    ir::{Identifier, attribute::StringAttribute},
};

/// Creates a `function.arg_name` named attribute.
pub fn arg_name_attr<'c>(context: &'c Context, name: &str) -> NamedAttribute<'c> {
    (
        Identifier::new(context, "function.arg_name"),
        StringAttribute::new(context, name).into(),
    )
}

/// Creates a `function.res_name` named attribute.
pub fn res_name_attr<'c>(context: &'c Context, name: &str) -> NamedAttribute<'c> {
    (
        Identifier::new(context, "function.res_name"),
        StringAttribute::new(context, name).into(),
    )
}
