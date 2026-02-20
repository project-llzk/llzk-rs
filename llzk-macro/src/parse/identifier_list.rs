//! List of identifiers. Based on [melior]'s.
//!
//! [melior]: https://github.com/mlir-rs/melior/blob/main/macro/src/parse/identifier_list.rs.

use proc_macro2::Ident;
use syn::{
    Attribute, Result, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub struct Identifier {
    pub attrs: Vec<Attribute>,
    pub ident: Ident,
}

impl Parse for Identifier {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            attrs: input.call(Attribute::parse_outer)?,
            ident: input.parse()?,
        })
    }
}

/// Represents a comma-separated list of identifiers. Used as the DSL for the [`crate::conversion_passes`] macro.
pub struct IdentifierList {
    identifiers: Vec<Identifier>,
}
impl IdentifierList {
    pub fn identifiers(&self) -> &[Identifier] {
        &self.identifiers
    }
}
impl Parse for IdentifierList {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            identifiers: Punctuated::<Identifier, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        })
    }
}
