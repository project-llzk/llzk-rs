//! Utilities related to MLIR attributes.

use melior::ir::{Attribute, Identifier};

pub mod array;

/// An attribute associated to a name.
pub type NamedAttribute<'c> = (Identifier<'c>, Attribute<'c>);
