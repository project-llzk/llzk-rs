//! Implementation of `!struct.type` type.

use crate::attributes::array::ArrayAttribute;
use crate::error::Error;
use crate::symbol_lookup::SymbolLookupResult;
use crate::utils::FromRaw;
use crate::utils::IsA;
use llzk_sys::llzkStructTypeGetParams;
use llzk_sys::{llzkStructTypeGetName, llzkStructTypeGetWithArrayAttr, llzkTypeIsAStructType};
use melior::ir::Module;
use melior::ir::operation::OperationLike;
use melior::{
    Context,
    ir::{Attribute, AttributeLike as _, Type, TypeLike, attribute::FlatSymbolRefAttribute},
};
use mlir_sys::MlirLogicalResult;
use mlir_sys::MlirType;

/// Represents the `!struct.type` type.
#[derive(Copy, Clone, Debug)]
pub struct StructType<'c> {
    t: Type<'c>,
}

impl<'c> StructType<'c> {
    /// Creates a new struct type.
    ///
    /// The params array must match the number of params and their kind as defined by the associated
    /// `struct.def` operation.
    pub fn new(name: FlatSymbolRefAttribute<'c>, params: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(llzkStructTypeGetWithArrayAttr(
                name.to_raw(),
                ArrayAttribute::new(name.context().to_ref(), params).to_raw(),
            ))
        }
    }

    /// Creates a new struct type from a string reference.
    ///
    /// The returned type won't have any parameters.
    pub fn from_str(context: &'c Context, name: &str) -> Self {
        Self::new(FlatSymbolRefAttribute::new(context, name), &[])
    }

    /// Creates a new struct type from string references for both its name and parameter names.
    pub fn from_str_params(context: &'c Context, name: &str, params: &[&str]) -> Self {
        let params: Vec<Attribute> = params
            .iter()
            .map(|param| FlatSymbolRefAttribute::new(context, param).into())
            .collect();
        Self::new(FlatSymbolRefAttribute::new(context, name), &params)
    }

    /// Get the struct's name.
    pub fn name(&self) -> FlatSymbolRefAttribute<'c> {
        FlatSymbolRefAttribute::try_from(unsafe {
            Attribute::from_raw(llzkStructTypeGetName(self.to_raw()))
        })
        .expect("struct type must be constructed from FlatSymbolRefAttribute")
    }

    /// Get the struct's params.
    pub fn params(&self) -> ArrayAttribute<'c> {
        ArrayAttribute::try_from(unsafe {
            Attribute::from_raw(llzkStructTypeGetParams(self.to_raw()))
        })
        .expect("struct type's params must be an array attribute")
    }

    /// Get the struct's params as a vector of attributes.
    pub fn params_vec(&self) -> Vec<Attribute<'c>> {
        self.params().into_iter().collect()
    }

    /// Actual implementation of the [`get_definition`](Self::get_definition) and
    /// [`get_definition_from_module`](Self::get_definition_from_module) methods.
    fn get_definition_impl<O>(
        &self,
        o: O,
        f: unsafe extern "C" fn(
            MlirType,
            O,
            *mut llzk_sys::LlzkSymbolLookupResult,
        ) -> MlirLogicalResult,
    ) -> Result<SymbolLookupResult<'c>, Error> {
        let mut lookup = SymbolLookupResult::new();
        let result = unsafe { f(self.to_raw(), o, lookup.as_raw_mut()) };
        (result.value != 0)
            .then_some(lookup)
            .ok_or_else(|| Error::SymbolNotFound(self.name().value().to_owned()))
    }

    /// Looks up the definition of this struct using the given op as root.
    pub fn get_definition<'o>(
        &self,
        root: &impl OperationLike<'c, 'o>,
    ) -> Result<SymbolLookupResult<'c>, Error>
    where
        'c: 'o,
    {
        self.get_definition_impl(root.to_raw(), llzk_sys::llzkStructStructTypeGetDefinition)
    }

    /// Looks up the definition of this struct using the given module as root.
    pub fn get_definition_from_module(
        &self,
        root: &Module<'c>,
    ) -> Result<SymbolLookupResult<'c>, Error> {
        self.get_definition_impl(
            root.to_raw(),
            llzk_sys::llzkStructStructTypeGetDefinitionFromModule,
        )
    }
}

impl<'c> FromRaw<MlirType> for StructType<'c> {
    unsafe fn from_raw(t: MlirType) -> Self {
        Self {
            t: unsafe { Type::from_raw(t) },
        }
    }
}

impl<'c> TypeLike<'c> for StructType<'c> {
    fn to_raw(&self) -> MlirType {
        self.t.to_raw()
    }
}

impl<'c> TryFrom<Type<'c>> for StructType<'c> {
    type Error = melior::Error;

    fn try_from(t: Type<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkTypeIsAStructType(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::TypeExpected("llzk struct", t.to_string()))
        }
    }
}

impl<'c> std::fmt::Display for StructType<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.t, formatter)
    }
}

impl<'c> From<StructType<'c>> for Type<'c> {
    fn from(s: StructType<'c>) -> Type<'c> {
        s.t
    }
}

/// Return `true` iff the given [Type] is a [StructType].
#[inline]
pub fn is_struct_type(t: Type) -> bool {
    t.isa::<StructType>()
}
