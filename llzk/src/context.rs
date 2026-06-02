//! Types related to MLIR contexts.

use std::{borrow::Borrow, ops::Deref};

use log::Log;
use melior::{
    Context, StringRef,
    diagnostic::DiagnosticHandlerId,
    dialect::{DialectRegistry, arith},
    ir::{Type, r#type::IntegerType},
};

use crate::{
    diagnostics::log_diagnostic,
    prelude::{
        BoolAttribute, FeltConstAttribute, FeltType, FlatSymbolRefAttribute, IntegerAttribute,
        Location, Operation, PodRecordAttribute, PodType, TVarType,
    },
    register_all_llzk_dialects,
};

/// A batteries-included MLIR context that automatically loads all the LLZK dialects.
pub struct LlzkContext {
    ctx: Context,
    diagnostics_handler: Option<DiagnosticHandlerId>,
    field: Option<String>,
    _registry: DialectRegistry,
}

impl LlzkContext {
    /// Creates a new [`LlzkContext`] with all LLZK dialects loaded and the diagnostics engine
    /// configured to emit diagnostics to the global [`Log`].
    ///
    /// To create a context that does not set logging see [`LlzkContext::new_no_log`].
    pub fn new() -> Self {
        let mut llzk = Self::new_no_log();
        llzk.log_diagnostics();
        llzk
    }

    /// Creates a new [`LlzkContext`] with all LLZK dialects loaded.
    ///
    /// To create a context that enables logging by default see [`LlzkContext::new`].
    pub fn new_no_log() -> Self {
        let ctx = Context::new();
        let registry = DialectRegistry::new();

        register_all_llzk_dialects(&registry);
        ctx.append_dialect_registry(&registry);
        ctx.load_all_available_dialects();
        Self {
            ctx,
            diagnostics_handler: None,
            field: None,
            _registry: registry,
        }
    }

    /// Configures MLIR to write diagnostics to the global [`Log`].
    pub fn log_diagnostics(&mut self) {
        self.log_diagnostics_to_logger(log::logger());
    }

    /// Configures MLIR to write diagnostics to the given [`Log`].
    pub fn log_diagnostics_to_logger(&mut self, logger: &dyn Log) {
        self.stop_logging_diagnostics();
        self.diagnostics_handler = Some(
            self.ctx
                .attach_diagnostic_handler(|diag| log_diagnostic(diag, logger)),
        );
    }

    /// Stops logging diagnostics to a [`Log`].
    pub fn stop_logging_diagnostics(&mut self) {
        if let Some(id) = self.diagnostics_handler.take() {
            self.ctx.detach_diagnostic_handler(id);
        }
    }

    /// Returns the name of the default prime field.
    pub fn field(&self) -> Option<&str> {
        self.field.as_deref()
    }

    /// Sets the default prime field.
    pub fn set_field(&mut self, field: &str) {
        self.field = Some(field.to_owned())
    }

    /// Returns the unknown location.
    #[inline]
    pub fn unknown_location(&self) -> Location<'_> {
        Location::unknown(self)
    }

    /// Returns an instance of the index type.
    #[inline]
    pub fn index_type(&self) -> Type<'_> {
        Type::index(self)
    }

    /// Returns an instance of the type used for representing booleans in LLZK.
    #[inline]
    pub fn bool_type(&self) -> Type<'_> {
        IntegerType::new(self, 1).into()
    }

    /// Returns an instance of the type used to represent field elements in LLZK, using the default
    /// prime field.
    ///
    /// To create one with a field other than the default, use [`FeltType::new`] or
    /// [`FeltType::with_field`] instead.
    pub fn felt_type(&self) -> FeltType<'_> {
        self.field()
            .map(|field| FeltType::with_field(self, field))
            .unwrap_or_else(|| FeltType::new(self))
    }

    /// Returns an instance of the polymorphic type variable type for the given name.
    #[inline]
    pub fn tvar_type(&self, name: &str) -> TVarType<'_> {
        TVarType::new(self, StringRef::new(name))
    }

    /// Returns an instance of a pod type with the given records.
    #[inline]
    pub fn pod_type<'c>(&'c self, records: &[(&str, Type<'c>)]) -> PodType<'c> {
        let records = records
            .iter()
            .map(|(name, r#type)| PodRecordAttribute::new(name, *r#type))
            .collect::<Vec<_>>();
        PodType::new(self, &records)
    }

    /// Returns an [`IntegerAttribute`] with index type and the given value.
    #[inline]
    pub fn index_attr<T>(&self, integer: T) -> IntegerAttribute<'_>
    where
        T: Into<i64>,
    {
        IntegerAttribute::new(self.index_type(), integer.into())
    }

    /// Returns a [`BoolAttribute`] with the given value.
    #[inline]
    pub fn bool_attr(&self, val: bool) -> BoolAttribute<'_> {
        BoolAttribute::new(self, val)
    }

    /// Returns a [`FeltConstAttribute`] with the given value and bitwidth, using the default prime
    /// field. If the bitwidth is not specified, it defaults to 64 bits.
    ///
    /// To create one with a field other than the default, use [`FeltConstAttribute::new`] or
    /// [`FeltConstAttribute::new_with_bitlen`] instead.
    ///
    /// # Panics
    ///
    /// If `val` is greater than `i64::MAX`.
    #[inline]
    pub fn felt_attr<T>(&self, bitlen: Option<u32>, val: T) -> FeltConstAttribute<'_>
    where
        T: Into<u64>,
    {
        match bitlen {
            Some(b) => FeltConstAttribute::new_with_bitlen(self, b, val.into(), self.field()),
            None => FeltConstAttribute::new(self, val.into(), self.field()),
        }
    }

    /// Returns a [`FeltConstAttribute`] with the given bitwidth, from a base 10 string
    /// representation using the default prime field.
    ///
    /// To create one with a field other than the default, use [`FeltConstAttribute::parse`]
    /// instead.
    #[inline]
    pub fn felt_attr_from_str<T>(&self, bitlen: u32, val: T) -> FeltConstAttribute<'_>
    where
        T: AsRef<str>,
    {
        FeltConstAttribute::parse(self, bitlen, val.as_ref(), self.field())
    }

    /// Returns a [`FeltConstAttribute`] with the given bitwidth, from a slice of bigint parts in
    /// LSB order using the default prime field.
    ///
    /// To create one with a field other than the default, use [`FeltConstAttribute::from_parts`]
    /// instead.
    ///
    /// # Notes
    ///
    /// If the number represented by the parts is unsigned, set the bit length to at least one more
    /// than the minimum number of bits required to represent the value. Otherwise the number will
    /// be interpreted as signed and may cause unexpected behaviors.
    #[inline]
    pub fn felt_attr_from_parts(&self, bitlen: u32, val_parts: &[u64]) -> FeltConstAttribute<'_> {
        FeltConstAttribute::from_parts(self, bitlen, val_parts, self.field())
    }

    /// Returns a [`FeltConstAttribute`] created from a [`num_bigint::BigUint`] using the default
    /// prime field.
    ///
    /// To create one with a field other than the default, use [`FeltConstAttribute::from_biguint`]
    /// instead.
    ///
    /// # Panics
    ///
    /// If the number of bits required to represent the BigUint exceeds `u32::MAX - 1`.
    #[inline]
    #[cfg(feature = "bigint")]
    pub fn felt_attr_from_biguint(&self, value: &num_bigint::BigUint) -> FeltConstAttribute<'_> {
        FeltConstAttribute::from_biguint(self, value, self.field())
    }

    /// Returns a [`FlatSymbolRefAttribute`] created from the given string.
    #[inline]
    pub fn flat_sym_attr(&self, sym: impl AsRef<str>) -> FlatSymbolRefAttribute<'_> {
        FlatSymbolRefAttribute::new(self, sym.as_ref())
    }

    /// Returns a new `arith.constant` operation that produces a boolean constant with the given value.
    #[inline]
    pub fn new_bool_const_op<'c>(&'c self, val: bool, location: Location<'c>) -> Operation<'c> {
        arith::constant(self, self.bool_attr(val).into(), location)
    }

    /// Returns a new `arith.constant` operation that produces an index constant with the given value.
    #[inline]
    pub fn new_index_const_op<'c, T>(&'c self, val: T, location: Location<'c>) -> Operation<'c>
    where
        T: Into<i64>,
    {
        arith::constant(self, self.index_attr(val).into(), location)
    }
}

impl Default for LlzkContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for LlzkContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl Borrow<Context> for LlzkContext {
    fn borrow(&self) -> &Context {
        &self.ctx
    }
}

impl AsRef<Context> for LlzkContext {
    fn as_ref(&self) -> &Context {
        &self.ctx
    }
}

impl Drop for LlzkContext {
    fn drop(&mut self) {
        self.stop_logging_diagnostics();
    }
}

impl std::fmt::Debug for LlzkContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlzkContext")
            .field("registered dialects", &self.registered_dialect_count())
            .field("loaded dialects", &self.loaded_dialect_count())
            .field("ctx", &self.ctx)
            .field("registry", &self._registry)
            .field("diagnostics_handler", &self.diagnostics_handler)
            .field("field", &self.field)
            .finish()
    }
}
