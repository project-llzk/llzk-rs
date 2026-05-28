//! Types related to MLIR contexts.

use std::{borrow::Borrow, ops::Deref};

use log::Log;
use melior::{
    Context,
    diagnostic::DiagnosticHandlerId,
    dialect::DialectRegistry,
    ir::{Type, r#type::IntegerType},
};

use crate::{diagnostics::log_diagnostic, prelude::FeltType, register_all_llzk_dialects};

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

    /// Returns an instance of `!felt.type` using the default prime field.
    ///
    /// If you need to create one with a different field that the default use [`FeltType::new`] or
    /// [`FeltType::with_field`] instead.
    pub fn felt_type(&self) -> FeltType {
        self.field()
            .map(|field| FeltType::with_field(self, field))
            .unwrap_or_else(|| FeltType::new(self))
    }

    /// Returns an instance of the type used for representing booleans in LLZK.
    pub fn bool_type(&self) -> Type {
        IntegerType::new(self, 1).into()
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
