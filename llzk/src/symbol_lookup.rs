//! Types for handling operation lookup results.

use std::marker::PhantomData;

use llzk_sys::LlzkSymbolLookupResult;
use melior::ir::OperationRef;

use crate::context::LlzkContext;

/// Owns the lookup result and provides a reference to the looked up operation.
#[derive(Debug)]
pub struct SymbolLookupResult<'ctx> {
    raw: LlzkSymbolLookupResult,
    _context: PhantomData<&'ctx LlzkContext>,
}

impl<'ctx> SymbolLookupResult<'ctx> {
    pub(crate) fn new() -> Self {
        Self {
            raw: LlzkSymbolLookupResult {
                ptr: std::ptr::null_mut(),
            },
            _context: PhantomData,
        }
    }

    pub(crate) fn as_raw_mut(&mut self) -> &mut LlzkSymbolLookupResult {
        &mut self.raw
    }

    /// Returns a reference to the operation obtained from the lookup.
    pub fn get_operation<'a>(&'a self) -> Option<OperationRef<'ctx, 'a>> {
        unsafe {
            OperationRef::from_option_raw(llzk_sys::LlzkSymbolLookupResultGetOperation(self.raw))
        }
    }
}

impl Drop for SymbolLookupResult<'_> {
    fn drop(&mut self) {
        unsafe {
            llzk_sys::llzkSymbolLookupResultDestroy(self.raw);
        }
    }
}
