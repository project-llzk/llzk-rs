//! Utilities related to symbol tables.

use std::mem;

use melior::ir::{
    Operation,
    operation::{OperationLike, OperationRef},
};

/// Insert a new symbol operation into the symbol table owned by `sym_table_op`.
///
/// The inserted symbol is renamed automatically if necessary to avoid collisions. Ownership of
/// `new_symbol_op` is transferred to the symbol table.
pub fn insert<'c: 'a, 'a>(
    sym_table_op: &impl OperationLike<'c, 'a>,
    new_symbol_op: Operation<'c>,
) -> OperationRef<'c, 'a> {
    let raw = new_symbol_op.to_raw();

    unsafe {
        llzk_sys::llzkSymbolTableInsert(sym_table_op.to_raw(), raw);
    }

    // The symbol table now owns the operation.
    mem::forget(new_symbol_op);

    unsafe { OperationRef::from_raw(raw) }
}
