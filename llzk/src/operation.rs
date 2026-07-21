//! Functions related to operations.

use crate::error::{DiagnosticError, Error};
use core::ffi::c_void;
use llzk_sys::mlirOperationWalkReverse;
use melior::{
    Context,
    diagnostic::DiagnosticSeverity,
    ir::{
        Block, Operation, ValueLike,
        operation::{OperationLike, OperationMutLike, OperationRefMut, WalkOrder, WalkResult},
    },
};
use mlir_sys::{MlirOperation, MlirWalkResult, mlirOperationWalk};

use crate::builder::OpBuilder;

/// Walk iterator over mutable operation.
pub trait WalkOperationMutLike<'c: 'a, 'a> {
    /// Walk this operation (and all nested operations) in either pre- or
    /// post-order.
    ///
    /// The closure is called once per operation; by returning
    /// `WalkResult::Advance`/`Skip`/`Interrupt` you control the traversal.
    fn walk_mut<F>(&mut self, order: WalkOrder, callback: F)
    where
        F: for<'x, 'y> FnMut(OperationRefMut<'x, 'y>) -> WalkResult;

    /// Walk this operation (and all nested operations) in either pre- or
    /// post-order, with reverse iteration over operations at the same level.
    ///
    /// The closure is called once per operation; by returning
    /// `WalkResult::Advance`/`Skip`/`Interrupt` you control the traversal.
    fn walk_rev_mut<F>(&mut self, order: WalkOrder, callback: F)
    where
        F: for<'x, 'y> FnMut(OperationRefMut<'x, 'y>) -> WalkResult;
}

macro_rules! impl_walk_method {
    ($method_name:ident, $walk_fn:path) => {
        fn $method_name<F>(&mut self, order: WalkOrder, mut callback: F)
        where
            F: for<'x, 'y> FnMut(OperationRefMut<'x, 'y>) -> WalkResult,
        {
            // trampoline from C to Rust
            extern "C" fn tramp<'c: 'a, 'a, F: FnMut(OperationRefMut<'c, 'a>) -> WalkResult>(
                operation: MlirOperation,
                data: *mut c_void,
            ) -> MlirWalkResult {
                unsafe {
                    let callback: &mut F = &mut *(data as *mut F);
                    (callback)(OperationRefMut::from_raw(operation)) as _
                }
            }
            unsafe {
                $walk_fn(
                    self.to_raw(),
                    Some(tramp::<'c, 'a, F>),
                    &mut callback as *mut _ as *mut _,
                    order as _,
                );
            }
        }
    };
}

impl<'c: 'a, 'a, T> WalkOperationMutLike<'c, 'a> for T
where
    T: OperationMutLike<'c, 'a>,
{
    impl_walk_method!(walk_mut, mlirOperationWalk);
    impl_walk_method!(walk_rev_mut, mlirOperationWalkReverse);
}

/// Verifies the operation, returning an error if it failed.
pub fn verify_operation<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Result<(), Error> {
    if op.verify() {
        return Ok(());
    }
    Err(Error::OpVerificationFailed {
        name: op.name().as_string_ref().as_str()?.to_owned(),
        ir: op.to_string(),
        location: op.location().to_string(),
        diags: None,
    })
}

/// Verifies the operation, returning an error with the emitted diagnostics if it failed.
pub fn verify_operation_with_diags<'c: 'a, 'a>(
    op: &impl OperationLike<'c, 'a>,
) -> Result<(), Error> {
    let mut errors: Vec<DiagnosticError> = Vec::with_capacity(1);
    let ctx_ref = op.context();
    let id = unsafe { ctx_ref.to_ref() }.attach_diagnostic_handler(|diag| {
        if matches!(diag.severity(), DiagnosticSeverity::Error) {
            errors.push(diag.into());
        }
        // Return false to propagate the diagnostic to other handlers.
        false
    });

    let result = verify_operation(op).map_err(|mut err| {
        match &mut err {
            Error::OpVerificationFailed { diags, .. } if !errors.is_empty() => {
                diags.get_or_insert_default().extend(errors)
            }
            _ => {}
        };
        err
    });
    unsafe { ctx_ref.to_ref() }.detach_diagnostic_handler(id);
    result
}

/// Replace uses of 'of' value with the 'with' value inside the 'op' operation.
#[inline]
pub fn replace_uses_of_with<'c: 'a, 'a>(
    op: &impl OperationLike<'c, 'a>,
    of: impl ValueLike<'c> + Copy,
    with: impl ValueLike<'c> + Copy,
) {
    unsafe {
        llzk_sys::mlirOperationReplaceUsesOfWith(op.to_raw(), of.to_raw(), with.to_raw());
    }
}

/// Moves the operation right after the reference op.
#[inline]
pub fn move_op_after<'c: 'a, 'a>(
    reference: &impl OperationLike<'c, 'a>,
    op: &impl OperationLike<'c, 'a>,
) {
    unsafe { mlir_sys::mlirOperationMoveAfter(op.to_raw(), reference.to_raw()) }
}

/// Erase the given operation.
#[inline]
pub fn erase_op<'c: 'a, 'a>(op: impl OperationLike<'c, 'a>) {
    unsafe {
        mlir_sys::mlirOperationDestroy(op.to_raw());
    }
}

/// Detach a builder-created operation from its current block and return ownership of it.
///
/// LLZK's builder APIs insert operations immediately. However, some users may need to
/// hold an owned operation to be inserted at a later time.
pub fn detach_owned_operation<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Operation<'c> {
    let raw = op.to_raw();
    // SAFETY: `raw` is a valid operation inserted in a scratch block. Removing it transfers
    // responsibility for destroying it to the owned `Operation` constructed below.
    unsafe {
        let mut op_ref = OperationRefMut::from_raw(raw);
        op_ref.remove_from_parent();
        Operation::from_raw(raw)
    }
}

/// Opaque handle to an operation created for [`build_owned_operation`].
///
/// This type intentionally erases the borrow lifetime carried by [`OperationRef`].
/// `build_owned_operation` creates its scratch [`OpBuilder`] locally, so a callback
/// cannot safely return an `OperationRef` whose borrow is tied to that local builder.
/// Asking the callback to return `OperationRef<'_, '_>` directly also forces Rust to
/// infer a concrete return lifetime for the closure, which can over-constrain captured
/// values and produce spurious `'static` requirements.
///
/// `LifetimeErasedOpRef` keeps the raw operation handle private while letting the callback
/// convert an inserted operation reference into a lifetime-free handoff value. The
/// operation is detached immediately before the scratch block and builder are dropped.
#[derive(Clone, Copy, Debug)]
pub struct LifetimeErasedOpRef {
    raw: MlirOperation,
}

impl LifetimeErasedOpRef {
    /// Create a handle from an operation reference returned by a builder API.
    #[inline]
    pub fn from_operation<'c: 'a, 'a>(op: impl OperationLike<'c, 'a>) -> Self {
        Self { raw: op.to_raw() }
    }
}

impl<'c: 'a, 'a, T> From<T> for LifetimeErasedOpRef
where
    T: OperationLike<'c, 'a>,
{
    #[inline]
    fn from(op: T) -> Self {
        Self::from_operation(op)
    }
}

/// Build an owned operation that is not attached to any block.
///
/// The build closure should insert an operation with the provided builder and return
/// a handle to the operation that should be detached.
pub fn build_owned_operation<'c, E>(
    context: &'c Context,
    build: impl FnOnce(&OpBuilder<'c, '_>) -> Result<LifetimeErasedOpRef, E>,
) -> Result<Operation<'c>, E> {
    let scratch = Block::new(&[]);
    let builder = OpBuilder::at_block_end(context, &scratch);
    let raw = build(&builder)?.raw;
    // SAFETY: `raw` is the operation inserted into the scratch block.
    // Removing it transfers ownership to the returned `Operation`.
    unsafe {
        let mut op_ref = OperationRefMut::from_raw(raw);
        op_ref.remove_from_parent();
        Ok(Operation::from_raw(raw))
    }
}

/// Detach the given operation from its parent block, then erase it.
#[inline]
pub fn detach_and_erase_op<'c: 'a, 'a>(op: impl OperationLike<'c, 'a>) {
    erase_op(detach_owned_operation(&op));
}

/// Return `true` iff the given op is has the given name.
#[inline]
pub fn isa<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>, name: &str) -> bool {
    op.name().as_string_ref().as_str() == Result::Ok(name)
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{OpBuilder, OpBuilderLike as _},
        context::LlzkContext,
        dialect::poly,
        operation::{build_owned_operation, detach_owned_operation},
        test::ctx,
    };
    use melior::{
        Context,
        dialect::arith,
        ir::{
            BlockLike as _, Location, Module, Type, attribute::IntegerAttribute,
            operation::OperationLike as _,
        },
    };
    use rstest::rstest;

    fn index_constant<'c: 'a, 'a>(
        builder: &'a OpBuilder<'c, '_>,
        location: Location<'c>,
        value: i64,
    ) -> melior::ir::OperationRef<'c, 'a> {
        builder.insert(location, |ctx, loc| {
            arith::constant(
                ctx,
                IntegerAttribute::new(Type::index(ctx), value).into(),
                loc,
            )
        })
    }

    #[rstest]
    fn detached_operation_survives_scratch_block_drop(ctx: Context) {
        let location = Location::unknown(&ctx);

        let op = build_owned_operation(&ctx, |builder| {
            Ok::<_, core::convert::Infallible>(index_constant(builder, location, 7).into())
        })
        .unwrap();

        assert!(
            unsafe { mlir_sys::mlirOperationGetBlock(op.to_raw()) }
                .ptr
                .is_null()
        );
        assert_eq!(op.name().as_string_ref().as_str(), Ok("arith.constant"));
        assert!(op.verify());
    }

    #[rstest]
    fn detached_operation_can_be_reinserted_once(ctx: Context) {
        let location = Location::unknown(&ctx);
        let scratch = melior::ir::Block::new(&[]);
        let builder = OpBuilder::at_block_end(&ctx, &scratch);
        let inserted = index_constant(&builder, location, 11);
        let raw = inserted.to_raw();

        let op = detach_owned_operation(&inserted);

        assert!(scratch.first_operation().is_none());
        assert_eq!(op.to_raw().ptr, raw.ptr);
        assert!(
            unsafe { mlir_sys::mlirOperationGetBlock(op.to_raw()) }
                .ptr
                .is_null()
        );

        let module = Module::new(location);
        let reinserted = module.body().append_operation(op);

        assert_eq!(reinserted.to_raw().ptr, raw.ptr);
        assert_eq!(module.body().first_operation(), Some(reinserted));
        assert_eq!(
            unsafe { mlir_sys::mlirOperationGetBlock(reinserted.to_raw()) }.ptr,
            module.body().to_raw().ptr
        );
    }

    #[test]
    fn build_owned_operation_poly_param_lifetime_repro() {
        let context = LlzkContext::new();
        let location = Location::unknown(&context);

        let _ = build_owned_operation(&context, |builder| {
            poly::param(builder, location, "T", None).map(Into::into)
        });
    }
}
