//! Types and traits for working with operation builders.

use std::{marker::PhantomData, os::raw::c_void, pin::Pin, ptr::null_mut};

use llzk_sys::{
    MlirOpBuilder, MlirOpBuilderInsertPoint, MlirOpBuilderListener,
    mlirOpBuilderClearInsertionPoint, mlirOpBuilderCreate, mlirOpBuilderCreateWithListener,
    mlirOpBuilderDestroy, mlirOpBuilderGetContext, mlirOpBuilderGetInsertionBlock,
    mlirOpBuilderGetInsertionPoint, mlirOpBuilderListenerCreate, mlirOpBuilderListenerDestroy,
    mlirOpBuilderRestoreInsertionPoint, mlirOpBuilderSaveInsertionPoint,
    mlirOpBuilderSetInsertionPoint, mlirOpBuilderSetInsertionPointAfter,
    mlirOpBuilderSetInsertionPointAfterValue, mlirOpBuilderSetInsertionPointToEnd,
    mlirOpBuilderSetInsertionPointToStart,
};
use melior::{
    Context, ContextRef,
    ir::{
        BlockLike, BlockRef, Location, Operation, OperationRef, RegionRef, ValueLike,
        operation::OperationLike,
    },
};
use mlir_sys::{MlirBlock, MlirOperation, MlirRegion};

/// Defines the general functionality of a builder.
pub trait OpBuilderLike<'c> {
    /// Returns the raw representation of the builder.
    fn to_raw(&self) -> MlirOpBuilder;

    /// Returns a reference to the context associated with the builder.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOpBuilderGetContext(self.to_raw())) }
    }

    /// Sets the insertion point to the start of the given block.
    fn set_insertion_point_at_start<'a, B: BlockLike<'c, 'a>>(&self, block: B) {
        unsafe {
            mlirOpBuilderSetInsertionPointToStart(self.to_raw(), block.to_raw());
        }
    }

    /// Sets the insertion point to the end of the given block.
    fn set_insertion_point_at_end<'a, B: BlockLike<'c, 'a>>(&self, block: B) {
        unsafe {
            mlirOpBuilderSetInsertionPointToEnd(self.to_raw(), block.to_raw());
        }
    }

    /// Sets the insertion point right before the given operation.
    fn set_insertion_point<'a>(&self, op: impl OperationLike<'c, 'a>)
    where
        'c: 'a,
    {
        unsafe {
            mlirOpBuilderSetInsertionPoint(self.to_raw(), op.to_raw());
        }
    }

    /// Sets the insertion point right after the given operation.
    fn set_insertion_point_after<'a>(&self, op: impl OperationLike<'c, 'a>)
    where
        'c: 'a,
    {
        unsafe {
            mlirOpBuilderSetInsertionPointAfter(self.to_raw(), op.to_raw());
        }
    }

    /// Sets the insertion point right after the given value is defined.
    fn set_insertion_point_after_value<'a>(&self, value: impl ValueLike<'c>) {
        unsafe {
            mlirOpBuilderSetInsertionPointAfterValue(self.to_raw(), value.to_raw());
        }
    }

    /// Return a saved insertion point.
    fn save_insertion_point(&self) -> InsertPoint<'c, '_> {
        unsafe { InsertPoint::from_raw(mlirOpBuilderSaveInsertionPoint(self.to_raw())) }
    }

    /// Restore the insert point to a previously saved point.
    fn restore_insertion_point(&self, point: InsertPoint<'c, '_>) {
        unsafe {
            mlirOpBuilderRestoreInsertionPoint(self.to_raw(), point.to_raw());
        }
    }

    /// Reset the insertion point to no location.
    fn clear_insertion_point(&self) {
        unsafe {
            mlirOpBuilderClearInsertionPoint(self.to_raw());
        }
    }

    /// Returns a reference to the block where the builder will insert operations.
    fn insertion_block<'a>(&self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(mlirOpBuilderGetInsertionBlock(self.to_raw())) }
    }

    /// Returns a reference to the operation where the builder will insert operations after.
    fn insertion_point<'a>(&self) -> OperationRef<'c, 'a> {
        unsafe { OperationRef::from_raw(mlirOpBuilderGetInsertionPoint(self.to_raw())) }
    }

    /// Inserts the operation produced by the closure and returns a reference to it.
    fn insert<'a, F: FnOnce(ContextRef<'c>, Location<'c>) -> Operation<'c>>(
        &'c self,
        loc: Location<'c>,
        f: F,
    ) -> OperationRef<'c, 'a> {
        let op = f(self.context(), loc);
        self.insertion_block()
            .insert_operation_after(self.insertion_point(), op)
    }
}

/// An owned operation builder.
#[derive(Debug)]
pub struct OpBuilder<'c, 'l> {
    raw: MlirOpBuilder,
    _listener: Option<ListenerWrap<'l>>,
    _context: PhantomData<&'c Context>,
}

impl<'c, 'l> OpBuilder<'c, 'l> {
    /// Creates a new operation builder with the given listener.
    pub fn new_with_listener(
        context: &'c Context,
        listener: impl OpBuilderListener + 'l + std::marker::Unpin,
    ) -> Self {
        unsafe {
            let ctx = context.to_raw();
            let listener = ListenerWrap::new(listener);
            Self {
                raw: mlirOpBuilderCreateWithListener(ctx, listener.raw),
                _listener: Some(listener),
                _context: PhantomData,
            }
        }
    }
}

impl<'c> OpBuilder<'c, '_> {
    /// Creates a new operation builder.
    pub fn new(context: &'c Context) -> Self {
        unsafe {
            let ctx = context.to_raw();
            Self {
                raw: mlirOpBuilderCreate(ctx),
                _listener: None,
                _context: Default::default(),
            }
        }
    }

    /// Creates an operation builder from its raw representation.
    ///
    /// # Safety
    ///
    /// The reference must be valid.
    pub fn from_raw(raw: MlirOpBuilder) -> Self {
        Self {
            raw,
            _listener: None,
            _context: Default::default(),
        }
    }

    /// Creates a new operation builder with the given block as its insertion point.
    pub fn at_block_begin<'a, B: BlockLike<'c, 'a>>(ctx: &'c Context, block: B) -> Self {
        let b = Self::new(ctx);
        b.set_insertion_point_at_start(block);
        b
    }
}

impl<'c> OpBuilderLike<'c> for OpBuilder<'c, '_> {
    fn to_raw(&self) -> MlirOpBuilder {
        self.raw
    }
}

impl Drop for OpBuilder<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlirOpBuilderDestroy(self.raw) }
    }
}

/// Reference to an operation builder.
#[derive(Debug)]
pub struct OpBuilderRef<'c, 'a, 'l> {
    raw: MlirOpBuilder,
    _reference: PhantomData<&'a OpBuilder<'c, 'l>>,
}

impl<'c, 'a> OpBuilderRef<'c, 'a, '_> {
    /// Creates an operation builder reference from its raw representation.
    ///
    /// # Safety
    ///
    /// The reference must be valid.
    pub fn from_raw(raw: MlirOpBuilder) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }
}

impl<'c> OpBuilderLike<'c> for OpBuilderRef<'c, '_, '_> {
    fn to_raw(&self) -> MlirOpBuilder {
        self.raw
    }
}

/// Insertion point of a [`OpBuilderLike`].
#[derive(Debug, Copy, Clone)]
pub struct InsertPoint<'ctx, 'blk> {
    block: BlockRef<'ctx, 'blk>,
    point: Option<OperationRef<'ctx, 'blk>>,
}

impl<'ctx, 'blk> InsertPoint<'ctx, 'blk> {
    /// Creates an insertion point from its raw representation.
    ///
    /// # Safety
    ///
    /// The inner block and operations must be valid.
    unsafe fn from_raw(point: MlirOpBuilderInsertPoint) -> Self {
        Self {
            block: unsafe { BlockRef::from_raw(point.block) },
            point: unsafe { OperationRef::from_option_raw(point.point) },
        }
    }

    /// Returns its raw representation.
    fn to_raw(&self) -> MlirOpBuilderInsertPoint {
        MlirOpBuilderInsertPoint {
            block: self.block.to_raw(),
            point: self
                .point
                .map(|o| o.to_raw())
                .unwrap_or(MlirOperation { ptr: null_mut() }),
        }
    }

    /// Returns the block where the insert point is located.
    pub fn block(&self) -> BlockRef<'ctx, 'blk> {
        self.block
    }

    /// Returns the insert point.
    pub fn point(&self) -> Option<OperationRef<'ctx, 'blk>> {
        self.point
    }
}

/// Trait defining [`OpBuilderLike`] listeners.
///
/// For simple use cases you can use [`SimpleOpBuilderListener`].
pub trait OpBuilderListener {
    /// Notifies the listener that an operation has been inserted.
    fn notify_operation_inserted<'ctx, 'blk>(
        &mut self,
        op: OperationRef<'ctx, 'blk>,
        point: InsertPoint<'ctx, 'blk>,
    );

    /// Notifies the listener that a block has been inserted.
    fn notify_block_inserted<'ctx, 'blk>(
        &mut self,
        block: BlockRef<'ctx, 'blk>,
        region: RegionRef<'ctx, 'blk>,
        point: BlockRef<'ctx, 'blk>,
    );
}

/// Simple [`OpBuilderListener`].
#[derive(Debug)]
pub struct SimpleOpBuilderListener<F1, F2> {
    f1: F1,
    f2: F2,
}

impl<F1, F2> SimpleOpBuilderListener<F1, F2> {
    /// Creates a new listener.
    pub fn new(f1: F1, f2: F2) -> Self {
        Self { f1, f2 }
    }
}

impl<F1, F2> OpBuilderListener for SimpleOpBuilderListener<F1, F2>
where
    F1: FnMut(OperationRef, InsertPoint) -> (),
    F2: FnMut(BlockRef, RegionRef, BlockRef) -> (),
{
    fn notify_operation_inserted<'ctx, 'blk>(
        &mut self,
        op: OperationRef<'ctx, 'blk>,
        point: InsertPoint<'ctx, 'blk>,
    ) {
        (self.f1)(op, point)
    }

    fn notify_block_inserted<'ctx, 'blk>(
        &mut self,
        block: BlockRef<'ctx, 'blk>,
        region: RegionRef<'ctx, 'blk>,
        point: BlockRef<'ctx, 'blk>,
    ) {
        (self.f2)(block, region, point)
    }
}

/// Handles a listener's lifetime within an [`OpBuilder`].
struct ListenerWrap<'l> {
    raw: MlirOpBuilderListener,
    _listener: Pin<Box<dyn OpBuilderListener + 'l>>,
}

impl<'l> ListenerWrap<'l> {
    fn new<L: OpBuilderListener + 'l + std::marker::Unpin>(listener: L) -> Self {
        let mut listener = Pin::new(Box::new(listener));
        let raw = unsafe {
            mlirOpBuilderListenerCreate(
                Some(notify_operation_inserted_cb),
                Some(notify_block_inserted_cb),
                listener.as_mut().get_mut() as *mut L as *mut c_void,
            )
        };

        Self {
            raw,
            _listener: listener,
        }
    }
}

impl Drop for ListenerWrap<'_> {
    fn drop(&mut self) {
        unsafe { mlirOpBuilderListenerDestroy(self.raw) }
    }
}

impl std::fmt::Debug for ListenerWrap<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListenerWrap")
            .field("raw", &self.raw)
            .finish()
    }
}

/// Transparent wrapper to help with casting.
#[repr(transparent)]
struct Wrap<'a, 'b>(&'a mut (dyn OpBuilderListener + 'b));

unsafe extern "C" fn notify_operation_inserted_cb(
    op: MlirOperation,
    point: MlirOpBuilderInsertPoint,
    data: *mut c_void,
) {
    let data = unsafe { &mut *(data as *mut Wrap) };
    data.0
        .notify_operation_inserted(unsafe { OperationRef::from_raw(op) }, unsafe {
            InsertPoint::from_raw(point)
        });
}

unsafe extern "C" fn notify_block_inserted_cb(
    block: MlirBlock,
    region: MlirRegion,
    point: MlirBlock,
    data: *mut c_void,
) {
    let data = unsafe { &mut *(data as *mut Wrap) };
    data.0.notify_block_inserted(
        unsafe { BlockRef::from_raw(block) },
        unsafe { RegionRef::from_raw(region) },
        unsafe { BlockRef::from_raw(point) },
    );
}
