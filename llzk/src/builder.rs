//! Types and traits for working with operation builders.

use std::{marker::PhantomData, os::raw::c_void, ptr::null_mut};

use llzk_sys::{
    MlirOpBuilder, MlirOpBuilderInsertPoint, MlirOpBuilderListener, mlirOpBuilderCreate,
    mlirOpBuilderCreateWithListener, mlirOpBuilderDestroy, mlirOpBuilderGetContext,
    mlirOpBuilderGetInsertionBlock, mlirOpBuilderGetInsertionPoint, mlirOpBuilderInsert,
    mlirOpBuilderListenerCreate, mlirOpBuilderListenerDestroy, mlirOpBuilderRestoreInsertionPoint,
    mlirOpBuilderSaveInsertionPoint, mlirOpBuilderSetInsertionPoint,
    mlirOpBuilderSetInsertionPointAfter, mlirOpBuilderSetInsertionPointAfterValue,
    mlirOpBuilderSetInsertionPointToEnd, mlirOpBuilderSetInsertionPointToStart,
};
use melior::{
    Context, ContextRef,
    ir::{
        Block, BlockLike, BlockRef, Location, Operation, OperationRef, RegionRef, Value, ValueLike,
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
    fn set_insertion_point_at_start<'a>(&self, block: impl BlockLike<'c, 'a>) {
        unsafe {
            mlirOpBuilderSetInsertionPointToStart(self.to_raw(), block.to_raw());
        }
    }

    /// Sets the insertion point to the end of the given block.
    fn set_insertion_point_at_end<'a>(&self, block: impl BlockLike<'c, 'a>) {
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
    fn set_insertion_point_after_value(&self, value: impl ValueLike<'c>) {
        unsafe {
            mlirOpBuilderSetInsertionPointAfterValue(self.to_raw(), value.to_raw());
        }
    }

    /// Return a saved insertion point.
    fn save_insertion_point<'a>(&self) -> InsertPoint<'c, 'a> {
        unsafe { InsertPoint::from_raw(mlirOpBuilderSaveInsertionPoint(self.to_raw())) }
    }

    /// Restore the insert point to a previously saved point.
    fn restore_insertion_point<'a>(&self, point: InsertPoint<'c, 'a>) {
        unsafe {
            mlirOpBuilderRestoreInsertionPoint(self.to_raw(), point.to_raw());
        }
    }

    /// Returns a reference to the block where the builder will insert operations.
    fn insertion_block<'a>(&self) -> Option<BlockRef<'c, 'a>> {
        unsafe { BlockRef::from_option_raw(mlirOpBuilderGetInsertionBlock(self.to_raw())) }
    }

    /// Returns a reference to the operation where the builder will insert operations after.
    fn insertion_point<'a>(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirOpBuilderGetInsertionPoint(self.to_raw())) }
    }

    /// Inserts the operation produced by the closure and returns a reference to it.
    fn insert<'a, F: FnOnce(&'c Context, Location<'c>) -> Operation<'c>>(
        &'c self,
        loc: Location<'c>,
        f: F,
    ) -> OperationRef<'c, 'a> {
        let ctx = self.context();
        let op = f(unsafe { ctx.to_ref() }, loc);
        unsafe { OperationRef::from_raw(mlirOpBuilderInsert(self.to_raw(), op.into_raw())) }
    }
}

mod sealed {
    use melior::ir::{Block, BlockRef};

    pub trait BlockInsertPointLikeSealed {}

    impl BlockInsertPointLikeSealed for BlockRef<'_, '_> {}
    impl BlockInsertPointLikeSealed for &Block<'_> {}
}

/// Extension trait for types that can be used for defining block insert points,
/// namely [`Block`] and [`BlockRef`].
///
/// This trait is sealed and shouldn't be implemented by clients downstream.
pub trait BlockInsertPointLike<'c, 'a>: Sized + sealed::BlockInsertPointLikeSealed {
    /// Returns an insert point at the beginning of the block.
    fn at_start(self) -> EntryPoint<'c, 'a> {
        EntryPoint::Start(self.to_block_ref())
    }

    /// Returns an insert point at the end of the block.
    fn at_end(self) -> EntryPoint<'c, 'a> {
        EntryPoint::End(self.to_block_ref())
    }

    #[doc(hidden)]
    /// Returns a reference to the block.
    fn to_block_ref(self) -> BlockRef<'c, 'a>;
}

impl<'c, 'a> BlockInsertPointLike<'c, 'a> for &'a Block<'c> {
    fn to_block_ref(self) -> BlockRef<'c, 'a> {
        unsafe { BlockRef::from_raw(self.to_raw()) }
    }
}

impl<'c, 'a> BlockInsertPointLike<'c, 'a> for BlockRef<'c, 'a> {
    fn to_block_ref(self) -> BlockRef<'c, 'a> {
        self
    }
}

/// Possible initial insert points when constructing [`OpBuilder`].
#[derive(Debug, Copy, Clone)]
pub enum EntryPoint<'c, 'a> {
    /// The start of a block.
    Start(BlockRef<'c, 'a>),
    /// The end of a block.
    End(BlockRef<'c, 'a>),
    /// Before the operation.
    Before(OperationRef<'c, 'a>),
    /// After the operation.
    After(OperationRef<'c, 'a>),
    /// After the value.
    AfterValue(Value<'c, 'a>),
}

impl<'c> EntryPoint<'c, '_> {
    /// Configures the given builder depending on the variant.
    fn configure_builder(self, b: &impl OpBuilderLike<'c>) {
        match self {
            EntryPoint::Start(block) => b.set_insertion_point_at_start(block),
            EntryPoint::End(block) => b.set_insertion_point_at_end(block),
            EntryPoint::Before(op) => b.set_insertion_point(op),
            EntryPoint::After(op) => b.set_insertion_point_after(op),
            EntryPoint::AfterValue(value) => b.set_insertion_point_after_value(value),
        }
    }
}

impl<'c, 'a> From<Value<'c, 'a>> for EntryPoint<'c, 'a> {
    fn from(value: Value<'c, 'a>) -> Self {
        Self::AfterValue(value)
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
    pub fn new_with_listener<'a>(
        context: &'c Context,
        point: EntryPoint<'c, 'a>,
        listener: impl OpBuilderListener + 'l,
    ) -> Self {
        let ctx = context.to_raw();
        let listener = ListenerWrap::new(listener);
        let b = Self {
            raw: unsafe { mlirOpBuilderCreateWithListener(ctx, listener.raw) },
            _listener: Some(listener),
            _context: PhantomData,
        };
        point.configure_builder(&b);
        b
    }
}

impl<'c> OpBuilder<'c, '_> {
    /// Creates a new operation builder.
    pub fn new(context: &'c Context, point: EntryPoint<'c, '_>) -> Self {
        let ctx = context.to_raw();
        let b = Self {
            raw: unsafe { mlirOpBuilderCreate(ctx) },
            _listener: None,
            _context: Default::default(),
        };
        point.configure_builder(&b);
        b
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
    pub fn at_block_begin<'a>(ctx: &'c Context, block: impl BlockInsertPointLike<'c, 'a>) -> Self
    where
        'c: 'a,
    {
        Self::new(ctx, block.at_start())
    }

    /// Creates a new operation builder with the given block as its insertion point.
    ///
    /// If the block already has a terminator it sets the insertion point right before the
    /// terminator.
    pub fn at_block_end<'a>(ctx: &'c Context, block: impl BlockInsertPointLike<'c, 'a>) -> Self
    where
        'c: 'a,
    {
        let block = block.to_block_ref();
        Self::new(
            ctx,
            match block.terminator() {
                Some(t) => EntryPoint::Before(t),
                None => EntryPoint::End(block),
            },
        )
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
    block: Option<BlockRef<'ctx, 'blk>>,
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
            block: unsafe { BlockRef::from_option_raw(point.block) },
            point: unsafe { OperationRef::from_option_raw(point.point) },
        }
    }

    /// Returns its raw representation.
    fn to_raw(self) -> MlirOpBuilderInsertPoint {
        MlirOpBuilderInsertPoint {
            block: self
                .block
                .map(|b| b.to_raw())
                .unwrap_or(MlirBlock { ptr: null_mut() }),
            point: self
                .point
                .map(|o| o.to_raw())
                .unwrap_or(MlirOperation { ptr: null_mut() }),
        }
    }

    /// Returns the block where the insert point is located.
    pub fn block(&self) -> Option<BlockRef<'ctx, 'blk>> {
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
    ///
    /// The callback receives a reference to the inserted operation and the point where it was
    /// inserted.
    fn notify_operation_inserted<'ctx, 'blk>(
        &mut self,
        op: OperationRef<'ctx, 'blk>,
        point: InsertPoint<'ctx, 'blk>,
    );

    /// Notifies the listener that a block has been inserted.
    ///
    /// The callback receives a reference to the inserted block, the region where it was inserted
    /// and the point of insertion.
    fn notify_block_inserted<'ctx, 'blk>(
        &mut self,
        block: BlockRef<'ctx, 'blk>,
        region: Option<RegionRef<'ctx, 'blk>>,
        point: Option<BlockRef<'ctx, 'blk>>,
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
    F1: FnMut(OperationRef, InsertPoint),
    F2: FnMut(BlockRef, Option<RegionRef>, Option<BlockRef>),
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
        region: Option<RegionRef<'ctx, 'blk>>,
        point: Option<BlockRef<'ctx, 'blk>>,
    ) {
        (self.f2)(block, region, point)
    }
}

/// Handles a listener's lifetime within an [`OpBuilder`].
#[derive(Debug)]
struct ListenerWrap<'l> {
    raw: MlirOpBuilderListener,
    listener: *mut Wrap<'l>,
}

impl<'l> ListenerWrap<'l> {
    fn new(listener: impl OpBuilderListener + 'l) -> Self {
        let listener: Box<Wrap<'l>> = Box::new(Wrap(Box::new(listener)));
        // Leak the pointer to pass it to the FFI function.
        // The destructor will reconstruct the box and dispose of it properly.
        let listener = Box::into_raw(listener);
        let raw = unsafe {
            mlirOpBuilderListenerCreate(
                Some(notify_operation_inserted_cb),
                Some(notify_block_inserted_cb),
                listener as *mut c_void,
            )
        };

        Self { raw, listener }
    }
}

impl Drop for ListenerWrap<'_> {
    fn drop(&mut self) {
        unsafe { mlirOpBuilderListenerDestroy(self.raw) }
        drop(unsafe { Box::from_raw(self.listener) })
    }
}

/// Wraps a pointer to a [`OpBuilderListener`] implementation.
/// This type is used as the user data pointer and
/// its lifetime is handled by [`ListenerWrap`].
struct Wrap<'l>(Box<dyn OpBuilderListener + 'l>);

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
        unsafe { RegionRef::from_option_raw(region) },
        unsafe { BlockRef::from_option_raw(point) },
    );
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::HashSet, rc::Rc};

    use melior::{
        Context,
        dialect::arith,
        ir::{
            BlockLike, Location, Module, Type, attribute::IntegerAttribute,
            operation::OperationLike,
        },
    };
    use rstest::rstest;

    use crate::test::ctx;

    use super::*;

    #[derive(Debug, Default, PartialEq)]
    struct ListenerState {
        listener_addrs: HashSet<usize>,
    }

    #[derive(Clone, Debug)]
    struct RecordingListener {
        state: Rc<RefCell<ListenerState>>,
    }

    impl RecordingListener {
        fn new(state: Rc<RefCell<ListenerState>>) -> Self {
            Self { state }
        }
    }

    impl OpBuilderListener for RecordingListener {
        fn notify_operation_inserted<'ctx, 'blk>(
            &mut self,
            _: OperationRef<'ctx, 'blk>,
            _: InsertPoint<'ctx, 'blk>,
        ) {
            let self_addr = self as *mut Self as usize;
            self.state.borrow_mut().listener_addrs.insert(self_addr);
        }

        fn notify_block_inserted<'ctx, 'blk>(
            &mut self,
            _: BlockRef<'ctx, 'blk>,
            _: Option<RegionRef<'ctx, 'blk>>,
            _: Option<BlockRef<'ctx, 'blk>>,
        ) {
            let self_addr = self as *mut Self as usize;
            self.state.borrow_mut().listener_addrs.insert(self_addr);
        }
    }

    fn index_constant<'c>(ctx: &'c Context, loc: Location<'c>, value: i64) -> Operation<'c> {
        arith::constant(
            ctx,
            IntegerAttribute::new(Type::index(ctx), value).into(),
            loc,
        )
    }

    /// Moves the builder through the heap and back.
    fn move_builder<'c, 'l>(builder: OpBuilder<'c, 'l>) -> OpBuilder<'c, 'l> {
        *Box::new(builder)
    }

    #[rstest]
    fn at_block_begin_inserts_before_existing_operations(ctx: Context) {
        let location = Location::unknown(&ctx);
        let module = Module::new(location);
        let body = module.body();
        let existing = body.append_operation(index_constant(&ctx, location, 2));
        let builder = OpBuilder::at_block_begin(&ctx, body);

        let inserted = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 1));

        assert_eq!(body.first_operation(), Some(inserted));
        assert_eq!(inserted.next_in_block(), Some(existing));
    }

    #[rstest]
    fn set_insertion_point_at_end_and_save_point_use_expected_block(ctx: Context) {
        let location = Location::unknown(&ctx);
        let module = Module::new(location);
        let body = module.body();
        body.append_operation(index_constant(&ctx, location, 2));
        let builder = OpBuilder::at_block_begin(&ctx, body);

        builder.set_insertion_point_at_end(body);
        assert_eq!(builder.insertion_block(), Some(body));

        let end = builder.save_insertion_point();
        let end_raw = end.to_raw();
        assert_eq!(end_raw.block.ptr, body.to_raw().ptr);
        assert!(end_raw.point.ptr.is_null());
    }

    #[rstest]
    fn set_insertion_point_inserts_before_target_operation(ctx: Context) {
        let location = Location::unknown(&ctx);
        let module = Module::new(location);
        let body = module.body();
        let first = body.append_operation(index_constant(&ctx, location, 1));
        let second = body.append_operation(index_constant(&ctx, location, 2));
        let builder = OpBuilder::at_block_begin(&ctx, body);

        builder.set_insertion_point(second);
        let before_second = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 3));
        assert_eq!(first.next_in_block(), Some(before_second));
        assert_eq!(before_second.next_in_block(), Some(second));
    }

    #[rstest]
    fn insertion_point_after_wrappers_insert_immediately_after_anchor(ctx: Context) {
        let location = Location::unknown(&ctx);
        let module = Module::new(location);
        let body = module.body();
        let first = body.append_operation(index_constant(&ctx, location, 1));
        let second = body.append_operation(index_constant(&ctx, location, 2));
        let builder = OpBuilder::new(&ctx, EntryPoint::After(first));

        let after_first = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 4));
        assert_eq!(first.next_in_block(), Some(after_first));
        assert_eq!(after_first.next_in_block(), Some(second));

        builder.set_insertion_point_after_value(first.result(0).unwrap());
        let after_value = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 5));
        assert_eq!(first.next_in_block(), Some(after_value));
        assert_eq!(after_value.next_in_block(), Some(after_first));
    }

    #[rstest]
    fn restoring_saved_insertion_point_rewinds_future_insertions(ctx: Context) {
        let location = Location::unknown(&ctx);
        let module = Module::new(location);
        let body = module.body();
        let builder = OpBuilder::at_block_begin(&ctx, body);

        let first = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 1));
        let saved = builder.save_insertion_point();

        builder.set_insertion_point_at_start(body);
        let second = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 2));
        builder.restore_insertion_point(saved);
        let third = builder.insert(location, |ctx, loc| index_constant(ctx, loc, 3));
        assert_eq!(second.next_in_block(), Some(first));
        assert_eq!(first.next_in_block(), Some(third));
    }

    fn listener_addr(builder: &OpBuilder) -> usize {
        unsafe {
            let Some(listener_wrap) = builder._listener.as_ref().unwrap().listener.as_ref() else {
                return 0;
            };

            listener_wrap.0.as_ref() as *const dyn OpBuilderListener as *const c_void as usize
        }
    }

    #[rstest]
    fn listener_callback_keeps_same_listener_address_when_builder_moves(ctx: Context) {
        let location = Location::unknown(&ctx);
        let state = Rc::new(RefCell::new(ListenerState {
            listener_addrs: HashSet::new(),
        }));
        let module = Module::new(location);
        let body = module.body();
        let builder = OpBuilder::new_with_listener(
            &ctx,
            EntryPoint::Start(body),
            RecordingListener::new(state.clone()),
        );
        let listener_addr = listener_addr(&builder);

        builder.set_insertion_point_at_start(body);
        builder.insert(location, |ctx, loc| index_constant(ctx, loc, 1));
        let first = body.first_operation().unwrap();
        let builder = move_builder(builder);
        builder.set_insertion_point_after(first);
        builder.insert(location, |ctx, loc| index_constant(ctx, loc, 2));

        let expected = ListenerState {
            listener_addrs: HashSet::from_iter([listener_addr]),
        };
        let state = state.borrow();
        assert_eq!(*state, expected);
    }
}
