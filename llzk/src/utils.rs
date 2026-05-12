//! General utilities

use melior::{
    StringRef,
    ir::{
        Block, BlockLike, BlockRef, Operation, OperationRef, Region, RegionLike, RegionRef,
        operation::OperationLike,
    },
};
use mlir_sys::MlirStringRef;
use std::{
    ffi::c_void,
    fmt::{self, Formatter},
};

/// Creates an instance from its low-level unsafe representation.
pub trait FromRaw<RawT> {
    /// Constructs Self from RawT via some unsafe function.
    /// # Safety
    /// The raw value must be a valid reference to some MLIR object.
    unsafe fn from_raw(raw: RawT) -> Self;
}

#[allow(dead_code)]
pub(crate) unsafe extern "C" fn print_callback(string: MlirStringRef, data: *mut c_void) {
    unsafe {
        let (formatter, result) = &mut *(data as *mut (&mut Formatter, fmt::Result));

        if result.is_err() {
            return;
        }

        *result = (|| {
            write!(
                formatter,
                "{}",
                StringRef::from_raw(string)
                    .as_str()
                    .map_err(|_| fmt::Error)?
            )
        })();
    }
}

/// Creates an [`Identifier`].
///
/// [`Identifier`]: [`melior::ir::Identifier`].
#[macro_export]
macro_rules! ident {
    ($ctx:expr, $name:expr) => {{
        let ctx = $ctx;
        melior::ir::Identifier::new(unsafe { ctx.to_ref() }, $name)
    }};
}

/// Print a single operation using "assume verified" flag to avoid verification errors on
/// in-progress IR.
pub fn print_operation<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) {
    // Melior does not currently have a wrapper for `mlirOpPrintingFlagsAssumeVerified()`
    unsafe extern "C" fn print_chunk(s: mlir_sys::MlirStringRef, _user_data: *mut c_void) {
        unsafe {
            if let Ok(string) = StringRef::from_raw(s).as_str() {
                print!("{}", string);
            }
        }
    }
    unsafe {
        let flags = mlir_sys::mlirOpPrintingFlagsCreate();
        mlir_sys::mlirOpPrintingFlagsAssumeVerified(flags);
        mlir_sys::mlirOperationPrintWithFlags(
            op.to_raw(),
            flags,
            Some(print_chunk),
            std::ptr::null_mut(),
        );
        mlir_sys::mlirOpPrintingFlagsDestroy(flags);
    }
    println!();
}

/// Print all operations in a block using [`print_operation`].
pub fn print_block<'c: 'a, 'a>(block: &impl BlockLike<'c, 'a>) {
    let mut op = block.first_operation();
    while let Some(o) = op {
        print_operation(&o);
        op = o.next_in_block();
    }
}

/// Print all blocks (and their operations) in a region using [`print_block`].
pub fn print_region<'c: 'a, 'a>(region: &impl RegionLike<'c, 'a>) {
    let mut block = region.first_block();
    while let Some(b) = block {
        print_block(&b);
        block = b.next_in_region();
    }
}

/// Trait for converting melior types to their reference counterparts.
///
/// This trait provides a safe interface for types that have a `to_raw()` and `from_raw()` pattern,
/// enabling conversion from owned types to reference types (e.g., `Block` to `BlockRef`).
pub trait IntoRef<RefType> {
    /// Convert this type into its reference counterpart.
    fn into_ref(self) -> RefType;
}

/// Macro to implement `IntoRef` for melior types with the `to_raw()` + `from_raw()` pattern.
macro_rules! impl_into_ref {
    ($owned:ty, $ref:ty) => {
        impl<'c, 'a> IntoRef<$ref> for $owned {
            #[inline]
            fn into_ref(self) -> $ref {
                unsafe { <$ref>::from_raw(self.to_raw()) }
            }
        }
    };
}

impl_into_ref!(Block<'c>, BlockRef<'c, 'a>);
impl_into_ref!(Region<'c>, RegionRef<'c, 'a>);
impl_into_ref!(Operation<'c>, OperationRef<'c, 'a>);

/// Replicates MLIR `isa` functionality for Rust types using `TryFrom`.
pub trait IsA: Sized {
    /// Like MLIR `isa`, check if `self` can be converted to type `Out`.
    #[inline]
    fn isa<Out: TryFrom<Self>>(self) -> bool {
        Out::try_from(self).is_ok()
    }
}
impl<T> IsA for T {}
