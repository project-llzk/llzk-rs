//! `array` dialect operations and helper functions.
use super::ArrayType;
use crate::{builder::OpBuilderLike, map_operands::MapOperandsBuilder, value_ext::ValueRange};
use llzk_sys::{
    llzkArray_ArrayLengthOpBuild, llzkArray_CreateArrayOpBuildWithMapOperands,
    llzkArray_CreateArrayOpBuildWithValues, llzkArray_ExtractArrayOpBuild,
    llzkArray_InsertArrayOpBuild, llzkArray_ReadArrayOpBuild, llzkArray_WriteArrayOpBuild,
};
use melior::ir::{
    Location, OperationRef, Type, TypeLike, Value, ValueLike, attribute::DenseI32ArrayAttribute,
    operation::OperationLike,
};
use mlir_sys::MlirOperation;

/// Possible constructors for creating `array.new` operations.
#[derive(Debug)]
pub enum ArrayCtor<'c, 'a, 'b, 'd> {
    /// Creates an empty array of the given type. Alias for `Values(&[])`.
    Empty,
    /// Creates the array from a list of values. The list length must be either
    /// zero or equal to the length of the flattened/linearized result ArrayType.
    Values(&'a [Value<'c, 'b>]),
    /// Creates an empty array by specifying the values needed to instantiate
    /// AffineMap attributes used as dimension sizes in the result ArrayType.
    MapDimAttr(&'a [ValueRange<'c, 'a, 'b>], DenseI32ArrayAttribute<'c>),
    /// Creates an empty array by specifying the values needed to instantiate
    /// AffineMap attributes used as dimension sizes in the result ArrayType.
    MapDimSlice(&'a [ValueRange<'c, 'a, 'b>], &'d [i32]),
}

impl<'c, 'a, 'b, 'd> ArrayCtor<'c, 'a, 'b, 'd> {
    fn build(
        &self,
        builder: &impl OpBuilderLike<'c>,
        location: Location<'c>,
        r#type: ArrayType<'c>,
    ) -> MlirOperation {
        match self {
            Self::Empty => unsafe {
                llzkArray_CreateArrayOpBuildWithValues(
                    builder.to_raw(),
                    location.to_raw(),
                    r#type.to_raw(),
                    0,
                    std::ptr::null(),
                )
            },

            Self::Values(values) => unsafe {
                let raw_values = values.iter().map(|v| v.to_raw()).collect::<Vec<_>>();
                llzkArray_CreateArrayOpBuildWithValues(
                    builder.to_raw(),
                    location.to_raw(),
                    r#type.to_raw(),
                    isize::try_from(raw_values.len()).expect("value count too large"),
                    raw_values.as_ptr(),
                )
            },

            Self::MapDimAttr(map_operands, num_dims_per_map) => unsafe {
                let mut map_operands_builder = MapOperandsBuilder::new();

                for operands in *map_operands {
                    map_operands_builder.append_operands(*operands);
                }

                map_operands_builder.set_dims_per_map_from_attr(*num_dims_per_map);

                llzkArray_CreateArrayOpBuildWithMapOperands(
                    builder.to_raw(),
                    location.to_raw(),
                    r#type.to_raw(),
                    map_operands_builder.to_raw(),
                )
            },

            Self::MapDimSlice(map_operands, num_dims_per_map) => unsafe {
                assert_eq!(map_operands.len(), num_dims_per_map.len());
                let mut map_operands_builder = MapOperandsBuilder::new();
                for (operands, dim) in std::iter::zip(*map_operands, *num_dims_per_map) {
                    map_operands_builder.append_operands_with_dim_count(*operands, *dim);
                }
                llzkArray_CreateArrayOpBuildWithMapOperands(
                    builder.to_raw(),
                    location.to_raw(),
                    r#type.to_raw(),
                    map_operands_builder.to_raw(),
                )
            },
        }
    }
}

/// Creates an 'array.new' operation.
pub fn new<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    r#type: ArrayType<'c>,
    ctor: ArrayCtor<'c, '_, '_, '_>,
) -> OperationRef<'c, 'a> {
    unsafe { OperationRef::from_raw(ctor.build(builder, location, r#type)) }
}

/// Return `true` iff the given op is `array.new`.
#[inline]
pub fn is_array_new<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.new")
}

fn read_like_op<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result: Type<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirType,
        mlir_sys::MlirValue,
        isize,
        *const mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> OperationRef<'c, 'a> {
    let raw_indices = indices.iter().map(|v| v.to_raw()).collect::<Vec<_>>();
    unsafe {
        OperationRef::from_raw(build(
            builder.to_raw(),
            location.to_raw(),
            result.to_raw(),
            arr_ref.to_raw(),
            isize::try_from(raw_indices.len()).expect("indices too large"),
            raw_indices.as_ptr(),
        ))
    }
}

/// Creates an 'array.read' operation.
pub fn read<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result: Type<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
) -> OperationRef<'c, 'a> {
    read_like_op(
        builder,
        location,
        result,
        arr_ref,
        indices,
        llzkArray_ReadArrayOpBuild,
    )
}

/// Return `true` iff the given op is `array.read`.
#[inline]
pub fn is_array_read<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.read")
}

/// Creates an 'array.extract' operation.
pub fn extract<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    result: Type<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
) -> OperationRef<'c, 'a> {
    read_like_op(
        builder,
        location,
        result,
        arr_ref,
        indices,
        llzkArray_ExtractArrayOpBuild,
    )
}

/// Return `true` iff the given op is `array.extract`.
#[inline]
pub fn is_array_extract<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.extract")
}

fn write_like_op<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    rvalue: Value<'c, '_>,
    build: unsafe extern "C" fn(
        llzk_sys::MlirOpBuilder,
        mlir_sys::MlirLocation,
        mlir_sys::MlirValue,
        isize,
        *const mlir_sys::MlirValue,
        mlir_sys::MlirValue,
    ) -> mlir_sys::MlirOperation,
) -> OperationRef<'c, 'a> {
    let raw_indices = indices.iter().map(|v| v.to_raw()).collect::<Vec<_>>();
    unsafe {
        OperationRef::from_raw(build(
            builder.to_raw(),
            location.to_raw(),
            arr_ref.to_raw(),
            isize::try_from(raw_indices.len()).expect("indices too large"),
            raw_indices.as_ptr(),
            rvalue.to_raw(),
        ))
    }
}

/// Creates an 'array.write' operation.
pub fn write<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    rvalue: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    write_like_op(
        builder,
        location,
        arr_ref,
        indices,
        rvalue,
        llzkArray_WriteArrayOpBuild,
    )
}

/// Return `true` iff the given op is `array.write`.
#[inline]
pub fn is_array_write<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.write")
}

/// Creates an 'array.insert' operation.
pub fn insert<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    arr_ref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    rvalue: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    write_like_op(
        builder,
        location,
        arr_ref,
        indices,
        rvalue,
        llzkArray_InsertArrayOpBuild,
    )
}

/// Return `true` iff the given op is `array.insert`.
#[inline]
pub fn is_array_insert<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.insert")
}

/// Creates an 'array.len' operation.
pub fn len<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    arr_ref: Value<'c, '_>,
    dim: Value<'c, '_>,
) -> OperationRef<'c, 'a> {
    unsafe {
        OperationRef::from_raw(llzkArray_ArrayLengthOpBuild(
            builder.to_raw(),
            location.to_raw(),
            arr_ref.to_raw(),
            dim.to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `array.len`.
#[inline]
pub fn is_array_len<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "array.len")
}
