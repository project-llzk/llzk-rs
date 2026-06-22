//! `pod` dialect operations and helper functions.

use super::r#type::PodType;
use crate::{
    builder::{OpBuilder, OpBuilderLike}, map_operands::MapOperandsBuilder,
    prelude::FlatSymbolRefAttribute,
};
use llzk_sys::{
    LlzkRecordValue, llzkPod_NewPodOpBuild, llzkPod_NewPodOpBuildInferredFromInitialValues,
    llzkPod_NewPodOpBuildWithMapOperands, llzkPod_ReadPodOpBuild, llzkPod_WritePodOpBuild,
    mlirOpBuilderCreate,
};
use melior::StringRef;
use melior::ir::{
    Identifier, Location, Operation, Type, TypeLike, Value, ValueLike,
    operation::OperationLike,
};
use std::marker::PhantomData;

/// Wrapper around a `LlzkRecordValue`, used to initialize fields in a `pod.new` operation.
#[derive(Debug)]
pub struct RecordValue<'c, 'a> {
    raw: LlzkRecordValue,
    _context: PhantomData<Value<'c, 'a>>,
}

impl<'c, 'a> RecordValue<'c, 'a> {
    /// Creates a new record value.
    pub fn new(name: StringRef, value: Value<'c, 'a>) -> Self {
        Self {
            raw: LlzkRecordValue {
                name: name.to_raw(),
                value: value.to_raw(),
            },
            _context: PhantomData,
        }
    }

    /// Returns the raw representation of the record value.
    pub fn to_raw(&self) -> LlzkRecordValue {
        self.raw
    }

    /// Creates the record value from a raw pointer.
    pub fn from_raw(raw: LlzkRecordValue) -> Self {
        Self {
            raw,
            _context: PhantomData,
        }
    }
}

/// Creates a 'pod.new' operation from a list of initialization values. If the optional type
/// of the result pod is not given, it will be inferred from the provided initialization values.
pub fn new<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    values: &[RecordValue<'c, 'a>],
    r#type: Option<PodType<'c>>,
) -> Operation<'c> {
    let raw_values: Vec<_> = values.iter().map(RecordValue::to_raw).collect();
    if let Some(r#type) = r#type {
        unsafe {
            Operation::from_raw(llzkPod_NewPodOpBuild(
                builder.to_raw(),
                location.to_raw(),
                r#type.to_raw(),
                isize::try_from(raw_values.len()).expect("raw_values too large"),
                raw_values.as_ptr(),
            ))
        }
    } else {
        unsafe {
            Operation::from_raw(llzkPod_NewPodOpBuildInferredFromInitialValues(
                builder.to_raw(),
                location.to_raw(),
                isize::try_from(raw_values.len()).expect("raw_values too large"),
                raw_values.as_ptr(),
            ))
        }
    }
}

/// Creates a 'pod.new' operation from a list of initialization values and a`MapOperandsBuilder`
/// to instantiate top-level `affine_map` attributes appearing in the pod type.
pub fn new_with_affine_init<'c, 'a>(
    builder: &impl OpBuilderLike<'c>,
    location: Location<'c>,
    values: &[RecordValue<'c, 'a>],
    r#type: PodType<'c>,
    affine_init: MapOperandsBuilder,
) -> Operation<'c> {
    let raw_values: Vec<_> = values.iter().map(RecordValue::to_raw).collect();
    unsafe {
        Operation::from_raw(llzkPod_NewPodOpBuildWithMapOperands(
            builder.to_raw(),
            location.to_raw(),
            r#type.to_raw(),
            isize::try_from(raw_values.len()).expect("raw_values too large"),
            raw_values.as_ptr(),
            affine_init.to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `pod.new`.
#[inline]
pub fn is_pod_new<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "pod.new")
}

/// Creates a 'pod.read' operation.
pub fn read<'c>(
    location: Location<'c>,
    pod_ref: Value<'c, '_>,
    record_name: FlatSymbolRefAttribute<'c>,
    result: Type<'c>,
) -> Operation<'c> {
    let ctx = location.context();
    unsafe {
        let ctx = ctx.to_ref();
        let builder = OpBuilder::from_raw(mlirOpBuilderCreate(ctx.to_raw()));
        Operation::from_raw(llzkPod_ReadPodOpBuild(
            builder.to_raw(),
            location.to_raw(),
            result.to_raw(),
            pod_ref.to_raw(),
            Identifier::new(ctx, record_name.value()).to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `pod.read`.
#[inline]
pub fn is_pod_read<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "pod.read")
}

/// Creates a 'pod.write' operation.
pub fn write<'c>(
    location: Location<'c>,
    pod_ref: Value<'c, '_>,
    record_name: FlatSymbolRefAttribute<'c>,
    rvalue: Value<'c, '_>,
) -> Operation<'c> {
    let ctx = location.context();
    unsafe {
        let ctx = ctx.to_ref();
        let builder = OpBuilder::from_raw(mlirOpBuilderCreate(ctx.to_raw()));
        Operation::from_raw(llzkPod_WritePodOpBuild(
            builder.to_raw(),
            location.to_raw(),
            pod_ref.to_raw(),
            rvalue.to_raw(),
            Identifier::new(ctx, record_name.value()).to_raw(),
        ))
    }
}

/// Return `true` iff the given op is `pod.write`.
#[inline]
pub fn is_pod_write<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> bool {
    crate::operation::isa(op, "pod.write")
}
