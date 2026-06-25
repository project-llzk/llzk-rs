//! Types and functions for working with affine maps and expressions.

use std::{
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

use melior::{Context, ContextRef, ir::Attribute};
use mlir_sys::{
    MlirAffineExpr, MlirAffineMap, mlirAffineAddExprGet, mlirAffineConstantExprGet,
    mlirAffineDimExprGet, mlirAffineExprGetContext, mlirAffineMapAttrGet, mlirAffineMapGet,
    mlirAffineMulExprGet, mlirAffineSymbolExprGet,
};

/// An affine map.
///
/// This type is different from melior's [`AffineMap`](melior::ir::AffineMap).
/// `From<AffineMap>` is implemented for it to help with compatibility.
/// The reverse operation of `From<melior::ir::AffineMap> for AffineMap` is not
/// implemented because melior's version does not have a way of extracting the
/// raw CAPI type.
#[derive(Debug, Clone, Copy)]
pub struct AffineMap<'ctx> {
    raw: MlirAffineMap,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> AffineMap<'ctx> {
    /// Creates an affine with the given number of dimension and symbols.
    pub fn new(
        context: &'ctx Context,
        dims: usize,
        symbols: usize,
        exprs: &[AffineExpr<'ctx>],
    ) -> Self {
        let mut exprs = exprs
            .iter()
            .map(|expr| unsafe { expr.to_raw() })
            .collect::<Vec<_>>();
        Self {
            raw: unsafe {
                mlirAffineMapGet(
                    context.to_raw(),
                    dims as isize,
                    symbols as isize,
                    exprs.len() as isize,
                    exprs.as_mut_ptr(),
                )
            },
            _context: PhantomData,
        }
    }

    /// Returns the raw representation of the affine map.
    pub unsafe fn to_raw(&self) -> MlirAffineMap {
        self.raw
    }
}

impl<'ctx> From<AffineMap<'ctx>> for Attribute<'ctx> {
    fn from(value: AffineMap<'ctx>) -> Self {
        unsafe { Attribute::from_raw(mlirAffineMapAttrGet(value.to_raw())) }
    }
}

impl<'ctx> From<AffineMap<'ctx>> for melior::ir::AffineMap<'ctx> {
    fn from(value: AffineMap<'ctx>) -> Self {
        unsafe { Self::from_raw(value.to_raw()) }
    }
}

/// An affine expression.
#[derive(Debug, Clone, Copy)]
pub struct AffineExpr<'ctx> {
    raw: MlirAffineExpr,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> AffineExpr<'ctx> {
    /// Creates a constant expression.
    pub fn constant(context: &'ctx Context, constant: i64) -> Self {
        Self {
            raw: unsafe { mlirAffineConstantExprGet(context.to_raw(), constant) },
            _context: PhantomData,
        }
    }

    /// Creates a symbol expression.
    pub fn symbol(context: &'ctx Context, position: usize) -> Self {
        Self {
            raw: unsafe { mlirAffineSymbolExprGet(context.to_raw(), position as isize) },
            _context: PhantomData,
        }
    }

    /// Creates a dimension expression.
    pub fn dimension(context: &'ctx Context, position: usize) -> Self {
        Self {
            raw: unsafe { mlirAffineDimExprGet(context.to_raw(), position as isize) },
            _context: PhantomData,
        }
    }

    /// Returns the raw representation of the affine expression.
    pub unsafe fn to_raw(&self) -> MlirAffineExpr {
        self.raw
    }

    /// Returns a reference to the context.
    pub fn context(&self) -> ContextRef<'ctx> {
        unsafe { ContextRef::from_raw(mlirAffineExprGetContext(self.to_raw())) }
    }
}

impl Add for AffineExpr<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            raw: unsafe { mlirAffineAddExprGet(self.to_raw(), rhs.to_raw()) },
            _context: PhantomData,
        }
    }
}

impl Sub for AffineExpr<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for AffineExpr<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            raw: unsafe { mlirAffineMulExprGet(self.to_raw(), rhs.to_raw()) },
            _context: PhantomData,
        }
    }
}

impl Neg for AffineExpr<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let context = self.context();
        self * Self::constant(unsafe { context.to_ref() }, -1)
    }
}
