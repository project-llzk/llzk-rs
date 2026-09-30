//! Melior affine types and arithmetic extensions.

pub use melior::ir::{AffineExpr, AffineMap};

/// Arithmetic conveniences not provided by Melior's affine expressions.
pub trait AffineExprExt: Sized {
    /// Negates an affine expression.
    fn neg(self) -> Self;

    /// Subtracts an affine expression.
    fn sub(self, rhs: Self) -> Self;
}

impl AffineExprExt for AffineExpr<'_> {
    fn neg(self) -> Self {
        let context = self.context();
        self * Self::constant(unsafe { context.to_ref() }, -1)
    }

    fn sub(self, rhs: Self) -> Self {
        self + rhs.neg()
    }
}
