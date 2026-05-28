//! Extensions for melior types that represent MLIR types.

use melior::ir::r#type::FunctionType;

mod function_type_ext {
    use std::iter::FusedIterator;

    use melior::ir::{Type, r#type::FunctionType};

    /// Iterator over the inputs of a function type.
    #[derive(Debug)]
    pub struct InputsIter<'ctx> {
        fn_type: FunctionType<'ctx>,
        start: usize,
        end: usize,
    }

    impl<'ctx> InputsIter<'ctx> {
        pub(super) fn new(fn_type: FunctionType<'ctx>) -> Self {
            Self {
                fn_type,
                start: 0,
                end: fn_type.input_count(),
            }
        }
    }

    impl<'ctx> Iterator for InputsIter<'ctx> {
        type Item = Type<'ctx>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let n = self.start;
            self.start += 1;
            Some(self.fn_type.input(n).unwrap())
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            if self.start >= self.end {
                return (0, Some(0));
            }
            let size = self.end - self.start;
            (size, Some(size))
        }
    }

    impl ExactSizeIterator for InputsIter<'_> {}

    impl DoubleEndedIterator for InputsIter<'_> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let n = self.end;
            self.end -= 1;
            Some(self.fn_type.input(n).unwrap())
        }
    }

    impl FusedIterator for InputsIter<'_> {}

    /// Iterator over the results of a function type.
    #[derive(Debug)]
    pub struct ResultsIter<'ctx> {
        fn_type: FunctionType<'ctx>,
        start: usize,
        end: usize,
    }

    impl<'ctx> ResultsIter<'ctx> {
        pub(super) fn new(fn_type: FunctionType<'ctx>) -> Self {
            Self {
                fn_type,
                start: 0,
                end: fn_type.result_count(),
            }
        }
    }

    impl<'ctx> Iterator for ResultsIter<'ctx> {
        type Item = Type<'ctx>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let n = self.start;
            self.start += 1;
            Some(self.fn_type.result(n).unwrap())
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            if self.start >= self.end {
                return (0, Some(0));
            }
            let size = self.end - self.start;
            (size, Some(size))
        }
    }

    impl ExactSizeIterator for ResultsIter<'_> {}

    impl DoubleEndedIterator for ResultsIter<'_> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.start >= self.end {
                return None;
            }
            let n = self.end;
            self.end -= 1;
            Some(self.fn_type.result(n).unwrap())
        }
    }

    impl FusedIterator for ResultsIter<'_> {}

    /// Sealed trait to avoid additional implementations of `FunctionTypeExt`.
    pub trait FunctionTypeExtSealed {}
}

/// Extension methods for [`FunctionType`].
pub trait FunctionTypeExt<'ctx>: function_type_ext::FunctionTypeExtSealed {
    /// Returns an iterator of the input's types.
    fn inputs(&self) -> function_type_ext::InputsIter<'ctx>;

    /// Returns an iterator of the result's types.
    fn results(&self) -> function_type_ext::ResultsIter<'ctx>;
}

impl<'ctx> FunctionTypeExt<'ctx> for FunctionType<'ctx> {
    fn inputs(&self) -> function_type_ext::InputsIter<'ctx> {
        function_type_ext::InputsIter::new(*self)
    }

    fn results(&self) -> function_type_ext::ResultsIter<'ctx> {
        function_type_ext::ResultsIter::new(*self)
    }
}

impl function_type_ext::FunctionTypeExtSealed for FunctionType<'_> {}
