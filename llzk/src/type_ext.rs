//! Extensions for melior types that represent MLIR types.

use melior::ir::r#type::FunctionType;

mod function_type_ext {
    use melior::ir::{Type, r#type::FunctionType};

    /// Iterator over the inputs of a function type.
    pub(super) struct InputsIter<'ctx> {
        pub(super) fn_type: FunctionType<'ctx>,
        pub(super) count: usize,
    }

    impl<'ctx> Iterator for InputsIter<'ctx> {
        type Item = Type<'ctx>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.count >= self.fn_type.input_count() {
                return None;
            }
            let count = self.count;
            self.count += 1;
            self.fn_type.input(count).ok()
        }
    }

    /// Iterator over the results of a function type.
    pub(super) struct ResultsIter<'ctx> {
        pub(super) fn_type: FunctionType<'ctx>,
        pub(super) count: usize,
    }

    impl<'ctx> Iterator for ResultsIter<'ctx> {
        type Item = Type<'ctx>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.count >= self.fn_type.result_count() {
                return None;
            }
            let count = self.count;
            self.count += 1;
            self.fn_type.result(count).ok()
        }
    }

    /// Sealed trait to avoid additional implementations of `FunctionTypeExt`.
    pub(super) trait FunctionTypeExtSealed {}
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
        function_type_ext::InputsIter {
            fn_type: *self,
            count: 0,
        }
    }

    fn results(&self) -> function_type_ext::ResultsIter<'ctx> {
        function_type_ext::ResultsIter {
            fn_type: *self,
            count: 0,
        }
    }
}

impl function_type_ext::FunctionTypeExtSealed for FunctionType<'_> {}
