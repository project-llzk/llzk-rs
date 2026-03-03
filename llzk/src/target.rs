//! Interface to LLZK targets

use std::{ffi::c_void, fmt::Write};

use melior::{StringRef, ir::Module};

use mlir_sys::MlirStringRef;

use crate::error::Error;

/// Translate module to PCL lisp format using the given callback to write the
/// translation. The user_data is passed as the second argument to the cb, with
/// the first argument being from text of the translation.
// #[cfg(feature = "pcl-backend")]
pub unsafe fn translate_module_to_pcl_with_cb<'ctx>(
    module: &Module<'ctx>,
    cb: unsafe extern "C" fn(MlirStringRef, *mut c_void),
    user_data: *mut c_void,
) -> Result<(), Error> {
    use llzk_sys::llzkTranslateModuleToPCL;
    let res = unsafe {
        llzkTranslateModuleToPCL(
            module.as_operation().to_ref().clone().into_raw(),
            Some(cb),
            user_data,
        )
    };
    (res.value != 0)
        .then_some(())
        .ok_or(Error::GeneralError("could not translate module to PCL"))
}

unsafe extern "C" fn writer_callback<W: Write>(s: MlirStringRef, user_data: *mut c_void) {
    let writer: &mut W = unsafe { &mut *(user_data as *mut W) };
    let text = unsafe { StringRef::from_raw(s) }.as_str().unwrap();
    write!(writer, "{text}").unwrap();
}

/// Translate module to PCL lisp format, writing the translation to `writer`.
// #[cfg(feature = "pcl-backend")]
pub fn translate_module_to_pcl<'ctx, W: Write>(
    module: &Module<'ctx>,
    writer: &mut W,
) -> Result<(), Error> {
    let user_data = (writer as *mut W).cast::<c_void>();
    unsafe { translate_module_to_pcl_with_cb(module, writer_callback::<W>, user_data) }
}
