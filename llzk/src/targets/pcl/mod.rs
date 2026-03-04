//! Translation to PCL lisp.

use crate::error::Error;
use melior::StringRef;
use melior::ir::Module;
use mlir_sys::MlirStringRef;
use std::ffi::c_void;

/// Transforms a module in PCL IR into its string representation.
///
/// It is necessary that the module only contains PCL IR, either
/// added directly or by previously converting LLZK into PCL.
pub fn translate_module(module: &Module) -> Result<String, Error> {
    let mut data = (String::new(), Ok::<_, Error>(()));

    unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
        let (writer, result) = unsafe { &mut *(data as *mut (String, Result<(), Error>)) };

        if result.is_err() {
            return;
        }

        *result = (|| {
            writer.push_str(unsafe { StringRef::from_raw(string) }.as_str()?);

            Ok(())
        })();
    }

    let logical_result = unsafe {
        llzk_sys::llzkTranslateModuleToPCL(
            module.as_operation().to_raw(),
            Some(callback),
            &mut data as *mut _ as *mut c_void,
        )
    };
    let (dst, result) = data;
    // Emit errors created by the callback, if any.
    let _: () = result?;
    if logical_result.value == 0 {
        return Err(Error::PclTranslationError);
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::LlzkContext;

    #[test]
    fn test_translation_to_pcl() {
        let _ = simplelog::TestLogger::init(log::LevelFilter::Debug, simplelog::Config::default());

        let ctx = LlzkContext::new();

        let module_ir = include_str!("test_files/translation_to_pcl.mlir");
        let expected_lisp = include_str!("test_files/translation_to_pcl.pcl");

        let mut module = Module::parse(&ctx, module_ir).unwrap();
        let pm = PassManager::new(&ctx);
        pm.add_pass(llzk::passes::create_pcl_lowering_pass());
        pm.run(module).unwrap();
        let output = translate_module(&module).expect("pcl translation");
        similar_asserts::assert_eq!(output, expected_lisp);
    }
}
