//! LLZK passes.

use llzk_macro::passes;
use melior::dialect::DialectRegistry;

passes!(
    "LLZKTransformation",
    [
        mlirCreateLLZKTransformationRedundantOperationEliminationPass,
        mlirCreateLLZKTransformationRedundantReadAndWriteEliminationPass,
        mlirCreateLLZKTransformationUnusedDeclarationEliminationPass,
    ]
);

#[cfg(feature = "pcl-backend")]
passes!("PCLConversion", [mlirCreatePCLConversionPCLLoweringPass,]);

passes!(
    "LLZKArrayTransformation",
    [mlirCreateLLZKArrayTransformationArrayToScalarPass]
);

passes!(
    "LLZKIncludeTransformation",
    [mlirCreateLLZKIncludeTransformationInlineIncludesPass]
);

passes!(
    "LLZKPolymorphicTransformation",
    [mlirCreateLLZKPolymorphicTransformationFlatteningPass]
);

passes!(
    "LLZKValidation",
    [mlirCreateLLZKValidationMemberWriteValidatorPass]
);

// Contexts can be created concurrently, but MLIR's global pass registry is not thread-safe.
static REGISTRATION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Registers all core LLZK passes and pipelines.
pub fn register_core_llzk_passes(registry: &DialectRegistry) {
    let _guard = REGISTRATION_LOCK.lock().unwrap();
    unsafe { llzk_sys::llzkRegisterCorePasses(registry.to_raw()) }
}

/// Registers R1CS backend passes and pipelines.
pub fn register_r1cs_passes(registry: &DialectRegistry) {
    let _guard = REGISTRATION_LOCK.lock().unwrap();
    unsafe { llzk_sys::llzkRegisterR1CSPasses(registry.to_raw()) }
}

/// Registers PCL backend passes and pipelines.
///
/// Does nothing if LLZK was compiled without the PCL backend.
pub fn register_pcl_passes(registry: &DialectRegistry) {
    let _guard = REGISTRATION_LOCK.lock().unwrap();
    unsafe { llzk_sys::llzkRegisterPCLPasses(registry.to_raw()) }
}

/// Registers ZKLean backend conversion passes.
pub fn register_zklean_passes(registry: &DialectRegistry) {
    let _guard = REGISTRATION_LOCK.lock().unwrap();
    unsafe { llzk_sys::llzkRegisterZKLeanPasses(registry.to_raw()) }
}

/// Registers SMT backend conversion passes.
pub fn register_smt_passes(registry: &DialectRegistry) {
    let _guard = REGISTRATION_LOCK.lock().unwrap();
    unsafe { llzk_sys::llzkRegisterSMTPasses(registry.to_raw()) }
}

#[cfg(test)]
mod tests {
    //! Tests to make sure that the expected function were generated.

    use melior::{Context, pass::PassManager};

    #[cfg(feature = "pcl-backend")]
    #[test]
    fn generated_pcl_pass_functions() {
        let ctx = Context::new();
        let pm = PassManager::new(&ctx);
        super::register_pcl_conversion_passes();
        super::register_pcl_lowering_pass();
        pm.add_pass(super::create_pcl_lowering_pass());
    }

    #[test]
    fn generated_pass_functions() {
        let ctx = Context::new();
        // Use a PassManager to manage the lifetime of the created passes to avoid memory leaks.
        let pm = PassManager::new(&ctx);
        super::register_llzk_transformation_passes();
        super::register_redundant_operation_elimination_pass();
        super::register_redundant_read_and_write_elimination_pass();
        super::register_unused_declaration_elimination_pass();
        pm.add_pass(super::create_redundant_operation_elimination_pass());
        pm.add_pass(super::create_redundant_read_and_write_elimination_pass());
        pm.add_pass(super::create_unused_declaration_elimination_pass());

        super::register_llzk_array_transformation_passes();
        super::register_array_to_scalar_pass();
        pm.add_pass(super::create_array_to_scalar_pass());

        super::register_llzk_include_transformation_passes();
        super::register_inline_includes_pass();
        pm.add_pass(super::create_inline_includes_pass());

        super::register_llzk_polymorphic_transformation_passes();
        super::register_flattening_pass();
        pm.add_pass(super::create_flattening_pass());

        super::register_llzk_validation_passes();
        super::register_member_write_validator_pass();
        pm.add_pass(super::create_member_write_validator_pass());
    }
}
