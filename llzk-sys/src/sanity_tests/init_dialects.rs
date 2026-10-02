use rstest::rstest;

use crate::llzkRegisterCoreDialects;

use super::{TestRegistry, registry};

#[rstest]
fn test_llzk_register_core_dialects(registry: TestRegistry) {
    unsafe { llzkRegisterCoreDialects(registry.registry) }
}
