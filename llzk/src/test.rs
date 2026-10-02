use melior::{Context, dialect::DialectRegistry, utility};
use rstest::fixture;

use crate::register_core_llzk_dialects;

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    utility::register_all_dialects(&registry);
    register_core_llzk_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

#[fixture]
pub fn ctx() -> Context {
    let context = Context::new();

    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{}", diagnostic);
        true
    });

    load_all_dialects(&context);
    utility::register_all_llvm_translations(&context);

    context
}
