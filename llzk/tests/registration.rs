#![allow(unused_crate_dependencies)]
//! Smoke tests for the public core and backend registration APIs.

use llzk::{
    passes::*, register_core_llzk_dialects, register_pcl_dialects, register_r1cs_dialects,
    register_smt_dialects, register_zklean_dialects,
};
use melior::{Context, dialect::DialectRegistry};
use rstest::rstest;

#[rstest]
#[case::core(register_core_llzk_dialects)]
#[case::r1cs(register_r1cs_dialects)]
#[case::pcl(register_pcl_dialects)]
#[case::smt(register_smt_dialects)]
#[case::zklean(register_zklean_dialects)]
fn load_registered_dialects(#[case] register: fn(&DialectRegistry)) {
    let registry = DialectRegistry::new();
    register(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    assert_eq!(
        context.loaded_dialect_count(),
        context.registered_dialect_count(),
    );
}

#[test]
fn register_passes() {
    // Keep global pass registration sequential and isolated from the generated-pass unit tests.
    let registry = DialectRegistry::new();
    register_core_llzk_passes(&registry);
    register_r1cs_passes(&registry);
    register_pcl_passes(&registry);
    register_smt_passes(&registry);
    register_zklean_passes(&registry);
}
