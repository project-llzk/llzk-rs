use log::LevelFilter;
use simplelog::{Config, TestLogger};

#[macro_export]
/// Verifies an operation and compares the emitted module against an expected MLIR file.
macro_rules! assert_test {
    ($op:expr, $module:expr, @file $expected:literal ) => {{
        verify_operation_with_diags(&$op).unwrap();
        log::info!("Op passed verification");
        mlir_testutils::assert_module_eq_to_file!(&$module, $expected);
        log::info!("Module matches expected output");
    }};
}

pub fn setup() {
    let _ = TestLogger::init(LevelFilter::Debug, Config::default());
}
