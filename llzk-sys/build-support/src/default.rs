//! Implementation of the fundamenal configuration.

use std::{
    env,
    io::{Write, stdout},
    path::PathBuf,
};

use anyhow::{Result, bail};
use bindgen::Builder;
use cc::Build;
use cmake::Config;

use crate::{
    cargo_commands::{CargoCommands, whole_archive_config},
    config_traits::{bindgen::BindgenConfig, cc::CCConfig, cmake::CMakeConfig},
    llzk::LIBDIR,
    mlir::MlirConfig,
    pcl::PclConfig,
};

/// Fundamental configuration for the different build tasks.
#[derive(Debug, Clone)]
pub struct DefaultConfig<'a> {
    pcl: PclConfig,
    mlir: MlirConfig<'a>,
}

impl<'a> DefaultConfig<'a> {
    /// Creates a new configuration.
    pub const fn new(
        pcl_enabled: bool,
        passes: Vec<&'a str>,
        mlir_functions: &'a [&'a str],
        mlir_types: &'a [&'a str],
    ) -> Self {
        Self {
            pcl: PclConfig::new(pcl_enabled),
            mlir: MlirConfig::new(passes, mlir_functions, mlir_types),
        }
    }

    /// Name of the wrapper header file that includes all the exported headers.
    pub fn wrapper(&self) -> &'static str {
        "wrapper.h"
    }

    /// Returns the Clang directories for used by CMake to locate them.
    fn clang_cmake_flags(&self) -> Result<Vec<(&'static str, PathBuf)>> {
        Ok(vec![
            ("Clang_DIR", self.mlir.mlir_cmake_path()?),
            ("Clang_ROOT", self.mlir.mlir_path()?),
        ])
    }

    /// Emits cargo commands.
    pub fn emit_cargo_commands(&self) -> Result<()> {
        self.pcl
            .emit_cargo_commands(stdout(), whole_archive_config())
    }
}

impl CMakeConfig for DefaultConfig<'_> {
    fn apply(&self, cmake: &mut Config) -> Result<()> {
        cmake
            .define("LLZK_BUILD_DEVTOOLS", "OFF")
            .define("LLZK_ENABLE_BINDINGS_PYTHON", "OFF")
            // Force the install lib directory for consistency between Linux distros
            // See: https://stackoverflow.com/questions/76517286/how-does-cmake-decide-to-make-a-lib-or-lib64-directory-for-installations
            .define("CMAKE_INSTALL_LIBDIR", LIBDIR)
            .define("BUILD_TESTING", "OFF");

        for (k, v) in self.clang_cmake_flags()? {
            cmake.define(k, &*v);
        }
        CMakeConfig::apply(&self.pcl, cmake)?;
        CMakeConfig::apply(&self.mlir, cmake)
    }
}

impl BindgenConfig for DefaultConfig<'_> {
    fn apply(&self, bindgen: Builder) -> Result<Builder> {
        let bindgen = bindgen
            .allowlist_item("[Ll]lzk.*")
            .allowlist_var("LLZK_.*")
            .allowlist_recursively(false)
            // Needs to be defined as an opaque blob because bindgen won't derive Copy otherwise
            // because it cannot figure out that the inner MlirAttribute is Copy.
            .opaque_type("LlzkAffineMapOperandsBuilder")
            .impl_debug(true)
            .header(self.wrapper())
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
        let bindgen = BindgenConfig::apply(&self.mlir, bindgen)?;
        BindgenConfig::apply(&self.pcl, bindgen)
    }
}

impl CCConfig for DefaultConfig<'_> {
    fn apply(&self, cc: &mut Build) -> Result<()> {
        CCConfig::apply(&self.mlir, cc)
    }
}
