//! Configuration pertaining the PCL backend.

use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use cmake::Config;

use crate::{
    cargo_commands::CargoCommands,
    config_traits::{bindgen::BindgenConfig, cmake::CMakeConfig},
};

/// Configures the PCL backend.
#[derive(Debug, Clone)]
pub struct PclConfig {
    is_enabled: bool,
}

impl PclConfig {
    pub const fn new(is_enabled: bool) -> Self {
        Self { is_enabled }
    }

    fn pcl_path(&self) -> Result<PathBuf> {
        let path = PathBuf::from(env::var("LLZK_PCL_ROOT")?);
        if !path.is_dir() {
            bail!("PCL root path {} is not a directory", path.display());
        }
        Ok(path)
    }

    fn pcl_prefix_path(&self) -> Result<PathBuf> {
        let path = PathBuf::from(env::var("LLZK_PCL_PREFIX")?);
        if !path.is_dir() {
            bail!("PCL prefix path {} is not a directory", path.display());
        }
        Ok(path)
    }

    fn pcl_cmake_path(&self) -> Result<PathBuf> {
        let prefix = self.pcl_prefix_path()?;
        let lib_path = self.lib_path()?;
        // Priority order
        // 1. $prefix/lib/cmake/PCL
        // 2. $prefix/lib/cmake
        // 3. $prefix/cmake
        // 4. $prefix
        [
            lib_path.join("cmake").join("PCL"),
            lib_path.join("cmake"),
            prefix.join("cmake"),
            prefix,
        ]
        .into_iter()
        .find(|p| p.is_dir())
        .ok_or_else(|| anyhow::anyhow!("Failed to locate cmake directory"))
    }

    /// Returns CMake flags that configure the pcl backend, if enabled.
    fn pcl_cmake_flags(&self) -> Result<Vec<(&'static str, String)>> {
        if !self.is_enabled {
            return Ok(vec![]);
        }
        Ok(vec![
            ("LLZK_WITH_PCL", "ON".to_owned()),
            ("PCL_DIR", self.pcl_cmake_path()?.display().to_string()),
        ])
    }

    fn lib_path(&self) -> Result<PathBuf> {
        let root = self.pcl_prefix_path()?;
        let candidates = ["lib", "lib64", "lib32"];
        candidates
            .iter()
            .find_map(|dir| {
                let full_path = root.join(dir);
                full_path.is_dir().then_some(full_path)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "could not find library path in {}. Possible directories: {}",
                    root.display(),
                    candidates.join(", ")
                )
            })
    }

    fn expanded_lib_path(&self) -> Result<Vec<PathBuf>> {
        // PCL could leave its libraries in the inner directories instead of everything on the
        // root. This function looks in either place.
        let lib_path = self.lib_path()?;
        Ok([
            lib_path.join("Dialect"),
            lib_path.join("Transforms"),
            lib_path,
        ]
        .into_iter()
        .filter(|p| p.is_dir())
        .collect())
    }

    /// Emits cargo commands required for linking PCL against a cargo project.
    ///
    /// Accepts any implementation of [`Write`] for flexibility while testing.
    /// Within a build script simply pass [`std::io::stdout`].
    ///
    /// The `whole_archive_config` adds `+whole-archive` or `-whole-archive` to the link commands
    /// if it is `Some(true)` or `Some(false)` respectively.
    pub fn emit_cargo_commands<W: Write>(
        &self,
        out: W,
        whole_archive_config: Option<bool>,
    ) -> Result<()> {
        let mut cargo = CargoCommands::new(out);
        cargo.rerun_if_changed(self.pcl_path()?.join("include"))?;
        cargo.rerun_if_changed(self.pcl_path()?.join("lib"))?;

        for lib_path in self.expanded_lib_path()? {
            cargo.rustc_link_search(lib_path, Some("native"))?;
        }
        // Adding the whole archive modifier is optional since only seems to be required for some GNU-like linkers.
        let modifiers = whole_archive_config.map(|enable| ("whole-archive", enable));
        for lib in self.libraries()? {
            eprintln!("PCL library: {lib}");
            cargo.rustc_link_lib_static(&lib, modifiers)?;
        }

        Ok(())
    }

    /// Returns the libraries built by CMake.
    fn libraries(&self) -> Result<Vec<String>> {
        let mut all = vec![];
        self.expanded_lib_path()?
            .into_iter()
            .map(|path| self.libraries_at(&path))
            .try_for_each(|lib| -> Result<()> {
                // Doing it this way we propagate the errors.
                all.extend(lib?);
                Ok(())
            })?;
        if all.is_empty() {
            bail!("Did not find any library for PCL!");
        }
        Ok(all)
    }

    fn libraries_at(&self, lib_path: impl AsRef<Path>) -> Result<Vec<String>> {
        // All libraries are installed in the lib path.
        let entries = lib_path
            .as_ref()
            .read_dir()
            .with_context(|| format!("Failed to read directory {}", lib_path.as_ref().display()))?;
        entries
            .filter_map(|entry| {
                // For each entry try to get its file name, which is given as a OsString
                // and conversion can fail.
                entry
                    .context("Failed to read entry in directory")
                    .and_then(|entry| {
                        entry.file_name().into_string().map_err(|orig| {
                            anyhow::anyhow!("Failed to convert {orig:?} into a String")
                        })
                    })
                    // If conversion was succesful try to extract `XXX` from `libXXX.a`.
                    // Yield None if doesn't match.
                    .map(|name| {
                        name.strip_prefix("lib")
                            .and_then(|s| s.strip_suffix(".a"))
                            .map(ToOwned::to_owned)
                    })
                    // Convert from Result<Option> to Option<Result> to filter out file names
                    // that are not libraries.
                    .transpose()
            })
            .collect()
    }
}

impl CMakeConfig for PclConfig {
    fn apply(&self, cmake: &mut Config) -> Result<()> {
        if !self.is_enabled {
            return Ok(());
        }
        for (k, v) in self.pcl_cmake_flags()? {
            cmake.define(k, v);
        }
        Ok(())
    }
}

impl BindgenConfig for PclConfig {
    fn apply(&self, mut bindgen: bindgen::Builder) -> Result<bindgen::Builder> {
        if self.is_enabled {
            // Add an include to the PCL autogenerated CAPI functions.
            // The LLZK configuration will add the corresponding include paths s.t.
            // the header file included here is found.
            bindgen = bindgen.header_contents(
                "PCL_CAPI.h",
                r#"
#include "pcl-conv/Transforms/TransformationPasses.capi.h.inc"
"#,
            )
        }

        Ok(bindgen)
    }
}
