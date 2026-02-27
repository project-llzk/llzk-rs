//! Types and functions related to LLZK CMake builds.

use crate::{
    cargo_commands::CargoCommands,
    config_traits::{bindgen::BindgenConfig, cc::CCConfig},
};
use anyhow::{Context as _, Result};
use bindgen::Builder;
use cc::Build;
use std::{
    borrow::Cow,
    io::Write,
    path::{Path, PathBuf},
};

/// Common install location for libraries.
pub const LIBDIR: &str = "lib";

/// Represents a CMake build of the LLZK library.
#[derive(Debug)]
pub struct LlzkBuild<'s> {
    src_path: &'s Path,
    dst_path: PathBuf,
}

impl<'s> LlzkBuild<'s> {
    /// Creates a new build.
    pub(crate) fn new(src_path: &'s Path, dst_path: PathBuf) -> Self {
        Self { src_path, dst_path }
    }

    /// Returns the source path.
    pub fn src_path(&self) -> &'s Path {
        self.src_path
    }

    /// Returns the destination path of the build.
    pub fn dst_path(&self) -> &Path {
        &self.dst_path
    }

    /// Returns the library installation path of the build.
    pub fn lib_path(&self) -> PathBuf {
        self.dst_path.join(LIBDIR)
    }

    /// Returns the path where CMake stored intermediate build files.
    pub fn build_path(&self) -> PathBuf {
        self.dst_path.join("build")
    }

    /// Emits cargo commands required for linking LLZK against a cargo project.
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
        cargo.rerun_if_changed(self.src_path().join("include"))?;
        cargo.rerun_if_changed(self.src_path().join("lib"))?;

        cargo.rustc_link_search(self.lib_path(), Some("native"))?;
        // Adding the whole archive modifier is optional since only seems to be required for some GNU-like linkers.
        let modifiers = whole_archive_config.map(|enable| ("whole-archive", enable));
        for lib in self.libraries()? {
            cargo.rustc_link_lib_static(&lib, modifiers)?;
        }

        Ok(())
    }

    /// Returns the libraries built by CMake.
    fn libraries(&self) -> Result<Vec<String>> {
        // All libraries are installed in the lib path.
        let lib_path = self.lib_path();
        let entries = lib_path
            .read_dir()
            .with_context(|| format!("Failed to read directory {}", lib_path.display()))?;
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

    fn pcl_include_path(&self) -> Option<PathBuf> {
        // The PCL backend include path is in:
        // $build/backends/pcl-conv/include (the include part is added by the helper later.)
        let path = self.build_path().join("backends/pcl-conv");
        path.is_dir().then_some(path)
    }

    fn include_paths(&self) -> Vec<Cow<Path>> {
        // Base paths that we always include.
        [
            Cow::Borrowed(self.dst_path()),
            Cow::Borrowed(self.src_path()),
        ]
        .into_iter()
            // Optionally, add the PCL include path in the dst directory, if present.
            .chain(self.pcl_include_path().map(Cow::Owned))
        .collect()
    }
}

impl BindgenConfig for LlzkBuild<'_> {
    fn apply(&self, bindgen: Builder) -> Result<Builder> {
        let paths = self.include_paths();
        Ok(BindgenConfig::include_paths(
            self,
            bindgen,
            &paths.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        ))
    }
}

impl CCConfig for LlzkBuild<'_> {
    fn apply(&self, cc: &mut Build) -> Result<()> {
        CCConfig::include_paths(self, cc, &[self.dst_path(), self.src_path()]);
        Ok(())
    }
}
