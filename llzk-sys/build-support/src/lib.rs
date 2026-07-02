//! Helper modules for building and linking LLZK and generating the Rust bindings.

use std::{io::stdout, path::PathBuf};

use crate::{cargo_commands::whole_archive_config, llzk::LlzkBuild};
use anyhow::Result;

mod cargo_commands;
pub mod config_traits;
pub mod default;
pub mod llzk;
pub mod mlir;
mod pcl;
pub mod wrap_static_fns;

/// Links an existing installation of LLZK.
pub fn link_llzk(path: PathBuf) -> Result<LlzkBuild> {
    let llzk = LlzkBuild::new(path);
    llzk.emit_cargo_commands(stdout(), whole_archive_config())?;
    Ok(llzk)
}
