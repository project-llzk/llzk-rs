//! Bindgen configuration for the optional in-tree PCL backend.

use anyhow::Result;

use crate::config_traits::bindgen::BindgenConfig;

/// Configures bindings for the PCL backend bundled with LLZK.
#[derive(Debug, Clone)]
pub struct PclConfig {
    is_enabled: bool,
}

impl PclConfig {
    pub const fn new(is_enabled: bool) -> Self {
        Self { is_enabled }
    }
}

impl BindgenConfig for PclConfig {
    fn apply(&self, mut bindgen: bindgen::Builder) -> Result<bindgen::Builder> {
        if self.is_enabled {
            bindgen = bindgen.header_contents(
                "PCL_CAPI.h",
                r#"
#include "pcl/Conversion/ConversionPasses.capi.h.inc"
"#,
            );
        }
        Ok(bindgen)
    }
}
