//! Helper type for emitting cargo commands.

use std::{
    io::{Result as IOResult, Write},
    path::{Path, PathBuf},
};

/// Returns configuration for the linker regarding the `whole-archive` flag.
///
/// If the env var `LLZK_SYS_ENABLE_WHOLE_ARCHIVE` is not set returns `None`.
/// If its set, if the value is '0' returns `Some(false)`, otherwise
/// returns `Some(true)`.
pub(crate) fn whole_archive_config() -> Option<bool> {
    std::env::var("LLZK_SYS_ENABLE_WHOLE_ARCHIVE")
        .ok()
        .map(|var| var != "0")
}

/// Helper struct for emitting cargo commands to keep the emitter code more idiomatic.
pub(crate) struct CargoCommands<W>(W);

impl<W: Write> CargoCommands<W> {
    pub fn new(out: W) -> Self {
        Self(out)
    }

    /// Emits `cargo:rerun-if-changed`.
    pub fn rerun_if_changed(&mut self, path: impl AsRef<Path>) -> IOResult<()> {
        writeln!(self.0, "cargo:rerun-if-changed={}", path.as_ref().display())
    }

    /// Emits `cargo:rustc-link-search`
    pub fn rustc_link_search(
        &mut self,
        path: impl AsRef<Path>,
        modifiers: Option<&str>,
    ) -> IOResult<()> {
        write!(self.0, "cargo:rustc-link-search")?;
        if let Some(modifiers) = modifiers {
            write!(self.0, "={modifiers}")?;
        }
        writeln!(self.0, "={}", path.as_ref().display())
    }

    /// Emits `cargo:rustc-link-lib` with the `static` flag always on.
    pub fn rustc_link_lib_static<'s>(
        &mut self,
        lib: &str,
        modifiers: impl IntoIterator<Item = (&'s str, bool)>,
    ) -> IOResult<()> {
        write!(self.0, "cargo:rustc-link-lib=static")?;
        for (n, (modifier, enable)) in modifiers.into_iter().enumerate() {
            write!(
                self.0,
                "{}{}{}",
                // The modifier list is a comma separated list prefixed with ':'.
                if n == 0 { ":" } else { "," },
                if enable { "+" } else { "-" },
                modifier
            )?;
        }
        writeln!(self.0, "={lib}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llzk::LIBDIR;
    use std::io::Cursor;
    use tempfile::TempDir;

    macro_rules! cargo_command_test {
        ($name:ident, $cargo:ident, $t:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let mut buff = Vec::new();
                let mut $cargo = CargoCommands(Cursor::new(&mut buff));
                $t;
                let command = String::from_utf8(buff).unwrap();
                assert_eq!(command.trim(), $expected);
            }
        };
    }

    cargo_command_test!(
        test_rerun_if_changed,
        cargo,
        {
            cargo.rerun_if_changed(Path::new("example/path")).unwrap();
        },
        "cargo:rerun-if-changed=example/path"
    );

    cargo_command_test!(
        test_rustc_link_search_no_mod,
        cargo,
        {
            cargo
                .rustc_link_search(Path::new("example/path"), None)
                .unwrap();
        },
        "cargo:rustc-link-search=example/path"
    );

    cargo_command_test!(
        test_rustc_link_search_with_mod,
        cargo,
        {
            cargo
                .rustc_link_search(Path::new("example/path"), Some("native"))
                .unwrap();
        },
        "cargo:rustc-link-search=native=example/path"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_no_mod,
        cargo,
        {
            cargo.rustc_link_lib_static("example", None).unwrap();
        },
        "cargo:rustc-link-lib=static=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_1,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", true)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:+mod=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_2,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", false)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:-mod=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_3,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", true), ("other", true)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:+mod,+other=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_4,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", false), ("other", true)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:-mod,+other=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_5,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", true), ("other", false)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:+mod,-other=example"
    );

    cargo_command_test!(
        test_rustc_link_lib_static_with_mods_6,
        cargo,
        {
            cargo
                .rustc_link_lib_static("example", [("mod", false), ("other", false)])
                .unwrap();
        },
        "cargo:rustc-link-lib=static:-mod,-other=example"
    );

    fn setup_llzk<'s>(
        src: &'s Path,
        dst: &Path,
        libraries: &[&str],
        others: &[&str],
    ) -> LlzkBuild<'s> {
        let libdir = dst.join(LIBDIR);
        std::fs::create_dir(&libdir).unwrap();
        for l in libraries {
            std::fs::write(libdir.join(format!("lib{l}.a")), []).unwrap();
        }
        for o in others {
            std::fs::write(libdir.join(o), []).unwrap();
        }
        LlzkBuild::new(src, dst.to_owned())
    }

    fn emit_commands(llzk: &LlzkBuild, wac: Option<bool>) -> Vec<String> {
        let mut buff = Vec::new();
        llzk.emit_cargo_commands(Cursor::new(&mut buff), wac)
            .unwrap();
        let mut cmds: Vec<_> = String::from_utf8(buff)
            .unwrap()
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        cmds.sort();
        cmds
    }

    #[test]
    fn test_llzk_cargo_commands() {
        let src = TempDir::with_prefix("src").unwrap();
        let dst = TempDir::with_prefix("dst").unwrap();
        let libraries = ["XXX", "YYY"];
        let others = ["other file"];
        let llzk = setup_llzk(src.path(), dst.path(), &libraries, &others);

        let commands = emit_commands(&llzk, None);
        let expected = vec![
            format!("cargo:rerun-if-changed={}/include", src.path().display()),
            format!("cargo:rerun-if-changed={}/lib", src.path().display()),
            "cargo:rustc-link-lib=static=XXX".to_string(),
            "cargo:rustc-link-lib=static=YYY".to_string(),
            format!(
                "cargo:rustc-link-search=native={}",
                dst.path().join(LIBDIR).display()
            ),
        ];
        assert_eq!(commands, expected)
    }

    #[test]
    fn test_llzk_cargo_commands_no_whole_archive() {
        let src = TempDir::with_prefix("src").unwrap();
        let dst = TempDir::with_prefix("dst").unwrap();
        let libraries = ["XXX", "YYY"];
        let others = ["other file"];
        let llzk = setup_llzk(src.path(), dst.path(), &libraries, &others);

        let commands = emit_commands(&llzk, Some(false));
        let expected = vec![
            format!("cargo:rerun-if-changed={}/include", src.path().display()),
            format!("cargo:rerun-if-changed={}/lib", src.path().display()),
            "cargo:rustc-link-lib=static:-whole-archive=XXX".to_string(),
            "cargo:rustc-link-lib=static:-whole-archive=YYY".to_string(),
            format!(
                "cargo:rustc-link-search=native={}",
                dst.path().join(LIBDIR).display()
            ),
        ];
        assert_eq!(commands, expected)
    }

    #[test]
    fn test_llzk_cargo_commands_with_whole_archive() {
        let src = TempDir::with_prefix("src").unwrap();
        let dst = TempDir::with_prefix("dst").unwrap();
        let libraries = ["XXX", "YYY"];
        let others = ["other file"];
        let llzk = setup_llzk(src.path(), dst.path(), &libraries, &others);

        let commands = emit_commands(&llzk, Some(true));
        let expected = vec![
            format!("cargo:rerun-if-changed={}/include", src.path().display()),
            format!("cargo:rerun-if-changed={}/lib", src.path().display()),
            "cargo:rustc-link-lib=static:+whole-archive=XXX".to_string(),
            "cargo:rustc-link-lib=static:+whole-archive=YYY".to_string(),
            format!(
                "cargo:rustc-link-search=native={}",
                dst.path().join(LIBDIR).display()
            ),
        ];
        assert_eq!(commands, expected)
    }
}
