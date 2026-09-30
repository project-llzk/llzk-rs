# llzk-sys

Rust bindings to the LLZK C API.

## Usage

Add `llzk-sys` to your Cargo project.
LLZK is maintained as a submodule and built in place when building the crate.

```
cargo add llzk-sys
```

For building, set the environment variable `MLIR_SYS_230_PREFIX` with the path to a distribution of LLVM 23. For example, to use LLVM 23 installed via Homebrew in macOS (`brew install llvm@23`) set it to `/opt/homebrew/opt/llvm@23/`. This is the only required variable. The recommended environment for building is as follows.

```
export MLIR_SYS_230_PREFIX=...
export TABLEGEN_230_PREFIX=<same as MLIR_SYS_230_PREFIX>
# Some system's default compilers are a bit old and you need a recent version of clang (+18) or gcc (+13).
export CXX=clang++
export CC=clang
# If MLIR's and LLVM's installation is not on standard paths set them here.
# For example, for a homebrew version of LLVM on macOS use this path.
export RUSTFLAGS='-L /opt/homebrew/lib/'
# This variable may need to be configured on macOS as well. If building fails try setting it.
export LIBCLANG_PATH=$MLIR_SYS_230_PREFIX/lib
```

Only Linux and macOS are planned to be supported.
