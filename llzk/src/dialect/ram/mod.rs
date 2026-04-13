//! `ram` dialect.

mod ops;
#[cfg(test)]
mod tests;

pub use ops::{alloc, is_ram_alloc, is_ram_load, is_ram_store, load, store};
