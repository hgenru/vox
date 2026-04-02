//! Shader builder binary.
//!
//! This crate exists to compile the `shaders` crate to SPIR-V via
//! `spirv-builder` in its `build.rs`. Running `cargo build -p shader-builder`
//! triggers SPIR-V compilation.
//!
//! The compiled SPIR-V module path is available at build time via the
//! `SHADERS_SPV_PATH` environment variable. All entry points (clear_grid,
//! p2g, grid_update, g2p, voxelize, clear_voxels) are in a single module.

fn main() {
    let spv_path = env!("SHADERS_SPV_PATH");
    println!("Shader SPIR-V module compiled at: {spv_path}");
}
