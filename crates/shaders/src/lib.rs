//! # shaders
//!
//! rust-gpu shader crate compiled to SPIR-V via `spirv-builder`.
//! Contains compute shaders (clear_grid, p2g, grid_update, g2p, voxelize)
//! and graphics shaders (primary_rays, shadow, fullscreen quad).
//!
//! Depends on `shared` (no_std) for Particle/GridCell types.
//!
//! **NOTE:** Entry point functions MUST call at least one helper function,
//! otherwise the SPIR-V linker drops the entry point (rust-gpu bug).

#![cfg_attr(target_arch = "spirv", no_std)]
