//! # shaders
//!
//! rust-gpu shader crate compiled to SPIR-V via `spirv-builder`.
//! Contains compute shaders (clear_grid, p2g, grid_update, g2p, voxelize)
//! and graphics shaders (primary_rays, shadow, fullscreen quad).
//!
//! Uses local type definitions that mirror `shared::Particle`/`shared::GridCell`
//! with identical `#[repr(C)]` memory layout. This is necessary because the
//! `shared` crate depends on crates.io `glam` which doesn't compile for SPIR-V;
//! we use `spirv_std::glam` instead.
//!
//! **NOTE:** Entry point functions MUST call at least one helper function,
//! otherwise the SPIR-V linker drops the entry point (rust-gpu bug).

#![cfg_attr(target_arch = "spirv", no_std)]

pub mod ca_types;
pub mod compute;
pub mod pbmpm_types;
pub mod types;
