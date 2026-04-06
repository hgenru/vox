//! # shared
//!
//! Core types shared between CPU and GPU: `Particle`, `GridCell`, `MaterialParams`.
//! All structs are `#[repr(C)]` + `bytemuck::Pod` for zero-copy GPU upload.
//!
//! **CRITICAL:** Uses `Vec4` everywhere — never `Vec3` in GPU structs.
//! This crate is `no_std`-compatible (disable default feature `std`).

#![cfg_attr(not(feature = "std"), no_std)]

pub mod chunk_gpu;
pub mod config;
pub mod constants;
pub mod far_field;
pub mod indirect;
pub mod material;
pub mod material_ca;
pub mod particle;
pub mod phase;
pub mod physics;
pub mod reaction;
pub mod reaction_ca;
pub mod svd;
pub mod voxel;

pub use config::*;
pub use constants::*;
pub use far_field::*;
pub use indirect::IndirectDispatchArgs;
pub use material::*;
pub use particle::*;
