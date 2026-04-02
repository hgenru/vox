//! # shared
//!
//! Core types shared between CPU and GPU: `Particle`, `GridCell`, `MaterialParams`.
//! All structs are `#[repr(C)]` + `bytemuck::Pod` for zero-copy GPU upload.
//!
//! **CRITICAL:** Uses `Vec4` everywhere — never `Vec3` in GPU structs.
//! This crate is `no_std`-compatible (disable default feature `std`).

#![cfg_attr(not(feature = "std"), no_std)]

pub mod constants;
pub mod material;
pub mod particle;

pub use constants::*;
pub use material::*;
pub use particle::*;
