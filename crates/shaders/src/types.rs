//! GPU-side type definitions mirroring `shared::Particle` and `shared::GridCell`.
//!
//! These use `spirv_std::glam` types which are compatible with the SPIR-V target.
//! The memory layout is identical to the CPU-side types in the `shared` crate
//! (`#[repr(C)]`, same field order, Vec4/UVec4 everywhere).
//!
//! **Why not reuse `shared` directly?**
//! The `shared` crate depends on `glam` from crates.io, which doesn't compile
//! for the `spirv-unknown-vulkan1.3` target. The SPIR-V-compatible glam is
//! re-exported by `spirv_std`. Both are the same version and have identical
//! memory layouts, so `bytemuck::cast` between them is safe.

use spirv_std::glam::{UVec4, Vec4};

/// A single MPM particle. 144 bytes, 16-byte aligned.
///
/// Fields pack multiple values into Vec4 components:
/// - `pos_mass`: xyz = position, w = mass
/// - `vel_temp`: xyz = velocity, w = temperature
/// - `f_col0/1/2`: deformation gradient columns (w = unused)
/// - `c_col0/1/2`: APIC affine momentum columns (w = unused)
/// - `ids`: x = material_id, y = phase, z = object_id, w = padding
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Particle {
    /// Position (xyz) and mass (w).
    pub pos_mass: Vec4,
    /// Velocity (xyz) and temperature (w).
    pub vel_temp: Vec4,
    /// Deformation gradient F, column 0 (xyz), w unused.
    pub f_col0: Vec4,
    /// Deformation gradient F, column 1 (xyz), w unused.
    pub f_col1: Vec4,
    /// Deformation gradient F, column 2 (xyz), w unused.
    pub f_col2: Vec4,
    /// APIC affine momentum C, column 0 (xyz), w unused.
    pub c_col0: Vec4,
    /// APIC affine momentum C, column 1 (xyz), w unused.
    pub c_col1: Vec4,
    /// APIC affine momentum C, column 2 (xyz), w unused.
    pub c_col2: Vec4,
    /// IDs: x = material_id, y = phase, z = object_id, w = padding.
    pub ids: UVec4,
}

/// A single grid cell for MPM transfer. 48 bytes, 16-byte aligned.
///
/// - `velocity_mass`: xyz = velocity (or momentum during P2G), w = mass
/// - `force_pad`: xyz = force, w = solid flag
/// - `temp_pad`: x = accumulated temperature (weighted by mass during P2G,
///   normalized after grid_update), yzw = padding (reserved for future use)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GridCell {
    /// Velocity or momentum (xyz) and mass (w).
    pub velocity_mass: Vec4,
    /// Force (xyz) and solid flag (w).
    pub force_pad: Vec4,
    /// Temperature (x) and padding (yzw).
    pub temp_pad: Vec4,
}

/// Grid dimension (cells per axis). 256³ with 5cm voxels = 12.8m world.
pub const GRID_SIZE: u32 = 256;

/// Default hash grid capacity (synced with shared::constants).
pub const HASH_GRID_DEFAULT_CAPACITY: u32 = 1 << 21;

/// Empty sentinel for hash grid keys (synced with shared::constants).
pub const HASH_GRID_EMPTY_KEY: u32 = 0xFFFF_FFFF;

/// Fixed simulation timestep (seconds).
pub const DT: f32 = 0.001;

/// Gravity acceleration (grid-units/s^2, negative Y).
/// Scaled 20x from 9.81 for 256^3 grid with 5cm voxels.
pub const GRAVITY: f32 = -196.0;

/// Material parameters for GPU storage buffer. 64 bytes per material.
///
/// Mirrors `shared::material::MaterialParams` layout exactly.
/// - `elastic`: x=youngs_modulus, y=poissons_ratio, z=yield_stress, w=viscosity
/// - `thermal`: x=melting_point, y=boiling_point, z=heat_conductivity, w=specific_heat
/// - `visual`: x=density, y=emissive_temp, z=opacity, w=pad
/// - `color`: xyz=rgb (0..1), w=pad
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MaterialParams {
    /// Elastic properties: x=youngs_modulus, y=poissons_ratio, z=yield_stress, w=viscosity.
    pub elastic: Vec4,
    /// Thermal properties: x=melting_point, y=boiling_point, z=heat_conductivity, w=specific_heat.
    pub thermal: Vec4,
    /// Visual properties: x=density, y=emissive_temp, z=opacity, w=pad.
    pub visual: Vec4,
    /// Color: xyz=rgb (0..1), w=pad.
    pub color: Vec4,
}

/// Material ID constants (synced with shared::material).
pub const MAT_WATER: u32 = 1;
pub const MAT_WOOD: u32 = 3;
pub const MAT_ASH: u32 = 4;
pub const MAT_ICE: u32 = 5;
pub const MAT_GUNPOWDER: u32 = 6;

/// Number of materials in the table (synced with shared::material).
pub const MATERIAL_COUNT: usize = 7;

/// Maximum number of phase transition rules (synced with shared::reaction).
pub const MAX_PHASE_RULES: usize = 16;

/// Flags for phase transition behavior (synced with shared::reaction).
pub const FLAG_RESET_F: u32 = 1;
/// Flag to apply explosion behavior on transition.
pub const FLAG_EXPLOSION: u32 = 2;

/// A single phase transition rule. 32 bytes, 16-byte aligned.
///
/// Mirrors `shared::reaction::PhaseTransitionRule` layout exactly.
/// - `materials`: x=from_material, y=from_phase, z=to_material, w=to_phase
/// - `conditions`: x=temp_min, y=temp_max, z=flags (as f32), w=pad
#[repr(C)]
#[derive(Copy, Clone)]
pub struct PhaseTransitionRule {
    /// x = from_material_id, y = from_phase, z = to_material_id, w = to_phase.
    pub materials: UVec4,
    /// x = temp_min, y = temp_max, z = flags (as f32), w = pad.
    pub conditions: Vec4,
}
