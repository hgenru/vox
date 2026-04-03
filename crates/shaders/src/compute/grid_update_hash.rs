//! Hash grid update compute shader.
//!
//! Processes only occupied hash grid slots (key != EMPTY) instead of
//! the entire grid. Each thread checks one slot and applies physics
//! (momentum -> velocity, gravity, boundaries) if the slot is occupied.
//!
//! Workgroup size: 64 threads, 1D dispatch.
//! Dispatch with `(ceil(capacity / 64), 1, 1)` workgroups.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

use super::grid_update_sparse::{apply_boundary, apply_solid_damping};
use super::hash_grid;

/// Push constants for the hash grid update shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GridUpdateHashPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y, e.g., -196.0).
    pub gravity: f32,
    /// Hash grid capacity (number of slots).
    pub hash_capacity: u32,
}

/// Update a single occupied hash grid cell.
///
/// Same physics as `grid_update_sparse::update_active_cell` but takes
/// coordinates unpacked from the hash key.
/// Helper function for trap #4a compliance.
pub fn update_hash_cell(
    cell: &mut GridCell,
    ix: u32,
    iy: u32,
    iz: u32,
    grid_size: u32,
    dt: f32,
    gravity: f32,
) {
    let mass = cell.velocity_mass.w;

    if mass < 1.0e-6 {
        cell.velocity_mass = Vec4::ZERO;
        cell.temp_pad = Vec4::ZERO;
        return;
    }

    let mut vx = cell.velocity_mass.x / mass;
    let mut vy = cell.velocity_mass.y / mass;
    let mut vz = cell.velocity_mass.z / mass;

    vx += (cell.force_pad.x / mass) * dt;
    vy += (cell.force_pad.y / mass) * dt;
    vz += (cell.force_pad.z / mass) * dt;

    vy += gravity * dt;

    let (bx, by, bz) = apply_boundary(vx, vy, vz, ix, iy, iz, grid_size);
    vx = bx;
    vy = by;
    vz = bz;

    let solid_flag = cell.force_pad.w;
    let (sx, sy, sz) = apply_solid_damping(vx, vy, vz, solid_flag);
    vx = sx;
    vy = sy;
    vz = sz;

    cell.velocity_mass = Vec4::new(vx, vy, vz, mass);

    let cell_temp = cell.temp_pad.x / mass;
    cell.temp_pad = Vec4::new(cell_temp, 0.0, 0.0, 0.0);
}

/// Compute shader entry point: hash grid velocity update.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (hash_keys, read).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (hash_values, read-write).
/// Push constants: `GridUpdateHashPushConstants`.
#[spirv(compute(threads(64)))]
pub fn grid_update_hash(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &GridUpdateHashPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] hash_keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hash_values: &mut [GridCell],
) {
    let idx = id.x;
    if idx >= push.hash_capacity {
        return;
    }

    // Skip empty slots
    let key = hash_keys[idx as usize];
    if key == hash_grid::HASH_GRID_EMPTY_KEY {
        return;
    }

    let (ix, iy, iz) = hash_grid::unpack_key(key);
    update_hash_cell(&mut hash_values[idx as usize], ix, iy, iz, push.grid_size, push.dt, push.gravity);
}
