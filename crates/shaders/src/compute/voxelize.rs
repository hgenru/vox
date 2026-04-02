//! Voxelize compute shader.
//!
//! Each thread processes one particle, mapping its position to a voxel cell
//! and writing material_id + temperature to a 3D output buffer.
//! Used for BLAS brick occupancy detection and rendering.

use crate::types::Particle;
use spirv_std::glam::{UVec3, UVec4};
use spirv_std::spirv;

/// Push constants for the voxelize shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VoxelizePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Padding for alignment.
    pub _pad0: u32,
    /// Padding for alignment.
    pub _pad1: u32,
}

/// Voxel cell data packed into a UVec4 for atomic-friendly writes.
///
/// - x: material_id
/// - y: phase
/// - z: temperature as u32 bits (f32 reinterpreted)
/// - w: density counter (number of particles in this voxel)
///
/// When multiple particles map to the same voxel, the one with the
/// highest density counter "wins" via atomic max on the w component.
/// This is a simplification; a proper implementation would use
/// splatting or averaging.

/// Map a particle to a voxel cell and write its data.
///
/// Separated into a helper function for the rust-gpu linker bug workaround
/// (see CLAUDE.md trap #4a).
pub fn write_voxel(
    particle: &Particle,
    voxels: &mut [UVec4],
    grid_size: u32,
) {
    let pos = particle.pos_mass.truncate();

    // Convert continuous position to discrete voxel index
    let vx = pos.x as u32;
    let vy = pos.y as u32;
    let vz = pos.z as u32;

    // Bounds check
    if vx >= grid_size || vy >= grid_size || vz >= grid_size {
        return;
    }

    let index = (vz * grid_size * grid_size + vy * grid_size + vx) as usize;

    let material_id = particle.ids.x;
    let phase = particle.ids.y;
    let temp_bits = particle.vel_temp.w.to_bits();

    // Simple write — last particle wins for now.
    // A more sophisticated approach would use atomicMax on a packed value
    // (e.g., pack material_id + density into a single u32 for atomic comparison).
    // For MVP with low particle counts this is acceptable.
    voxels[index] = UVec4::new(material_id, phase, temp_bits, 1);
}

/// Clear a single voxel cell to empty.
///
/// Helper for the clear pass before voxelization.
pub fn clear_voxel(voxel: &mut UVec4) {
    *voxel = UVec4::ZERO;
}

/// Compute shader entry point: clear voxel grid.
///
/// Must be dispatched before `voxelize` to reset the voxel buffer.
/// Descriptor set 0, binding 0: storage buffer of `UVec4` (voxel grid).
/// Dispatch with `(GRID_SIZE/4, GRID_SIZE/4, GRID_SIZE/4)` workgroups.
#[spirv(compute(threads(4, 4, 4)))]
pub fn clear_voxels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &VoxelizePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] voxels: &mut [UVec4],
) {
    let grid_size = push.grid_size;
    if id.x >= grid_size || id.y >= grid_size || id.z >= grid_size {
        return;
    }
    let index = (id.z * grid_size * grid_size + id.y * grid_size + id.x) as usize;
    clear_voxel(&mut voxels[index]);
}

/// Compute shader entry point: voxelize particles.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `UVec4` (voxel grid, write).
/// Push constants: `VoxelizePushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn voxelize(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &VoxelizePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] voxels: &mut [UVec4],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    write_voxel(&particles[idx], voxels, push.grid_size);
}
