//! Chemical reaction compute shader.
//!
//! Runs AFTER voxelize, BEFORE render. Each thread processes one particle,
//! checking 6 face-neighbors in the voxel grid for reaction partners.
//!
//! Reactions:
//! - Water (material_id=1, phase=1) + adjacent Lava (material_id=2) -> Stone (material_id=0, phase=0)
//! - Lava (material_id=2, phase=1) + adjacent Water (material_id=1) -> Steam (material_id=1, phase=2, temp=100)

use crate::types::Particle;
use spirv_std::glam::{UVec3, UVec4, Vec4};
use spirv_std::spirv;

/// Push constants for the react shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ReactPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding for alignment.
    pub _pad0: u32,
    /// Padding for alignment.
    pub _pad1: u32,
}

/// Check whether a voxel at given coordinates contains the specified material.
///
/// Returns true if the voxel is occupied (w > 0) and its material_id matches.
fn has_neighbor_material(
    voxels: &[UVec4],
    grid_size: u32,
    x: i32,
    y: i32,
    z: i32,
    target_material: u32,
) -> bool {
    let gs = grid_size as i32;
    if x < 0 || x >= gs || y < 0 || y >= gs || z < 0 || z >= gs {
        return false;
    }
    let idx = (z as u32 * grid_size * grid_size + y as u32 * grid_size + x as u32) as usize;
    let voxel = voxels[idx];
    voxel.w > 0 && voxel.x == target_material
}

/// Check the 6 face-neighbors of a particle's voxel position for a given material.
///
/// Returns true if any of the +/-x, +/-y, +/-z neighbors contains the target material.
fn any_face_neighbor_has_material(
    voxels: &[UVec4],
    grid_size: u32,
    vx: i32,
    vy: i32,
    vz: i32,
    target_material: u32,
) -> bool {
    has_neighbor_material(voxels, grid_size, vx - 1, vy, vz, target_material)
        || has_neighbor_material(voxels, grid_size, vx + 1, vy, vz, target_material)
        || has_neighbor_material(voxels, grid_size, vx, vy - 1, vz, target_material)
        || has_neighbor_material(voxels, grid_size, vx, vy + 1, vz, target_material)
        || has_neighbor_material(voxels, grid_size, vx, vy, vz - 1, target_material)
        || has_neighbor_material(voxels, grid_size, vx, vy, vz + 1, target_material)
}

/// Process a single particle for chemical reactions.
///
/// Checks 6 face-neighbors in the voxel buffer and applies:
/// - Water + adjacent Lava -> Stone (reset F = Identity)
/// - Lava + adjacent Water -> Steam (reset F = Identity, temp = 100)
pub fn react_particle(
    particle: &mut Particle,
    voxels: &[UVec4],
    grid_size: u32,
) {
    let material_id = particle.ids.x;
    let phase = particle.ids.y;

    // Only react liquids (phase == 1)
    if phase != 1 {
        return;
    }

    let pos = particle.pos_mass.truncate();
    let vx = pos.x as i32;
    let vy = pos.y as i32;
    let vz = pos.z as i32;

    // Bounds check
    let gs = grid_size as i32;
    if vx < 0 || vx >= gs || vy < 0 || vy >= gs || vz < 0 || vz >= gs {
        return;
    }

    // Water (material_id=1) + adjacent Lava (material_id=2) -> Stone
    if material_id == 1
        && any_face_neighbor_has_material(voxels, grid_size, vx, vy, vz, 2)
    {
        // Become stone: material_id=0, phase=0 (solid)
        particle.ids.x = 0;
        particle.ids.y = 0;
        // Reset deformation gradient (trap #8)
        particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
        return;
    }

    // Lava (material_id=2) + adjacent Water (material_id=1) -> Steam
    if material_id == 2
        && any_face_neighbor_has_material(voxels, grid_size, vx, vy, vz, 1)
    {
        // Become steam: material_id=1 (water material), phase=2 (gas)
        particle.ids.x = 1;
        particle.ids.y = 2;
        // Set temperature to 100 (boiling point)
        particle.vel_temp = Vec4::new(
            particle.vel_temp.x,
            particle.vel_temp.y,
            particle.vel_temp.z,
            100.0,
        );
        // Reset deformation gradient (trap #8)
        particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
}

/// Compute shader entry point: chemical reactions.
///
/// Runs after voxelize to check particle neighbors via the voxel grid.
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read-write).
/// Descriptor set 0, binding 1: storage buffer of `UVec4` (voxel grid, read).
/// Push constants: `ReactPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn react(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ReactPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] voxels: &[UVec4],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    react_particle(&mut particles[idx], voxels, push.grid_size);
}
