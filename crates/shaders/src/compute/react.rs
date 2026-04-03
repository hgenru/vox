//! Chemical reaction compute shader.
//!
//! Runs AFTER voxelize, BEFORE render. Each thread processes one particle,
//! checking 6 face-neighbors in the voxel grid for reaction partners.
//!
//! Reactions:
//! - Water (material_id=1, phase=1) + adjacent Lava (material_id=2) -> Steam (material_id=1, phase=2, temp=100)
//! - Lava (material_id=2, phase=1) + adjacent Water (material_id=1) -> Stone (material_id=0, phase=0, temp=300)
//! - Wood (material_id=3) with T > 300 -> heats up, accumulates damage -> Ash (material_id=4)

use crate::types::{Particle, MAT_ASH, MAT_WOOD, DT};
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
    voxel.w > 0 && ((voxel.x >> 16) & 0xFF) == target_material
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

/// Simple hash for pseudo-random per-particle rate limiting.
///
/// Uses Knuth's multiplicative hash to produce a deterministic but well-distributed
/// value from a particle index and frame offset.
pub fn particle_hash(idx: u32, frame_offset: u32) -> u32 {
    let mut h = idx;
    h = h.wrapping_mul(2654435761); // Knuth's multiplicative hash
    h ^= h >> 16;
    h = h.wrapping_add(frame_offset);
    h ^= h >> 13;
    h
}

/// Process a single particle for chemical reactions.
///
/// Checks 6 face-neighbors in the voxel buffer and applies:
/// - Water + adjacent Lava -> Steam (water boils: material_id=1, phase=2, temp=100)
/// - Lava + adjacent Water -> Stone (lava cools: material_id=0, phase=0, temp=300)
/// - Wood with T > 300 -> heats up, accumulates damage -> Ash when fully burnt
///
/// Liquid reactions are rate-limited: only ~1.5% chance per particle per frame to avoid
/// instant mass conversion and visual flickering. Wood burning runs every frame.
pub fn react_particle(
    particle: &mut Particle,
    voxels: &[UVec4],
    grid_size: u32,
    particle_index: u32,
) {
    let material_id = particle.ids.x;
    let temperature = particle.vel_temp.w;

    // Wood burning: runs every frame (not rate-limited) for smooth progression
    if material_id == MAT_WOOD && temperature > 300.0 {
        react_wood_burning(particle);
        return;
    }

    // Rate-limit liquid reactions: only 1/64 chance (~1.5%) per particle per frame.
    // This spreads conversions over many frames for gradual visual transition.
    let hash = particle_hash(particle_index, 0);
    if hash % 64 != 0 {
        return;
    }

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

    // Water (material_id=1) + adjacent Lava (material_id=2) -> Steam
    // Water boils from contact with lava
    if material_id == 1
        && any_face_neighbor_has_material(voxels, grid_size, vx, vy, vz, 2)
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
        return;
    }

    // Lava (material_id=2) + adjacent Water (material_id=1) -> Stone
    // Lava cools from contact with water
    if material_id == 2
        && any_face_neighbor_has_material(voxels, grid_size, vx, vy, vz, 1)
    {
        // Become stone: material_id=0, phase=0 (solid)
        particle.ids.x = 0;
        particle.ids.y = 0;
        // Set temperature to 300 (cooled stone)
        particle.vel_temp = Vec4::new(
            particle.vel_temp.x,
            particle.vel_temp.y,
            particle.vel_temp.z,
            300.0,
        );
        // Reset deformation gradient (trap #8)
        particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
}

/// Process wood burning: temperature rises, damage accumulates, converts to ash.
///
/// Wood with temperature above 300 is on fire. Each frame the temperature
/// increases (self-sustaining combustion) and damage (ids.z) accumulates.
/// When damage reaches 200 the particle converts to ash.
fn react_wood_burning(particle: &mut Particle) {
    // Wood is on fire — heat up further (self-sustaining combustion)
    particle.vel_temp = Vec4::new(
        particle.vel_temp.x,
        particle.vel_temp.y,
        particle.vel_temp.z,
        particle.vel_temp.w + 50.0 * DT,
    );

    // Accumulate damage (stored in ids.z)
    let damage = particle.ids.z + 1;
    particle.ids.z = damage;

    // When fully burnt -> become ash
    if damage >= 200 {
        particle.ids.x = MAT_ASH;
        particle.ids.y = 0; // solid phase
        particle.ids.z = 0; // reset damage
        particle.vel_temp = Vec4::new(
            particle.vel_temp.x,
            particle.vel_temp.y,
            particle.vel_temp.z,
            200.0, // still warm
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
    react_particle(&mut particles[idx], voxels, push.grid_size, id.x);
}
