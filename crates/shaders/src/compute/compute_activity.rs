//! Per-brick activity tracking compute shader.
//!
//! Runs per-particle (1D dispatch, 64 threads/workgroup). For each particle,
//! checks whether it has significant velocity and atomically increments the
//! activity counter of the brick containing it.
//!
//! The activity_map buffer has one u32 per brick (total = (grid_size/brick_size)^3).
//! It must be cleared to zero before dispatch. After dispatch, each entry holds
//! the number of active particles in that brick.

use crate::types::Particle;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the compute_activity shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ComputeActivityPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

/// Squared velocity threshold below which a particle is considered inactive.
const SPEED_SQ_THRESHOLD: f32 = 0.01;

/// Atomically add a value to a u32 in a buffer, returning the old value.
///
/// Uses `spirv_std::arch::atomic_i_add` on SPIR-V targets.
/// Falls back to non-atomic increment on CPU (for testing).
///
/// # Safety
/// Caller must ensure `buffer[index]` is valid and properly aligned for atomic access.
#[inline]
pub unsafe fn atomic_add_u32(buffer: &mut [u32], index: usize, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { spirv_std::arch::atomic_i_add::<u32, 1u32, 0x0u32>(&mut buffer[index], value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let old = buffer[index];
        buffer[index] = old + value;
        old
    }
}

/// Check whether a particle is active and, if so, increment the activity
/// counter for the brick it belongs to.
///
/// A particle is active if its squared velocity exceeds `SPEED_SQ_THRESHOLD`.
/// Pinned solids (phase 0) with negligible velocity are naturally excluded.
///
/// This is extracted as a helper to satisfy trap #4a (entry points must call
/// at least one helper function).
pub fn process_particle(
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    grid_size: u32,
    brick_size: u32,
    activity_map: &mut [u32],
) {
    // Squared speed check
    let speed_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
    if speed_sq <= SPEED_SQ_THRESHOLD {
        return;
    }

    // Compute brick coordinates from particle position
    let bx = (pos_x as u32) / brick_size;
    let by = (pos_y as u32) / brick_size;
    let bz = (pos_z as u32) / brick_size;
    let bricks_per_axis = grid_size / brick_size;

    // Bounds check
    if bx >= bricks_per_axis || by >= bricks_per_axis || bz >= bricks_per_axis {
        return;
    }

    // ZYX order matching grid indexing convention
    let brick_idx = bz * bricks_per_axis * bricks_per_axis + by * bricks_per_axis + bx;

    // Atomic increment activity counter for this brick
    unsafe {
        atomic_add_u32(activity_map, brick_idx as usize, 1);
    }
}

/// Compute shader entry point: per-brick activity tracking.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (activity_map, read-write).
/// Push constants: `ComputeActivityPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn compute_activity(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ComputeActivityPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] activity_map: &mut [u32],
) {
    let idx = id.x;
    if idx >= push.num_particles {
        return;
    }

    // Copy particle fields to locals (trap #21: no references to buffer elements)
    let pos_mass = particles[idx as usize].pos_mass;
    let vel_temp = particles[idx as usize].vel_temp;

    process_particle(
        pos_mass.x,
        pos_mass.y,
        pos_mass.z,
        vel_temp.x,
        vel_temp.y,
        vel_temp.z,
        push.grid_size,
        push.brick_size,
        activity_map,
    );
}
