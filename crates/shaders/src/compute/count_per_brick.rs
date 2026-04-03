//! Count particles per brick compute shader.
//!
//! Runs per-particle (1D dispatch, 64 threads/workgroup). For each particle,
//! computes the brick_id from its position and atomically increments the
//! corresponding entry in the brick_count buffer.
//!
//! This is the first step of the counting sort pipeline:
//! count_per_brick -> prefix_sum -> scatter_particles -> compact_active_bricks

use crate::types::Particle;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the count_per_brick shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CountPerBrickPushConstants {
    /// Total number of particles.
    pub num_particles: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

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

/// Compute the brick index for a particle at the given position.
///
/// Returns the flat ZYX brick index, or `u32::MAX` if out of bounds.
/// Extracted as a helper to satisfy trap #4a (entry points must call a helper).
pub fn compute_brick_index(pos_x: f32, pos_y: f32, pos_z: f32, grid_size: u32, brick_size: u32) -> u32 {
    let bx = (pos_x as u32) / brick_size;
    let by = (pos_y as u32) / brick_size;
    let bz = (pos_z as u32) / brick_size;
    let bricks_per_axis = grid_size / brick_size;

    if bx >= bricks_per_axis || by >= bricks_per_axis || bz >= bricks_per_axis {
        return u32::MAX;
    }

    // ZYX order matching grid indexing convention
    bz * bricks_per_axis * bricks_per_axis + by * bricks_per_axis + bx
}

/// Compute shader entry point: count particles per brick.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (brick_count, read-write).
/// Push constants: `CountPerBrickPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn count_per_brick(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &CountPerBrickPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] brick_count: &mut [u32],
) {
    let idx = id.x;
    if idx >= push.num_particles {
        return;
    }

    // Copy to locals to avoid buffer references in loops (trap #21)
    let pos_mass = particles[idx as usize].pos_mass;

    let brick_id = compute_brick_index(
        pos_mass.x,
        pos_mass.y,
        pos_mass.z,
        push.grid_size,
        push.brick_size,
    );

    if brick_id != u32::MAX {
        unsafe {
            atomic_add_u32(brick_count, brick_id as usize, 1);
        }
    }
}
