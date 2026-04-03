//! Scatter particles to sorted positions compute shader.
//!
//! Runs per-particle (1D dispatch, 64 threads/workgroup). For each particle,
//! computes its brick_id, atomically increments the write offset for that brick,
//! and copies the particle to the sorted position.
//!
//! This is the third step of the counting sort pipeline:
//! count_per_brick -> prefix_sum -> scatter_particles -> compact_active_bricks
//!
//! Uses a separate write_offset buffer (copy of brick_offset) because atomicAdd
//! modifies the values. The original brick_offset is preserved for later use.

use crate::types::Particle;
use spirv_std::glam::{UVec4, Vec4, UVec3};
use spirv_std::spirv;

/// Push constants for the scatter_particles shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ScatterParticlesPushConstants {
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

/// Copy a particle from the source buffer to the destination buffer at the given index.
///
/// Copies all 9 Vec4/UVec4 fields individually to avoid taking references
/// to buffer elements inside loops (trap #21). Extracted as a helper to
/// satisfy trap #4a (entry points must call a helper).
pub fn copy_particle_to_sorted(
    src_idx: usize,
    dst_idx: usize,
    particles: &[Particle],
    sorted: &mut [Particle],
) {
    let pos_mass = particles[src_idx].pos_mass;
    let vel_temp = particles[src_idx].vel_temp;
    let f_col0 = particles[src_idx].f_col0;
    let f_col1 = particles[src_idx].f_col1;
    let f_col2 = particles[src_idx].f_col2;
    let c_col0 = particles[src_idx].c_col0;
    let c_col1 = particles[src_idx].c_col1;
    let c_col2 = particles[src_idx].c_col2;
    let ids = particles[src_idx].ids;

    sorted[dst_idx].pos_mass = pos_mass;
    sorted[dst_idx].vel_temp = vel_temp;
    sorted[dst_idx].f_col0 = f_col0;
    sorted[dst_idx].f_col1 = f_col1;
    sorted[dst_idx].f_col2 = f_col2;
    sorted[dst_idx].c_col0 = c_col0;
    sorted[dst_idx].c_col1 = c_col1;
    sorted[dst_idx].c_col2 = c_col2;
    sorted[dst_idx].ids = ids;
}

/// Compute shader entry point: scatter particles to sorted positions.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read, source particles).
/// Descriptor set 0, binding 1: storage buffer of `u32` (write_offset, read-write, atomics).
/// Descriptor set 0, binding 2: storage buffer of `Particle` (write, sorted particles).
/// Push constants: `ScatterParticlesPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn scatter_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ScatterParticlesPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] write_offset: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted: &mut [Particle],
) {
    let idx = id.x;
    if idx >= push.num_particles {
        return;
    }

    // Copy position to locals (trap #21)
    let pos_mass = particles[idx as usize].pos_mass;

    let brick_id = crate::compute::count_per_brick::compute_brick_index(
        pos_mass.x,
        pos_mass.y,
        pos_mass.z,
        push.grid_size,
        push.brick_size,
    );

    if brick_id == u32::MAX {
        return;
    }

    // Atomically get the next write position for this brick
    let sorted_idx = unsafe { atomic_add_u32(write_offset, brick_id as usize, 1) };

    // Copy particle to sorted position
    copy_particle_to_sorted(idx as usize, sorted_idx as usize, particles, sorted);
}
