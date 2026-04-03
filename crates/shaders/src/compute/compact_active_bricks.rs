//! Compact active bricks compute shader.
//!
//! Runs per-brick (1D dispatch, 64 threads/workgroup). For each brick,
//! checks if it has particles (brick_count > 0) and is not sleeping
//! (sleep_state != 0). If both conditions are met, appends the brick_id
//! to the active_brick_list using an atomic counter.
//!
//! This is the fourth step of the counting sort pipeline:
//! count_per_brick -> prefix_sum -> scatter_particles -> compact_active_bricks

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the compact_active_bricks shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CompactActiveBricksPushConstants {
    /// Total number of bricks.
    pub total_bricks: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    /// Padding for 16-byte alignment.
    pub _pad1: u32,
    /// Padding for 16-byte alignment.
    pub _pad2: u32,
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

/// Check if a brick should be added to the active list and append it if so.
///
/// A brick is active if it has particles (count > 0) and is not fully asleep
/// (sleep_state != 0). Sleep_state of 0 means frozen; any non-zero value means
/// the brick has a tick period and should be processed.
///
/// Extracted as a helper to satisfy trap #4a (entry points must call a helper).
pub fn try_compact_brick(
    brick_id: u32,
    total_bricks: u32,
    brick_count: &[u32],
    sleep_state: &[u32],
    active_brick_list: &mut [u32],
    active_brick_count: &mut [u32],
) {
    if brick_id >= total_bricks {
        return;
    }

    let count = brick_count[brick_id as usize];
    let sleep = sleep_state[brick_id as usize];

    // Include brick if it has particles and is not frozen (sleep_state == 0 means frozen)
    if count > 0 && sleep != 0 {
        let slot = unsafe { atomic_add_u32(active_brick_count, 0, 1) };
        active_brick_list[slot as usize] = brick_id;
    }
}

/// Compute shader entry point: compact active bricks into a list.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (brick_count, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (sleep_state, read).
/// Descriptor set 0, binding 2: storage buffer of `u32` (active_brick_list, write).
/// Descriptor set 0, binding 3: storage buffer of `u32` (active_brick_count, read-write).
/// Push constants: `CompactActiveBricksPushConstants`.
/// Dispatch with `(ceil(total_bricks / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn compact_active_bricks(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &CompactActiveBricksPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] brick_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sleep_state: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_brick_list: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_brick_count: &mut [u32],
) {
    let brick_id = id.x;
    try_compact_brick(
        brick_id,
        push.total_bricks,
        brick_count,
        sleep_state,
        active_brick_list,
        active_brick_count,
    );
}
