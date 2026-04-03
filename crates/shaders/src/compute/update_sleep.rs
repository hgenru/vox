//! Per-brick sleep state update compute shader.
//!
//! Runs per-brick (1D dispatch, 64 threads/workgroup). For each brick,
//! reads the activity counter from the activity map. If the brick has been
//! inactive for `sleep_threshold` consecutive frames, it is marked as sleeping.
//! Active bricks reset their sleep counter immediately.
//!
//! Sleeping bricks are skipped by P2G and G2P to save compute bandwidth.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the update_sleep shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct UpdateSleepPushConstants {
    /// Total number of bricks in the grid.
    pub total_bricks: u32,
    /// Number of inactive frames before a brick is put to sleep.
    pub sleep_threshold: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    /// Padding for 16-byte alignment.
    pub _pad1: u32,
}

/// Update sleep state for a single brick based on its activity counter.
///
/// Returns `(new_counter, new_sleep_state)`:
/// - If the brick was active (`activity > 0`): counter resets to 0, state = awake (0).
/// - If inactive: counter increments. Once it reaches the threshold, state = sleeping (1).
///
/// This helper satisfies trap #4a (entry points must call at least one helper).
pub fn update_brick(activity: u32, counter: u32, threshold: u32) -> (u32, u32) {
    if activity > 0 {
        // Brick had active particles this frame: reset counter, mark awake
        (0, 0)
    } else {
        let new_counter = counter + 1;
        if new_counter >= threshold {
            // Brick has been inactive long enough: mark sleeping
            (new_counter, 1)
        } else {
            // Still counting inactive frames, not sleeping yet
            (new_counter, 0)
        }
    }
}

/// Compute shader entry point: update per-brick sleep state.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (activity_map, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (sleep_counter, read-write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (sleep_state, write).
/// Push constants: `UpdateSleepPushConstants`.
/// Dispatch with `(ceil(total_bricks / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn update_sleep(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &UpdateSleepPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] activity_map: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sleep_counter: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sleep_state: &mut [u32],
) {
    let idx = id.x;
    if idx >= push.total_bricks {
        return;
    }

    // Copy to locals to avoid reference issues (trap #21)
    let activity = activity_map[idx as usize];
    let counter = sleep_counter[idx as usize];

    let (new_counter, new_state) = update_brick(activity, counter, push.sleep_threshold);

    sleep_counter[idx as usize] = new_counter;
    sleep_state[idx as usize] = new_state;
}
