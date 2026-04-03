//! Per-brick sleep state update compute shader with neighbor wake-up.
//!
//! Runs per-brick (1D dispatch, 64 threads/workgroup). For each brick,
//! reads the activity counter from the activity map. Based on how long
//! the brick has been inactive, assigns a graduated tick period:
//!
//! - `1` = tick every frame (active / just went quiet)
//! - `3` = every 3rd frame (cooling down)
//! - `10` = every 10th frame (deeply idle)
//! - `0` = frozen (past sleep threshold)
//!
//! **Neighbor wake-up:** When a brick is active (tick_period > 0), its 6
//! face-adjacent neighbors are also woken to at least tick_period=1.
//! This is critical because P2G scatter writes to a 3x3x3 cell neighborhood,
//! which can span into neighboring bricks. Without this, sleeping neighbors
//! would lose momentum from cross-brick scatter.
//!
//! P2G and G2P use `frame_number % tick_period` to decide whether to skip.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Frames of inactivity before switching from tick_period=1 to tick_period=3.
const COOLDOWN_FAST: u32 = 10;

/// Frames of inactivity before switching from tick_period=3 to tick_period=10.
const COOLDOWN_SLOW: u32 = 20;

/// Push constants for the update_sleep shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct UpdateSleepPushConstants {
    /// Total number of bricks in the grid.
    pub total_bricks: u32,
    /// Number of inactive frames before a brick is fully frozen (tick_period=0).
    pub sleep_threshold: u32,
    /// Number of bricks per axis (grid_size / brick_size).
    pub bricks_per_axis: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

/// Determine tick period for a brick that has been inactive for `counter` frames
/// and the counter is below `COOLDOWN_FAST`.
///
/// Split into separate helpers to avoid >3 if/else branches (trap #15).
pub fn tick_period_from_counter(counter: u32, threshold: u32) -> u32 {
    if counter < COOLDOWN_FAST {
        return 1;
    }
    tick_period_slow(counter, threshold)
}

/// Determine tick period for a brick with counter >= COOLDOWN_FAST.
///
/// Split helper to avoid >3 branches in a single function (trap #15).
pub fn tick_period_slow(counter: u32, threshold: u32) -> u32 {
    if counter < COOLDOWN_SLOW {
        return 3;
    }
    tick_period_frozen(counter, threshold)
}

/// Determine tick period for a brick with counter >= COOLDOWN_SLOW.
///
/// Returns 10 if below threshold, 0 (frozen) if at or above threshold.
/// Split helper to avoid >3 branches in a single function (trap #15).
pub fn tick_period_frozen(counter: u32, threshold: u32) -> u32 {
    if counter < threshold { 10 } else { 0 }
}

/// Update sleep state for a single brick based on its activity counter.
///
/// Returns `(new_counter, tick_period)`:
/// - If the brick was active (`activity > 0`): counter resets to 0, tick_period = 1 (every frame).
/// - If inactive: counter increments and tick_period is graduated based on inactivity duration.
///
/// This helper satisfies trap #4a (entry points must call at least one helper).
pub fn update_brick(activity: u32, counter: u32, threshold: u32) -> (u32, u32) {
    if activity > 0 {
        // Brick had active particles this frame: reset counter, full rate
        (0, 1)
    } else {
        let new_counter = counter + 1;
        (new_counter, tick_period_from_counter(new_counter, threshold))
    }
}

/// Atomically set the any_active flag if a brick is active.
///
/// Uses `atomic_u_max` on SPIR-V targets. Falls back to simple write on CPU.
///
/// # Safety
/// Caller must ensure `any_active[0]` is valid and properly aligned.
pub unsafe fn set_any_active_flag(any_active: &mut [u32], tick_period: u32) {
    if tick_period == 0 {
        return;
    }
    #[cfg(target_arch = "spirv")]
    {
        unsafe {
            spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut any_active[0], 1);
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        any_active[0] = 1;
    }
}

/// Convert a 1D brick index to 3D (bx, by, bz) coordinates.
///
/// Layout: idx = bz * bpa * bpa + by * bpa + bx
pub fn brick_idx_to_3d(idx: u32, bricks_per_axis: u32) -> (u32, u32, u32) {
    let bpa = bricks_per_axis;
    let bz = idx / (bpa * bpa);
    let remainder = idx % (bpa * bpa);
    let by = remainder / bpa;
    let bx = remainder % bpa;
    (bx, by, bz)
}

/// Convert 3D brick coordinates to a 1D index.
pub fn brick_3d_to_idx(bx: u32, by: u32, bz: u32, bricks_per_axis: u32) -> u32 {
    let bpa = bricks_per_axis;
    bz * bpa * bpa + by * bpa + bx
}

/// Wake the 6 face-adjacent neighbors of brick at (bx, by, bz) using atomicMax.
///
/// Sets each neighbor's sleep_state to at least 1 (awake). Uses atomicMax
/// to avoid race conditions when multiple active bricks wake the same neighbor.
///
/// Split into a helper to keep the entry point simple and satisfy trap #4a.
pub fn wake_neighbors(
    bx: u32,
    by: u32,
    bz: u32,
    bricks_per_axis: u32,
    sleep_state: &mut [u32],
) {
    wake_neighbor_neg_x(bx, by, bz, bricks_per_axis, sleep_state);
    wake_neighbor_pos_x(bx, by, bz, bricks_per_axis, sleep_state);
    wake_neighbor_neg_y(bx, by, bz, bricks_per_axis, sleep_state);
    wake_neighbor_pos_y(bx, by, bz, bricks_per_axis, sleep_state);
    wake_neighbor_neg_z(bx, by, bz, bricks_per_axis, sleep_state);
    wake_neighbor_pos_z(bx, by, bz, bricks_per_axis, sleep_state);
}

/// Wake -X neighbor.
fn wake_neighbor_neg_x(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if bx > 0 {
        let nidx = brick_3d_to_idx(bx - 1, by, bz, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Wake +X neighbor.
fn wake_neighbor_pos_x(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if bx + 1 < bpa {
        let nidx = brick_3d_to_idx(bx + 1, by, bz, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Wake -Y neighbor.
fn wake_neighbor_neg_y(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if by > 0 {
        let nidx = brick_3d_to_idx(bx, by - 1, bz, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Wake +Y neighbor.
fn wake_neighbor_pos_y(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if by + 1 < bpa {
        let nidx = brick_3d_to_idx(bx, by + 1, bz, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Wake -Z neighbor.
fn wake_neighbor_neg_z(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if bz > 0 {
        let nidx = brick_3d_to_idx(bx, by, bz - 1, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Wake +Z neighbor.
fn wake_neighbor_pos_z(bx: u32, by: u32, bz: u32, bpa: u32, sleep_state: &mut [u32]) {
    if bz + 1 < bpa {
        let nidx = brick_3d_to_idx(bx, by, bz + 1, bpa) as usize;
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut sleep_state[nidx], 1); }
    }
}

/// Compute shader entry point: update per-brick sleep state with neighbor wake-up.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (activity_map, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (sleep_counter, read-write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (sleep_state, read-write).
/// Descriptor set 0, binding 3: storage buffer of `u32` (any_active flag, write via atomicMax).
/// Push constants: `UpdateSleepPushConstants`.
/// Dispatch with `(ceil(total_bricks / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn update_sleep(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &UpdateSleepPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] activity_map: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sleep_counter: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sleep_state: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] any_active: &mut [u32],
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

    // Set any_active flag if this brick has a non-zero tick period
    unsafe {
        set_any_active_flag(any_active, new_state);
    }

    // If this brick is awake, also wake its 6 face-adjacent neighbors.
    // This is critical: P2G scatter writes to a 3x3x3 cell neighborhood,
    // which can span into adjacent bricks. If those are sleeping, momentum is lost.
    if new_state > 0 {
        let (bx, by, bz) = brick_idx_to_3d(idx, push.bricks_per_axis);
        wake_neighbors(bx, by, bz, push.bricks_per_axis, sleep_state);
    }
}
