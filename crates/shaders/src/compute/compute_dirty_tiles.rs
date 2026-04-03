//! Compute dirty tiles shader for selective rendering.
//!
//! Divides the screen into 16x16 pixel tiles and marks each tile as dirty
//! (needs re-render) or clean (can reuse previous frame data).
//!
//! A tile is dirty if:
//! - The camera moved (all tiles dirty), OR
//! - Any brick that projects into the tile's screen region is active
//!   (sleep_state != 0, i.e., not frozen)
//!
//! Dispatched 1D with 64 threads/workgroup, one thread per tile.

use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Tile size in pixels (16x16).
const TILE_SIZE: u32 = 16;

/// Push constants for the compute_dirty_tiles shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DirtyTilesPushConstants {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Number of bricks per axis (grid_size / 8).
    pub bricks_per_axis: u32,
    /// Current camera eye position (xyz), w = camera_moved flag (1.0 = moved).
    pub eye: Vec4,
    /// Current camera target position (xyz), w unused.
    pub target: Vec4,
}

/// Check if the camera has moved by inspecting the flag in eye.w.
///
/// Helper function to satisfy trap #4a (entry points must call at least one helper).
fn camera_moved(push: &DirtyTilesPushConstants) -> bool {
    push.eye.w > 0.5
}

/// Mark all tiles as dirty when camera moved, or check sleep_state per brick.
///
/// For the simple approach: if camera moved, all tiles are dirty.
/// If camera is static, we check all bricks for activity. Any brick that is
/// not frozen (sleep_state != 0) means any tile that could see that brick
/// should be re-rendered. Since projecting bricks to screen tiles is expensive,
/// we use a simpler heuristic: if ANY brick is active, mark ALL tiles dirty.
/// This still saves rendering when the entire world is static.
fn compute_tile_dirty(
    tile_idx: u32,
    total_tiles: u32,
    total_bricks: u32,
    is_camera_moved: bool,
    sleep_state: &[u32],
) -> u32 {
    if tile_idx >= total_tiles {
        return 0;
    }

    if is_camera_moved {
        return 1;
    }

    // Check if any brick is active (not frozen).
    // sleep_state: 0 = frozen, 1 = every frame, 3 = every 3rd, 10 = every 10th
    check_any_brick_active(total_bricks, sleep_state)
}

/// Scan bricks to see if any is not frozen (sleep_state != 0).
///
/// Returns 1 if any active brick found, 0 otherwise.
/// Split into helper to keep branch count low (trap #15).
fn check_any_brick_active(total_bricks: u32, sleep_state: &[u32]) -> u32 {
    let mut i = 0u32;
    while i < total_bricks {
        if sleep_state[i as usize] != 0 {
            return 1;
        }
        i += 1;
    }
    0
}

/// Compute shader entry point: compute dirty tiles.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (sleep_state, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (dirty_tile_buffer, write).
/// Push constants: `DirtyTilesPushConstants`.
/// Dispatch with `(ceil(total_tiles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn compute_dirty_tiles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &DirtyTilesPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] sleep_state: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dirty_tiles: &mut [u32],
) {
    let tile_idx = id.x;
    let tiles_x = (push.width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (push.height + TILE_SIZE - 1) / TILE_SIZE;
    let total_tiles = tiles_x * tiles_y;
    let bpa = push.bricks_per_axis;
    let total_bricks = bpa * bpa * bpa;

    let is_moved = camera_moved(push);
    let dirty = compute_tile_dirty(tile_idx, total_tiles, total_bricks, is_moved, sleep_state);

    if tile_idx < total_tiles {
        dirty_tiles[tile_idx as usize] = dirty;
    }
}
