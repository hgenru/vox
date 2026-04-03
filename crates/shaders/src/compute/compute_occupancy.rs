//! Per-brick occupancy compute shader.
//!
//! Runs after the voxelize pass. Dispatched 1D with 64 threads/workgroup,
//! one thread per brick. Each thread scans the 8x8x8 voxels in its brick
//! and writes 1 to `brick_occupied[brick_idx]` if any voxel is non-empty,
//! or 0 otherwise.
//!
//! The render shader uses this map to skip empty bricks during DDA ray march,
//! turning O(grid_size) per-voxel checks into O(bricks_per_axis) per-brick checks
//! for sky rays that miss all geometry.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the compute_occupancy shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ComputeOccupancyPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Number of bricks per axis (grid_size / brick_size).
    pub bricks_per_axis: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

/// Number of u32 values per voxel cell in the voxel buffer.
const U32S_PER_VOXEL: u32 = 4;

/// Scan an 8x8x8 brick and determine if any voxel is occupied.
///
/// Returns `true` if any voxel in the brick has a non-zero occupied slot
/// (offset +3 in the 4-u32-per-cell layout). Returns early on first hit.
///
/// This is extracted as a helper to satisfy trap #4a (entry points must call
/// at least one helper function).
pub fn scan_brick(
    bx: u32,
    by: u32,
    bz: u32,
    brick_size: u32,
    grid_size: u32,
    voxels: &[u32],
) -> bool {
    let base_x = bx * brick_size;
    let base_y = by * brick_size;
    let base_z = bz * brick_size;

    let mut dz = 0u32;
    while dz < brick_size {
        let mut dy = 0u32;
        while dy < brick_size {
            let mut dx = 0u32;
            while dx < brick_size {
                let cx = base_x + dx;
                let cy = base_y + dy;
                let cz = base_z + dz;

                // Bounds check (should always pass if bricks_per_axis is correct)
                if cx < grid_size && cy < grid_size && cz < grid_size {
                    let cell_idx = cz * grid_size * grid_size + cy * grid_size + cx;
                    // Occupied slot is at offset +3 in the 4-u32-per-cell layout
                    let occupied_offset = (cell_idx * U32S_PER_VOXEL + 3) as usize;
                    if voxels[occupied_offset] > 0 {
                        return true;
                    }
                }

                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }

    false
}

/// Compute shader entry point: per-brick occupancy detection.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (voxel grid, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (brick_occupied, write).
/// Push constants: `ComputeOccupancyPushConstants`.
/// Dispatch with `(ceil(total_bricks / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn compute_occupancy(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ComputeOccupancyPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] voxels: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] brick_occupied: &mut [u32],
) {
    let idx = id.x;
    let bpa = push.bricks_per_axis;
    let total_bricks = bpa * bpa * bpa;

    if idx >= total_bricks {
        return;
    }

    // Decompose linear brick index into 3D brick coordinates
    let bx = idx % bpa;
    let by = (idx / bpa) % bpa;
    let bz = idx / (bpa * bpa);

    let occupied = scan_brick(bx, by, bz, push.brick_size, push.grid_size, voxels);
    brick_occupied[idx as usize] = if occupied { 1 } else { 0 };
}
