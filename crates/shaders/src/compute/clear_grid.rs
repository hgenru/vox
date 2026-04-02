//! Clear grid compute shader.
//!
//! Zeros all `GridCell` fields before the P2G pass.
//! Each thread handles one grid cell. Workgroup size: (4, 4, 4) = 64 threads.
//! For a 32^3 grid, dispatch (8, 8, 8) workgroups.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Zero out a single grid cell's fields.
///
/// Separated into a helper function to work around the rust-gpu linker bug
/// that drops entry points with no function calls (see CLAUDE.md trap #4a).
pub fn zero_cell(cell: &mut GridCell) {
    cell.velocity_mass = Vec4::ZERO;
    cell.force_pad = Vec4::ZERO;
}

/// Compute shader entry point: clears all grid cells to zero.
///
/// Descriptor set 0, binding 0: storage buffer containing `GridCell` array.
/// Dispatch with `(GRID_SIZE/4, GRID_SIZE/4, GRID_SIZE/4)` workgroups.
#[spirv(compute(threads(4, 4, 4)))]
pub fn clear_grid(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut [GridCell],
) {
    let grid_size = crate::types::GRID_SIZE;
    if id.x >= grid_size || id.y >= grid_size || id.z >= grid_size {
        return;
    }
    let index = (id.z * grid_size * grid_size + id.y * grid_size + id.x) as usize;
    zero_cell(&mut grid[index]);
}
