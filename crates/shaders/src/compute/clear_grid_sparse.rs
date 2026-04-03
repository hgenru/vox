//! Sparse clear grid compute shader.
//!
//! Zeros only the active grid cells (identified by the mark_active pass)
//! instead of clearing the entire grid. Also resets the mark buffer entries
//! for those cells, preparing them for the next frame's mark_active pass.
//!
//! Dispatched indirectly using the output of prepare_indirect.
//! Workgroup size: 64 threads, 1D dispatch.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Zero out a single grid cell's fields.
///
/// Separated into a helper function for the rust-gpu linker bug workaround
/// (see CLAUDE.md trap #4a).
pub fn zero_active_cell(grid: &mut [GridCell], mark: &mut [u32], cell_idx: u32) {
    let i = cell_idx as usize;
    grid[i].velocity_mass = Vec4::ZERO;
    grid[i].force_pad = Vec4::ZERO;
    grid[i].temp_pad = Vec4::ZERO;
    mark[i] = 0;
}

/// Compute shader entry point: sparse grid clear.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (active_cells list, read).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (grid, write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (mark buffer, write).
/// Descriptor set 0, binding 3: storage buffer of `u32` (active_count, read).
/// Dispatched indirectly via `vkCmdDispatchIndirect`.
#[spirv(compute(threads(64)))]
pub fn clear_grid_sparse(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] active_cells: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &mut [GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mark: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_count: &[u32],
) {
    let idx = id.x;
    let count = active_count[0];
    if idx >= count {
        return;
    }
    let cell_idx = active_cells[idx as usize];
    zero_active_cell(grid, mark, cell_idx);
}
