//! Prepare indirect dispatch arguments compute shader.
//!
//! Single-invocation shader that reads the active cell count and writes
//! the indirect dispatch arguments (group_x, group_y, group_z) for the
//! sparse grid passes (clear_grid_sparse, grid_update_sparse).

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Compute the indirect dispatch group counts from the active cell count.
///
/// Writes `(ceil(count / 64), 1, 1)` to the indirect args buffer.
/// Helper function to satisfy the entry-point-must-call-helper requirement (trap #4a).
pub fn write_indirect_args(active_count: &[u32], indirect_args: &mut [u32]) {
    let count = active_count[0];
    indirect_args[0] = (count + 63) / 64;
    indirect_args[1] = 1;
    indirect_args[2] = 1;
}

/// Compute shader entry point: prepare indirect dispatch arguments.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (active_count, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (indirect_args[3], write).
/// Dispatch with `(1, 1, 1)` workgroups.
#[spirv(compute(threads(1)))]
pub fn prepare_indirect(
    #[spirv(global_invocation_id)] _id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] active_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32],
) {
    write_indirect_args(active_count, indirect_args);
}
