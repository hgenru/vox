//! Clear hash grid compute shader.
//!
//! Resets the hash grid keys array to EMPTY sentinel values and
//! zeroes the corresponding GridCell values. This is a bulk clear
//! over the entire capacity — still much cheaper than clearing a
//! dense 256^3 grid (1M slots vs 16M cells).
//!
//! Workgroup size: 64 threads, 1D dispatch.
//! Dispatch with `(ceil(capacity / 64), 1, 1)` workgroups.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

use super::hash_grid;

/// Push constants for the clear hash grid shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ClearHashGridPushConstants {
    /// Hash grid capacity (number of slots).
    pub hash_capacity: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
    /// Padding to 16-byte alignment.
    pub _pad2: u32,
}

/// Reset one hash grid slot: set key to EMPTY and zero the GridCell.
///
/// Helper function for trap #4a compliance (entry points must call helpers).
pub fn clear_hash_slot(keys: &mut [u32], values: &mut [GridCell], index: u32) {
    keys[index as usize] = hash_grid::HASH_GRID_EMPTY_KEY;
    values[index as usize].velocity_mass = Vec4::ZERO;
    values[index as usize].force_pad = Vec4::ZERO;
    values[index as usize].temp_pad = Vec4::ZERO;
}

/// Compute shader entry point: clear the entire hash grid.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (hash_keys, write).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (hash_values, write).
/// Push constants: `ClearHashGridPushConstants`.
#[spirv(compute(threads(64)))]
pub fn clear_hash_grid(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ClearHashGridPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] hash_keys: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hash_values: &mut [GridCell],
) {
    let idx = id.x;
    if idx >= push.hash_capacity {
        return;
    }
    clear_hash_slot(hash_keys, hash_values, idx);
}
