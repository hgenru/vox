//! PB-MPM grid clear compute shader.
//!
//! Zeros all PB-MPM grid cells before each P2G pass. Each thread handles one
//! grid cell (8 u32s = 32 bytes). Must be dispatched before every P2G iteration
//! in the PB-MPM loop (trap #6: grid must be zeroed before P2G).
//!
//! Dispatch: `(ceil(grid_cell_count / 256), 1, 1)` workgroups.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::pbmpm_types::PBMPM_GRID_CELL_SIZE_U32;

/// Push constants shared by all PB-MPM compute passes.
///
/// 32 bytes, 16-byte aligned.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PbmpmPush {
    /// Grid dimension: cells per axis (e.g., 32, 64, or 128).
    pub grid_size: u32,
    /// Total number of particles in this zone.
    pub particle_count: u32,
    /// Offset into the global particle buffer (in particles, not bytes).
    pub particle_offset: u32,
    /// Offset into the global grid buffer (in cells, not bytes).
    pub grid_offset: u32,
    /// Simulation timestep for this iteration.
    pub dt: f32,
    /// Gravity acceleration (grid-units/s^2, typically negative).
    pub gravity: f32,
    /// Current PB-MPM iteration index (0..N).
    pub iteration: u32,
    /// Padding to 32-byte alignment.
    pub _pad: u32,
}

/// Zero out a single PB-MPM grid cell (8 u32s).
///
/// Helper function to satisfy trap #4a (entry points must call a helper).
fn zero_pbmpm_cell(grid: &mut [u32], base: u32) {
    let b = base as usize;
    grid[b] = 0;
    grid[b + 1] = 0;
    grid[b + 2] = 0;
    grid[b + 3] = 0;
    grid[b + 4] = 0;
    grid[b + 5] = 0;
    grid[b + 6] = 0;
    grid[b + 7] = 0;
}

/// Compute shader entry point: clear PB-MPM grid cells to zero.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (grid, read-write).
/// Push constants: `PbmpmPush`.
///
/// Each thread zeros one grid cell (8 u32s). Cells beyond the zone's total
/// grid cell count are skipped.
#[spirv(compute(threads(256)))]
pub fn pbmpm_clear_grid(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut [u32],
    #[spirv(push_constant)] push: &PbmpmPush,
) {
    let cell_idx = global_id.x;
    let total_cells = push.grid_size * push.grid_size * push.grid_size;
    if cell_idx >= total_cells {
        return;
    }

    let absolute_cell = push.grid_offset + cell_idx;
    let base = absolute_cell * PBMPM_GRID_CELL_SIZE_U32;
    zero_pbmpm_cell(grid, base);
}
