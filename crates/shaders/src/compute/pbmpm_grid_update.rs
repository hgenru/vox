//! PB-MPM grid update compute shader.
//!
//! After P2G scatter: normalize momentum to velocity (divide by mass),
//! apply gravity, and enforce boundary conditions at grid edges.
//!
//! Each thread processes one grid cell.
//! Dispatch: `(ceil(grid_cell_count / 256), 1, 1)` workgroups.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use super::pbmpm_clear_grid::PbmpmPush;
use crate::pbmpm_types;

/// Minimum mass threshold below which a grid cell is considered empty.
///
/// Cells with mass below this threshold are left at zero velocity.
const MIN_MASS: f32 = 1.0e-6;

/// Normalize momentum to velocity, apply gravity, and enforce boundary conditions.
///
/// Helper function to satisfy trap #4a (entry points must call a helper).
fn update_grid_cell(
    grid: &mut [u32],
    cell_idx: u32,
    cx: u32,
    cy: u32,
    cz: u32,
    push: &PbmpmPush,
) {
    let mass = pbmpm_types::grid_mass(grid, cell_idx);

    if mass <= MIN_MASS {
        return;
    }

    // Normalize momentum to velocity
    let inv_mass = 1.0 / mass;
    let mut vx = pbmpm_types::grid_momentum_x(grid, cell_idx) * inv_mass;
    let mut vy = pbmpm_types::grid_momentum_y(grid, cell_idx) * inv_mass;
    let mut vz = pbmpm_types::grid_momentum_z(grid, cell_idx) * inv_mass;

    // Apply gravity (Y-axis only)
    vy += push.gravity * push.dt;

    // Boundary conditions: zero velocity at grid edges.
    // Use a 2-cell margin to prevent particles from getting stuck.
    let margin = 2u32;
    let upper = push.grid_size - margin;

    vx = apply_boundary_x(vx, cx, margin, upper);
    vy = apply_boundary_y(vy, cy, margin, upper);
    vz = apply_boundary_z(vz, cz, margin, upper);

    // Write normalized velocity back (mass preserved for G2P reference)
    pbmpm_types::set_grid_velocity_mass(grid, cell_idx, vx, vy, vz, mass);
}

/// Apply boundary condition on X axis: zero X velocity at edges.
///
/// Separated to keep branch count per function low (trap #15).
fn apply_boundary_x(vx: f32, cx: u32, margin: u32, upper: u32) -> f32 {
    if cx < margin || cx >= upper {
        return 0.0;
    }
    vx
}

/// Apply boundary condition on Y axis: zero Y velocity at edges.
fn apply_boundary_y(vy: f32, cy: u32, margin: u32, upper: u32) -> f32 {
    if cy < margin || cy >= upper {
        return 0.0;
    }
    vy
}

/// Apply boundary condition on Z axis: zero Z velocity at edges.
fn apply_boundary_z(vz: f32, cz: u32, margin: u32, upper: u32) -> f32 {
    if cz < margin || cz >= upper {
        return 0.0;
    }
    vz
}

/// Compute shader entry point: PB-MPM grid update.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (grid, read-write).
/// Push constants: `PbmpmPush`.
#[spirv(compute(threads(256)))]
pub fn pbmpm_grid_update(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut [u32],
    #[spirv(push_constant)] push: &PbmpmPush,
) {
    let local_idx = global_id.x;
    let total_cells = push.grid_size * push.grid_size * push.grid_size;
    if local_idx >= total_cells {
        return;
    }

    // Decompose flat index to 3D coordinates for boundary checks
    let gs = push.grid_size;
    let cz = local_idx / (gs * gs);
    let cy = (local_idx / gs) % gs;
    let cx = local_idx % gs;

    let absolute_cell = push.grid_offset + local_idx;
    update_grid_cell(grid, absolute_cell, cx, cy, cz, push);
}
