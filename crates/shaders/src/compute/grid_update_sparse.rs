//! Sparse grid update compute shader.
//!
//! Processes only active grid cells (identified by the mark_active pass)
//! instead of iterating the entire grid. Each thread processes one active cell:
//! converts accumulated momentum to velocity, applies gravity, and enforces
//! boundary conditions.
//!
//! Dispatched indirectly using the output of prepare_indirect.
//! Workgroup size: 64 threads, 1D dispatch.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Push constants for the sparse grid update shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GridUpdateSparsePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y, e.g., -196.0).
    pub gravity: f32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// Apply boundary damping for cells at grid edges.
///
/// Returns the (possibly clamped) velocity components.
/// Helper to keep the main update logic concise (trap #15).
pub fn apply_boundary(
    vx: f32,
    vy: f32,
    vz: f32,
    ix: u32,
    iy: u32,
    iz: u32,
    grid_size: u32,
) -> (f32, f32, f32) {
    let boundary = 1_u32;
    let upper = grid_size - boundary - 1;
    let mut out_vx = vx;
    let mut out_vy = vy;
    let mut out_vz = vz;
    if ix <= boundary || ix >= upper {
        out_vx = 0.0;
    }
    if iy <= boundary || iy >= upper {
        out_vy = 0.0;
    }
    if iz <= boundary || iz >= upper {
        out_vz = 0.0;
    }
    (out_vx, out_vy, out_vz)
}

/// Apply solid boundary damping based on the solid flag accumulated by P2G.
///
/// Deep inside solid (flag > 6.0): fully zero velocity.
/// At boundary (flag > 2.0): mild damp — fluid can slide along surfaces.
/// Helper to keep the main update logic concise (trap #15).
pub fn apply_solid_damping(vx: f32, vy: f32, vz: f32, solid_flag: f32) -> (f32, f32, f32) {
    if solid_flag > 6.0 {
        return (0.0, 0.0, 0.0);
    }
    if solid_flag > 2.0 {
        let damp = 1.0 - ((solid_flag - 2.0) / 8.0).min(0.7);
        return (vx * damp, vy * damp, vz * damp);
    }
    (vx, vy, vz)
}

/// Update a single active grid cell: momentum -> velocity, gravity, boundaries.
///
/// Contains the same physics logic as `grid_update::update_cell` but takes a
/// flat cell index and grid_size for index decomposition.
/// Separated into a helper function for the rust-gpu linker bug workaround (trap #4a).
pub fn update_active_cell(
    cell: &mut GridCell,
    ix: u32,
    iy: u32,
    iz: u32,
    grid_size: u32,
    dt: f32,
    gravity: f32,
) {
    let mass = cell.velocity_mass.w;

    // Skip empty cells (mass below threshold)
    if mass < 1.0e-6 {
        cell.velocity_mass = Vec4::ZERO;
        cell.temp_pad = Vec4::ZERO;
        return;
    }

    // Convert momentum to velocity: v = momentum / mass
    let mut vx = cell.velocity_mass.x / mass;
    let mut vy = cell.velocity_mass.y / mass;
    let mut vz = cell.velocity_mass.z / mass;

    // Add force contribution: v += (force / mass) * dt
    vx += (cell.force_pad.x / mass) * dt;
    vy += (cell.force_pad.y / mass) * dt;
    vz += (cell.force_pad.z / mass) * dt;

    // Apply gravity
    vy += gravity * dt;

    // Apply boundary conditions
    let (bx, by, bz) = apply_boundary(vx, vy, vz, ix, iy, iz, grid_size);
    vx = bx;
    vy = by;
    vz = bz;

    // Apply solid damping
    let solid_flag = cell.force_pad.w;
    let (sx, sy, sz) = apply_solid_damping(vx, vy, vz, solid_flag);
    vx = sx;
    vy = sy;
    vz = sz;

    cell.velocity_mass = Vec4::new(vx, vy, vz, mass);

    // Normalize accumulated temperature by mass
    let cell_temp = cell.temp_pad.x / mass;
    cell.temp_pad = Vec4::new(cell_temp, 0.0, 0.0, 0.0);
}

/// Compute shader entry point: sparse grid velocity update.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (active_cells, read).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (grid, read-write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (active_count, read).
/// Push constants: `GridUpdateSparsePushConstants`.
/// Dispatched indirectly via `vkCmdDispatchIndirect`.
#[spirv(compute(threads(64)))]
pub fn grid_update_sparse(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &GridUpdateSparsePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] active_cells: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &mut [GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_count: &[u32],
) {
    let idx = id.x;
    let count = active_count[0];
    if idx >= count {
        return;
    }
    let cell_idx = active_cells[idx as usize] as usize;

    // Convert flat index back to 3D for boundary checks.
    // Layout is ZYX (cell = Z*G² + Y*G + X), matching P2G indexing.
    let gz = push.grid_size;
    let iz = (cell_idx as u32) / (gz * gz);        // Z
    let iy = ((cell_idx as u32) / gz) % gz;         // Y
    let ix = (cell_idx as u32) % gz;                // X

    update_active_cell(&mut grid[cell_idx], ix, iy, iz, gz, push.dt, push.gravity);
}
