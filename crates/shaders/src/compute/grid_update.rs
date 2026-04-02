//! Grid update compute shader.
//!
//! Each thread processes one grid cell: converts accumulated momentum to velocity,
//! applies gravity, and enforces boundary conditions.
//! Workgroup size: (4, 4, 4) = 64 threads.

use crate::types::GridCell;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Push constants for the grid update shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GridUpdatePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y, e.g., -9.81).
    pub gravity: f32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// Update a single grid cell: momentum -> velocity, gravity, boundaries.
///
/// Separated into a helper function for the rust-gpu linker bug workaround
/// (see CLAUDE.md trap #4a).
pub fn update_cell(
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

    // Boundary conditions: zero velocity at grid edges
    // (sticky boundary — particles stop at walls)
    let boundary = 1_u32;
    let upper = grid_size - boundary - 1;
    if ix <= boundary || ix >= upper {
        vx = 0.0;
    }
    if iy <= boundary || iy >= upper {
        vy = 0.0;
    }
    if iz <= boundary || iz >= upper {
        vz = 0.0;
    }

    cell.velocity_mass = Vec4::new(vx, vy, vz, mass);
}

/// Compute shader entry point: grid velocity update.
///
/// Descriptor set 0, binding 0: storage buffer of `GridCell` (read-write).
/// Push constants: `GridUpdatePushConstants`.
/// Dispatch with `(GRID_SIZE/4, GRID_SIZE/4, GRID_SIZE/4)` workgroups.
#[spirv(compute(threads(4, 4, 4)))]
pub fn grid_update(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &GridUpdatePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut [GridCell],
) {
    let grid_size = push.grid_size;
    if id.x >= grid_size || id.y >= grid_size || id.z >= grid_size {
        return;
    }
    let index = (id.z * grid_size * grid_size + id.y * grid_size + id.x) as usize;
    update_cell(&mut grid[index], id.x, id.y, id.z, grid_size, push.dt, push.gravity);
}
