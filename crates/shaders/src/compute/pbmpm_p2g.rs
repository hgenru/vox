//! PB-MPM Particle-to-Grid (P2G) compute shader.
//!
//! Each thread processes one particle, scattering mass and momentum (with APIC
//! affine term) to the 3x3x3 neighborhood of grid cells using quadratic B-spline
//! weights. Uses `spirv_std::arch::atomic_f_add()` for concurrent float accumulation.
//!
//! The APIC affine momentum term (C * offset) preserves angular momentum across
//! grid transfers, producing cohesive fluid motion instead of pure PIC dissipation.
//!
//! PB-MPM iterates the entire P2G->grid_update->G2P loop multiple times per frame
//! for stability at any timestep.
//!
//! Dispatch: `(ceil(particle_count / 256), 1, 1)` workgroups.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use super::pbmpm_clear_grid::PbmpmPush;
use super::quadratic_bspline_weights;
use crate::pbmpm_types;

/// Number of f32 values per PB-MPM grid cell (32 bytes / 4 = 8 floats).
const FLOATS_PER_CELL: u32 = pbmpm_types::PBMPM_GRID_CELL_SIZE_U32;

/// Scatter one particle's mass and APIC momentum to the grid.
///
/// For each of the 27 neighboring grid cells:
/// 1. Compute quadratic B-spline weight
/// 2. Atomically add weighted mass
/// 3. Atomically add weighted momentum: mass * (velocity + C * offset)
///    where C is the APIC affine momentum matrix and offset is cell-particle distance.
///
/// Grid cells are indexed with `grid_offset` to support multiple zones
/// sharing the same grid buffer.
fn scatter_particle_to_grid(
    particles: &[u32],
    grid: &mut [f32],
    p_idx: u32,
    push: &PbmpmPush,
) {
    // Read particle position, velocity, and C matrix into locals (trap #21)
    let abs_idx = push.particle_offset + p_idx;
    let px = pbmpm_types::particle_pos_x(particles, abs_idx);
    let py = pbmpm_types::particle_pos_y(particles, abs_idx);
    let pz = pbmpm_types::particle_pos_z(particles, abs_idx);
    let mass = pbmpm_types::particle_mass(particles, abs_idx);
    let vx = pbmpm_types::particle_vel_x(particles, abs_idx);
    let vy = pbmpm_types::particle_vel_y(particles, abs_idx);
    let vz = pbmpm_types::particle_vel_z(particles, abs_idx);

    // Read APIC C matrix columns into locals (trap #21)
    let c_col0 = pbmpm_types::particle_c_col0(particles, abs_idx);
    let c_col1 = pbmpm_types::particle_c_col1(particles, abs_idx);
    let c_col2 = pbmpm_types::particle_c_col2(particles, abs_idx);

    // Base cell: floor(pos - 0.5) clamped to 0
    let base_x = (px - 0.5).max(0.0) as u32;
    let base_y = (py - 0.5).max(0.0) as u32;
    let base_z = (pz - 0.5).max(0.0) as u32;

    // Fractional position within base cell (for B-spline weights)
    let fx = px - base_x as f32;
    let fy = py - base_y as f32;
    let fz = pz - base_z as f32;

    let wx = quadratic_bspline_weights(fx);
    let wy = quadratic_bspline_weights(fy);
    let wz = quadratic_bspline_weights(fz);

    let gs = push.grid_size;

    // Scatter to 3x3x3 neighborhood
    let mut di = 0u32;
    while di < 3 {
        let mut dj = 0u32;
        while dj < 3 {
            scatter_inner_loop_apic(
                grid, di, dj, base_x, base_y, base_z,
                px, py, pz,
                &wx, &wy, &wz, gs, mass, vx, vy, vz,
                &c_col0, &c_col1, &c_col2,
                push.grid_offset,
            );
            dj += 1;
        }
        di += 1;
    }
}

/// Inner loop of the P2G APIC scatter: iterates over the Z axis for a given (di, dj).
///
/// Scatters momentum with APIC affine term: mom = mass * (vel + C * offset).
/// Separated to keep branch count per function low (trap #15).
#[allow(clippy::too_many_arguments)]
fn scatter_inner_loop_apic(
    grid: &mut [f32],
    di: u32,
    dj: u32,
    base_x: u32,
    base_y: u32,
    base_z: u32,
    px: f32,
    py: f32,
    pz: f32,
    wx: &[f32; 3],
    wy: &[f32; 3],
    wz: &[f32; 3],
    grid_size: u32,
    mass: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    c_col0: &[f32; 3],
    c_col1: &[f32; 3],
    c_col2: &[f32; 3],
    grid_offset: u32,
) {
    let mut dk = 0u32;
    while dk < 3 {
        let ci = base_x + di;
        let cj = base_y + dj;
        let ck = base_z + dk;

        if ci < grid_size && cj < grid_size && ck < grid_size {
            let w = wx[di as usize] * wy[dj as usize] * wz[dk as usize];
            let weighted_mass = w * mass;

            // APIC: offset from particle to cell center
            let ox = ci as f32 + 0.5 - px;
            let oy = cj as f32 + 0.5 - py;
            let oz = ck as f32 + 0.5 - pz;

            // C * offset (column-major: result = col0*ox + col1*oy + col2*oz)
            let cx = c_col0[0] * ox + c_col1[0] * oy + c_col2[0] * oz;
            let cy = c_col0[1] * ox + c_col1[1] * oy + c_col2[1] * oz;
            let cz = c_col0[2] * ox + c_col1[2] * oy + c_col2[2] * oz;

            // Momentum with APIC affine term: mass * (vel + C * offset)
            let mom_x = (vx + cx) * weighted_mass;
            let mom_y = (vy + cy) * weighted_mass;
            let mom_z = (vz + cz) * weighted_mass;

            // Flat cell index with offset for multi-zone support
            let cell_idx = grid_offset
                + pbmpm_types::grid_cell_index(ci, cj, ck, grid_size);
            let base = (cell_idx * FLOATS_PER_CELL) as usize;

            // Atomically accumulate momentum and mass
            // Grid layout per cell: [mom_x, mom_y, mom_z, mass, force_x, force_y, force_z, pad]
            unsafe {
                pbmpm_types::grid_atomic_add_f32(grid, base, mom_x);
                pbmpm_types::grid_atomic_add_f32(grid, base + 1, mom_y);
                pbmpm_types::grid_atomic_add_f32(grid, base + 2, mom_z);
                pbmpm_types::grid_atomic_add_f32(grid, base + 3, weighted_mass);
            }
        }
        dk += 1;
    }
}

/// Compute shader entry point: PB-MPM Particle-to-Grid transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (particles, read).
/// Descriptor set 0, binding 1: storage buffer of `f32` (grid, read-write with atomics).
///   The grid buffer is bound as `&mut [f32]` to enable float atomics.
///   Layout: 8 floats per cell `[mom_x, mom_y, mom_z, mass, force_x, force_y, force_z, pad]`.
/// Push constants: `PbmpmPush`.
#[spirv(compute(threads(256)))]
pub fn pbmpm_p2g(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &mut [f32],
    #[spirv(push_constant)] push: &PbmpmPush,
) {
    let p_idx = global_id.x;
    if p_idx >= push.particle_count {
        return;
    }

    scatter_particle_to_grid(particles, grid, p_idx, push);
}
