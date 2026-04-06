//! PB-MPM Grid-to-Particle (G2P) compute shader.
//!
//! Each thread processes one particle, gathering velocity from the 3x3x3
//! neighborhood of grid cells using quadratic B-spline weights, then
//! updating position and velocity.
//!
//! PB-MPM key differences from MLS-MPM:
//! - No APIC affine momentum term (pure PIC gather for POC)
//! - No stress/deformation gradient update (handled by constraint projection)
//! - Iterative: this runs 2-4 times per frame, refining the solution each time
//!
//! Dispatch: `(ceil(particle_count / 256), 1, 1)` workgroups.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use super::pbmpm_clear_grid::PbmpmPush;
use super::quadratic_bspline_weights;
use crate::pbmpm_types;

/// Gather velocity from grid and update one particle.
///
/// 1. Compute B-spline weights from particle position
/// 2. Accumulate weighted grid velocity from 3x3x3 neighborhood
/// 3. Update position: pos += new_vel * dt
/// 4. Clamp position to grid bounds (with margin)
///
/// For this POC, uses pure PIC (no APIC affine term). Material-specific
/// constraint projection will be added in a later PR.
fn gather_and_update_particle(
    particles: &mut [u32],
    grid: &[u32],
    p_idx: u32,
    push: &PbmpmPush,
) {
    let abs_idx = push.particle_offset + p_idx;

    // Read particle position into locals (trap #21)
    let px = pbmpm_types::particle_pos_x(particles, abs_idx);
    let py = pbmpm_types::particle_pos_y(particles, abs_idx);
    let pz = pbmpm_types::particle_pos_z(particles, abs_idx);

    // Base cell: floor(pos - 0.5) clamped to 0
    let base_x = (px - 0.5).max(0.0) as u32;
    let base_y = (py - 0.5).max(0.0) as u32;
    let base_z = (pz - 0.5).max(0.0) as u32;

    let fx = px - base_x as f32;
    let fy = py - base_y as f32;
    let fz = pz - base_z as f32;

    let wx = quadratic_bspline_weights(fx);
    let wy = quadratic_bspline_weights(fy);
    let wz = quadratic_bspline_weights(fz);

    // Gather weighted velocity from 3x3x3 neighborhood
    let mut new_vx = 0.0_f32;
    let mut new_vy = 0.0_f32;
    let mut new_vz = 0.0_f32;

    let mut di = 0u32;
    while di < 3 {
        let mut dj = 0u32;
        while dj < 3 {
            let result = gather_inner_loop(
                grid, di, dj, base_x, base_y, base_z,
                &wx, &wy, &wz, push.grid_size, push.grid_offset,
            );
            new_vx += result[0];
            new_vy += result[1];
            new_vz += result[2];
            dj += 1;
        }
        di += 1;
    }

    // Advect position
    let mut new_px = px + new_vx * push.dt;
    let mut new_py = py + new_vy * push.dt;
    let mut new_pz = pz + new_vz * push.dt;

    // Clamp position to grid bounds (with margin to prevent boundary issues)
    let margin = 1.5_f32;
    let upper = push.grid_size as f32 - margin;
    new_px = clamp_f32(new_px, margin, upper);
    new_py = clamp_f32(new_py, margin, upper);
    new_pz = clamp_f32(new_pz, margin, upper);

    // Write back position and velocity
    pbmpm_types::set_particle_pos(particles, abs_idx, new_px, new_py, new_pz);
    pbmpm_types::set_particle_vel(particles, abs_idx, new_vx, new_vy, new_vz);
}

/// Inner loop of the G2P gather: iterates over Z axis for a given (di, dj).
///
/// Returns [accumulated_vx, accumulated_vy, accumulated_vz] for this row.
/// Separated to keep branch count per function low (trap #15).
fn gather_inner_loop(
    grid: &[u32],
    di: u32,
    dj: u32,
    base_x: u32,
    base_y: u32,
    base_z: u32,
    wx: &[f32; 3],
    wy: &[f32; 3],
    wz: &[f32; 3],
    grid_size: u32,
    grid_offset: u32,
) -> [f32; 3] {
    let mut acc_vx = 0.0_f32;
    let mut acc_vy = 0.0_f32;
    let mut acc_vz = 0.0_f32;

    let mut dk = 0u32;
    while dk < 3 {
        let ci = base_x + di;
        let cj = base_y + dj;
        let ck = base_z + dk;

        if ci < grid_size && cj < grid_size && ck < grid_size {
            let w = wx[di as usize] * wy[dj as usize] * wz[dk as usize];

            let cell_idx = grid_offset
                + pbmpm_types::grid_cell_index(ci, cj, ck, grid_size);

            // Read grid velocity (already normalized by mass in grid_update)
            let gvx = pbmpm_types::grid_velocity_x(grid, cell_idx);
            let gvy = pbmpm_types::grid_velocity_y(grid, cell_idx);
            let gvz = pbmpm_types::grid_velocity_z(grid, cell_idx);

            acc_vx += w * gvx;
            acc_vy += w * gvy;
            acc_vz += w * gvz;
        }
        dk += 1;
    }

    [acc_vx, acc_vy, acc_vz]
}

/// Clamp a f32 value to [min, max].
///
/// Helper to avoid `f32::clamp` which may not be available in no_std SPIR-V.
fn clamp_f32(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Compute shader entry point: PB-MPM Grid-to-Particle transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (particles, read-write).
/// Descriptor set 0, binding 1: storage buffer of `u32` (grid, read).
/// Push constants: `PbmpmPush`.
#[spirv(compute(threads(256)))]
pub fn pbmpm_g2p(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &[u32],
    #[spirv(push_constant)] push: &PbmpmPush,
) {
    let p_idx = global_id.x;
    if p_idx >= push.particle_count {
        return;
    }

    gather_and_update_particle(particles, grid, p_idx, push);
}
