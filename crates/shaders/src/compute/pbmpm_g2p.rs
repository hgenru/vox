//! PB-MPM Grid-to-Particle (G2P) compute shader.
//!
//! Each thread processes one particle, gathering velocity and APIC affine
//! momentum matrix C from the 3x3x3 neighborhood of grid cells using
//! quadratic B-spline weights, then updating position and velocity.
//!
//! Uses APIC (Affine Particle-In-Cell) transfer to preserve angular momentum
//! and produce cohesive, physically correct fluid motion. The C matrix captures
//! the local velocity gradient which is fed back into P2G for the next iteration.
//!
//! PB-MPM iterates the entire P2G->grid_update->G2P loop 2-4 times per frame
//! for stability at any timestep.
//!
//! Dispatch: `(ceil(particle_count / 256), 1, 1)` workgroups.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use super::pbmpm_clear_grid::PbmpmPush;
use super::quadratic_bspline_weights;
use crate::pbmpm_types;

/// Gather velocity and APIC C matrix from grid, update one particle.
///
/// 1. Compute B-spline weights from particle position
/// 2. Accumulate weighted grid velocity and APIC outer products from 3x3x3 neighborhood
/// 3. Update position: pos += new_vel * dt
/// 4. Clamp position to grid bounds (with margin)
/// 5. Store new velocity and C matrix for next P2G pass
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

    // Gather weighted velocity and APIC C matrix from 3x3x3 neighborhood
    // result: [vx, vy, vz, c00, c10, c20, c01, c11, c21, c02, c12, c22]
    let mut new_vx = 0.0_f32;
    let mut new_vy = 0.0_f32;
    let mut new_vz = 0.0_f32;
    // C matrix (column-major): 9 elements
    let mut c00 = 0.0_f32;
    let mut c10 = 0.0_f32;
    let mut c20 = 0.0_f32;
    let mut c01 = 0.0_f32;
    let mut c11 = 0.0_f32;
    let mut c21 = 0.0_f32;
    let mut c02 = 0.0_f32;
    let mut c12 = 0.0_f32;
    let mut c22 = 0.0_f32;

    let mut di = 0u32;
    while di < 3 {
        let mut dj = 0u32;
        while dj < 3 {
            let result = gather_inner_loop_apic(
                grid, di, dj, base_x, base_y, base_z, px, py, pz,
                &wx, &wy, &wz, push.grid_size, push.grid_offset,
            );
            new_vx += result[0];
            new_vy += result[1];
            new_vz += result[2];
            c00 += result[3];
            c10 += result[4];
            c20 += result[5];
            c01 += result[6];
            c11 += result[7];
            c21 += result[8];
            c02 += result[9];
            c12 += result[10];
            c22 += result[11];
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

    // Write back position, velocity, and APIC C matrix
    pbmpm_types::set_particle_pos(particles, abs_idx, new_px, new_py, new_pz);
    pbmpm_types::set_particle_vel(particles, abs_idx, new_vx, new_vy, new_vz);
    pbmpm_types::set_particle_c_matrix(
        particles,
        abs_idx,
        &[c00, c10, c20, c01, c11, c21, c02, c12, c22],
    );
}

/// Inner loop of the G2P APIC gather: iterates over Z axis for a given (di, dj).
///
/// Returns [vx, vy, vz, c00, c10, c20, c01, c11, c21, c02, c12, c22] for this row.
/// The C matrix terms are the weighted outer product: 4 * w * outer(grid_vel, offset).
/// The factor 4 comes from 4/dx^2 where dx=1 in grid units (standard APIC formulation).
///
/// Separated to keep branch count per function low (trap #15).
fn gather_inner_loop_apic(
    grid: &[u32],
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
    grid_offset: u32,
) -> [f32; 12] {
    let mut acc = [0.0_f32; 12];

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

            // Velocity gather
            acc[0] += w * gvx;
            acc[1] += w * gvy;
            acc[2] += w * gvz;

            // APIC: offset from particle to cell center
            let ox = ci as f32 + 0.5 - px;
            let oy = cj as f32 + 0.5 - py;
            let oz = ck as f32 + 0.5 - pz;

            // C += 4 * w * outer(grid_vel, offset)  (4/dx^2 where dx=1)
            let w4 = 4.0 * w;
            // Column 0 of C: grid_vel * ox
            acc[3] += w4 * gvx * ox;
            acc[4] += w4 * gvy * ox;
            acc[5] += w4 * gvz * ox;
            // Column 1 of C: grid_vel * oy
            acc[6] += w4 * gvx * oy;
            acc[7] += w4 * gvy * oy;
            acc[8] += w4 * gvz * oy;
            // Column 2 of C: grid_vel * oz
            acc[9] += w4 * gvx * oz;
            acc[10] += w4 * gvy * oz;
            acc[11] += w4 * gvz * oz;
        }
        dk += 1;
    }

    acc
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
