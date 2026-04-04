//! Sparse Grid-to-Particle (G2P) compute shader using hash grid.
//!
//! Same physics as `g2p.rs` but reads from a hash grid instead of a dense flat array.
//! Each thread processes one particle, gathering velocity and temperature
//! from hash grid slots via `hash_grid_lookup`.
//!
//! If a lookup returns EMPTY, the cell is treated as zero (no contribution).

use crate::types::{GridCell, Particle, PhaseTransitionRule, MAX_PHASE_RULES};
use spirv_std::glam::{UVec3, Vec3, Vec4};
use spirv_std::spirv;

use super::g2p;
use super::hash_grid;
use super::quadratic_bspline_weights;

/// Push constants for the sparse G2P shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct G2pSparsePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Number of valid phase transition rules.
    pub num_rules: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
    /// Hash grid capacity (number of slots, must be power of 2).
    pub hash_capacity: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
}

/// Read a GridCell from hash_values at the given slot index.
///
/// Converts flat f32 values back into a GridCell struct.
/// Helper function extracted for trap #4a compliance.
fn read_cell_from_hash(hash_values: &[GridCell], slot: u32) -> (Vec4, Vec4, Vec4) {
    let cell = hash_values[slot as usize];
    (cell.velocity_mass, cell.force_pad, cell.temp_pad)
}

/// Gather velocity from the hash grid and update one particle.
///
/// Same physics as `g2p::gather_particle` but uses hash grid lookups.
/// Cells not found in the hash grid contribute zero (empty space).
pub fn gather_particle_sparse(
    particle: &mut Particle,
    hash_keys: &[u32],
    hash_values: &[GridCell],
    voxels: &[u32],
    reactions: &[PhaseTransitionRule],
    grid_size: u32,
    dt: f32,
    hash_capacity: u32,
    num_rules: u32,
) {
    let pos = particle.pos_mass.truncate();
    let old_vel = particle.vel_temp.truncate();

    let base_x = (pos.x - 0.5).max(0.0) as u32;
    let base_y = (pos.y - 0.5).max(0.0) as u32;
    let base_z = (pos.z - 0.5).max(0.0) as u32;

    let fx = pos.x - base_x as f32;
    let fy = pos.y - base_y as f32;
    let fz = pos.z - base_z as f32;

    let wx = quadratic_bspline_weights(fx);
    let wy = quadratic_bspline_weights(fy);
    let wz = quadratic_bspline_weights(fz);

    let mut pic_vel = Vec3::ZERO;
    let mut c_col0 = Vec3::ZERO;
    let mut c_col1 = Vec3::ZERO;
    let mut c_col2 = Vec3::ZERO;
    let mut new_temp = 0.0_f32;

    // Gather from 3x3x3 neighborhood
    let mut di = 0u32;
    while di < 3 {
        let mut dj = 0u32;
        while dj < 3 {
            let mut dk = 0u32;
            while dk < 3 {
                let ci = base_x + di;
                let cj = base_y + dj;
                let ck = base_z + dk;

                if ci < grid_size && cj < grid_size && ck < grid_size {
                    let key = hash_grid::pack_key(ci, cj, ck);
                    let slot = hash_grid::hash_grid_lookup(hash_keys, key, hash_capacity);

                    // If not found in hash grid, treat as zero cell
                    if slot != hash_grid::HASH_GRID_EMPTY_KEY {
                        let w = wx[di as usize] * wy[dj as usize] * wz[dk as usize];
                        let (vel_mass, _force_pad, temp_pad) = read_cell_from_hash(hash_values, slot);
                        let cell_vel = vel_mass.truncate();

                        pic_vel += cell_vel * w;

                        let dx = Vec3::new(
                            ci as f32 - pos.x,
                            cj as f32 - pos.y,
                            ck as f32 - pos.z,
                        );

                        c_col0 += cell_vel * (w * dx.x);
                        c_col1 += cell_vel * (w * dx.y);
                        c_col2 += cell_vel * (w * dx.z);

                        let grid_temp = temp_pad.x;
                        new_temp += grid_temp * w;
                    }
                }
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }

    // APIC scaling factor
    let apic_scale = 4.0_f32;
    c_col0 *= apic_scale;
    c_col1 *= apic_scale;
    c_col2 *= apic_scale;

    // Thermal diffusion blend
    let thermal_blend = 0.002_f32;
    let old_temp = particle.vel_temp.w;
    new_temp = old_temp + thermal_blend * (new_temp - old_temp);

    // Radiative cooling
    new_temp = apply_radiative_cooling_sparse(new_temp);

    // PIC/FLIP blend (PIC_RATIO = 0.5)
    let pic_ratio = 0.5_f32;
    let mut new_vel = pic_ratio * pic_vel + (1.0 - pic_ratio) * old_vel;

    let phase = particle.ids.y;

    // Gas buoyancy
    if phase == 2 {
        new_vel.y += 196.0 * 0.7 * dt;
        new_vel.x *= 0.98;
        new_vel.z *= 0.98;
    }

    // Update deformation gradient
    let f0 = particle.f_col0.truncate();
    let f1 = particle.f_col1.truncate();
    let f2 = particle.f_col2.truncate();

    let dt_c0 = c_col0 * dt;
    let dt_c1 = c_col1 * dt;
    let dt_c2 = c_col2 * dt;

    let new_f0 = compute_new_f_col(f0, dt_c0, dt_c1, dt_c2);
    let new_f1 = compute_new_f_col(f1, dt_c0, dt_c1, dt_c2);
    let new_f2 = compute_new_f_col(f2, dt_c0, dt_c1, dt_c2);

    // Advect position
    let mut new_pos = pos + new_vel * dt;

    let margin = 1.5_f32;
    let upper = grid_size as f32 - margin;
    new_pos.x = new_pos.x.clamp(margin, upper);
    new_pos.y = new_pos.y.clamp(margin, upper);
    new_pos.z = new_pos.z.clamp(margin, upper);

    // Solid support check and update
    if phase == 0 {
        let supported = g2p::check_solid_support(pos, voxels, grid_size);
        if supported {
            write_pinned_solid(particle, new_f0, new_f1, new_f2, c_col0, c_col1, c_col2, new_temp, reactions, num_rules);
            return;
        }
        // Unsupported solid: direct gravity accumulation
        let old_vel_y = particle.vel_temp.y;
        let gravity = -196.0_f32;
        new_vel = Vec3::new(
            old_vel.x * 0.9,
            old_vel_y + gravity * dt,
            old_vel.z * 0.9,
        );
        if new_vel.y < -50.0 {
            new_vel.y = -50.0;
        }
    }

    // Write back to particle
    particle.pos_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, particle.pos_mass.w);
    particle.vel_temp = Vec4::new(new_vel.x, new_vel.y, new_vel.z, new_temp);
    particle.f_col0 = new_f0.extend(0.0);
    particle.f_col1 = new_f1.extend(0.0);
    particle.f_col2 = new_f2.extend(0.0);
    particle.c_col0 = c_col0.extend(0.0);
    particle.c_col1 = c_col1.extend(0.0);
    particle.c_col2 = c_col2.extend(0.0);

    g2p::apply_phase_transitions(particle, reactions, num_rules);
}

/// Compute updated deformation gradient column: (I + dt*C) * F_col.
///
/// Helper function to avoid code duplication for each F column.
fn compute_new_f_col(f: Vec3, dt_c0: Vec3, dt_c1: Vec3, dt_c2: Vec3) -> Vec3 {
    Vec3::new(
        f.x * (1.0 + dt_c0.x) + f.y * dt_c1.x + f.z * dt_c2.x,
        f.x * dt_c0.y + f.y * (1.0 + dt_c1.y) + f.z * dt_c2.y,
        f.x * dt_c0.z + f.y * dt_c1.z + f.z * (1.0 + dt_c2.z),
    )
}

/// Write back pinned solid particle (supported by voxel below).
///
/// Updates F, C, and temperature only. Position and velocity are frozen.
/// Extracted as helper for trap #15 (branch count).
fn write_pinned_solid(
    particle: &mut Particle,
    new_f0: Vec3, new_f1: Vec3, new_f2: Vec3,
    c_col0: Vec3, c_col1: Vec3, c_col2: Vec3,
    new_temp: f32,
    reactions: &[PhaseTransitionRule],
    num_rules: u32,
) {
    particle.f_col0 = new_f0.extend(0.0);
    particle.f_col1 = new_f1.extend(0.0);
    particle.f_col2 = new_f2.extend(0.0);
    particle.c_col0 = c_col0.extend(0.0);
    particle.c_col1 = c_col1.extend(0.0);
    particle.c_col2 = c_col2.extend(0.0);
    particle.vel_temp = Vec4::new(
        particle.vel_temp.x, particle.vel_temp.y, particle.vel_temp.z, new_temp,
    );
    g2p::apply_phase_transitions(particle, reactions, num_rules);
}

/// Radiative cooling for sparse G2P (identical to g2p.rs version).
///
/// Extracted as a separate function for trap #4a compliance.
fn apply_radiative_cooling_sparse(temp: f32) -> f32 {
    let ambient_temp = 20.0_f32;
    let cooling_rate = 0.0005_f32;
    temp + cooling_rate * (ambient_temp - temp)
}

/// Check if a position is near the domain boundary and flag it.
///
/// Sets `oob_flag[0]` to 1 via atomic max if the particle position is within
/// `margin` of the grid edge. This allows the CPU to detect when particles
/// are being clamped and trigger chunk streaming.
fn check_and_flag_oob_sparse(pos: Vec3, grid_size: u32, oob_flag: &mut [u32]) {
    let margin = 2.0_f32;
    let upper = grid_size as f32 - margin;
    if pos.x < margin || pos.x > upper || pos.y < margin || pos.y > upper || pos.z < margin || pos.z > upper {
        unsafe {
            spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut oob_flag[0], 1);
        }
    }
}

/// Compute shader entry point: sparse Grid-to-Particle transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read-write).
/// Descriptor set 0, binding 1: storage buffer of `u32` (hash_keys, read).
/// Descriptor set 0, binding 2: storage buffer of `GridCell` (hash_values, read).
/// Descriptor set 0, binding 3: storage buffer of `u32` (voxels, read).
/// Descriptor set 0, binding 4: storage buffer of `PhaseTransitionRule` (read).
/// Descriptor set 0, binding 5: storage buffer of `u32` (sleep_state, read).
/// Descriptor set 0, binding 6: storage buffer of `u32` (oob_flag, write).
/// Push constants: `G2pSparsePushConstants`.
#[spirv(compute(threads(64)))]
pub fn g2p_sparse(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &G2pSparsePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hash_keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hash_values: &[GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] voxels: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] reactions: &[PhaseTransitionRule],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sleep_state: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] oob_flag: &mut [u32],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }

    // NOTE: In sparse mode, the hash grid itself is the sparsity optimization.
    // Sleep-based brick skipping is NOT used here — see p2g_sparse.rs for details.
    // The sleep_state binding is kept to avoid descriptor set layout changes.
    let _sleep_state = sleep_state;

    gather_particle_sparse(
        &mut particles[idx],
        hash_keys,
        hash_values,
        voxels,
        reactions,
        push.grid_size,
        push.dt,
        push.hash_capacity,
        push.num_rules,
    );

    // Check OOB after position update
    let new_pos = particles[idx].pos_mass.truncate();
    check_and_flag_oob_sparse(new_pos, push.grid_size, oob_flag);
}
