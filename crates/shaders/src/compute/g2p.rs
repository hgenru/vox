//! Grid-to-Particle (G2P) compute shader.
//!
//! Each thread processes one particle, gathering velocity and temperature
//! from the 27 neighboring grid cells using quadratic B-spline weights.
//! Updates particle velocity (PIC/FLIP blend), temperature, APIC C matrix,
//! deformation gradient F, position, and phase transitions.

use crate::types::{GridCell, Particle};
use spirv_std::glam::{UVec3, Vec3, Vec4};
use spirv_std::spirv;

use super::quadratic_bspline_weights;

/// Push constants for the G2P shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct G2pPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// PIC/FLIP blend ratio (0.0 = pure FLIP, 1.0 = pure PIC).
/// 0.7 PIC balances stability with fluid-like flow (less viscous damping).
const PIC_RATIO: f32 = 0.5;

/// Number of u32 values per voxel cell (must match voxelize.rs layout).
const U32S_PER_VOXEL: u32 = 4;

/// Check if a solid particle has support from a solid voxel below it.
///
/// Reads the voxel buffer (from the previous frame's voxelize pass) to see
/// if the cell directly below this particle is occupied by a solid (phase==0).
/// Particles at the grid floor (vy <= 1) are always considered supported.
pub fn check_solid_support(pos: spirv_std::glam::Vec3, voxels: &[u32], grid_size: u32) -> bool {
    let vx = pos.x as u32;
    let vy = pos.y as u32;
    let vz = pos.z as u32;

    // At bottom of grid = supported by boundary
    if vy <= 1 {
        return true;
    }

    // Check cell below
    let below_y = vy - 1;
    if vx >= grid_size || below_y >= grid_size || vz >= grid_size {
        return true; // out of bounds = treat as supported (safe default)
    }

    let below_idx = (vz * grid_size * grid_size + below_y * grid_size + vx) as usize;
    let base = below_idx * U32S_PER_VOXEL as usize;

    // Check occupied flag (offset +3) and that the occupant is solid (phase==0)
    let occupied = voxels[base + 3] > 0;
    if !occupied {
        return false;
    }

    // Unpack phase from packed_material at offset +0: (weight<<24 | mat<<16 | phase<<8 | 0xFF)
    let packed = voxels[base];
    let phase_below = (packed >> 8) & 0xFF;
    phase_below == 0 // supported only if the voxel below is also solid
}

/// Gather velocity from the grid and update one particle.
///
/// Implements the G2P transfer step:
/// 1. Gather weighted velocity from 27 neighbor cells (PIC velocity)
/// 2. Blend PIC and FLIP velocities
/// 3. Compute APIC affine momentum matrix C
/// 4. Update deformation gradient F
/// 5. Advect position
/// 6. Check phase transitions by temperature
///
/// For solid particles (phase==0), checks support from the voxel below.
/// Supported solids update only F/C (pinned). Unsupported solids fall normally.
pub fn gather_particle(
    particle: &mut Particle,
    grid: &[GridCell],
    voxels: &[u32],
    grid_size: u32,
    dt: f32,
) {
    let pos = particle.pos_mass.truncate();
    let old_vel = particle.vel_temp.truncate();

    // Base cell index
    let base_x = (pos.x - 0.5).max(0.0) as u32;
    let base_y = (pos.y - 0.5).max(0.0) as u32;
    let base_z = (pos.z - 0.5).max(0.0) as u32;

    let fx = pos.x - base_x as f32;
    let fy = pos.y - base_y as f32;
    let fz = pos.z - base_z as f32;

    let wx = quadratic_bspline_weights(fx);
    let wy = quadratic_bspline_weights(fy);
    let wz = quadratic_bspline_weights(fz);

    // Accumulate PIC velocity, APIC C matrix, and temperature
    let mut pic_vel = Vec3::ZERO;
    let mut c_col0 = Vec3::ZERO;
    let mut c_col1 = Vec3::ZERO;
    let mut c_col2 = Vec3::ZERO;
    let mut new_temp = 0.0_f32;

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
                    let w = wx[di as usize] * wy[dj as usize] * wz[dk as usize];
                    let idx = (ck * grid_size * grid_size + cj * grid_size + ci) as usize;

                    let cell_vel = grid[idx].velocity_mass.truncate();

                    pic_vel += cell_vel * w;

                    // APIC: C += w * v_i * (x_i - x_p)^T
                    // The outer product contributes to each column of C
                    let dx = Vec3::new(
                        ci as f32 - pos.x,
                        cj as f32 - pos.y,
                        ck as f32 - pos.z,
                    );

                    // C_col0 accumulates how velocity changes in x-direction
                    c_col0 += cell_vel * (w * dx.x);
                    c_col1 += cell_vel * (w * dx.y);
                    c_col2 += cell_vel * (w * dx.z);

                    // Gather temperature from grid (normalized by mass in grid_update).
                    // This enables natural thermal diffusion through the MPM transfer.
                    let grid_temp = grid[idx].temp_pad.x;
                    new_temp += grid_temp * w;
                }
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }

    // APIC scaling factor: 4 / (dx^2) where dx = 1.0 (grid spacing)
    let apic_scale = 4.0_f32;
    c_col0 *= apic_scale;
    c_col1 *= apic_scale;
    c_col2 *= apic_scale;

    // PIC/FLIP blend
    // True FLIP requires old grid velocities which we don't store.
    // Instead, use a weighted blend of PIC (grid velocity) and the particle's
    // old velocity to reduce excessive damping while maintaining stability.
    let mut new_vel = PIC_RATIO * pic_vel + (1.0 - PIC_RATIO) * old_vel;

    // Phase extraction for gas buoyancy below
    let phase = particle.ids.y;

    // Gas buoyancy: counteract most of gravity and add slight upward force
    // so steam rises and persists visually instead of falling immediately.
    if phase == 2 {
        // Counteract 70% of gravity (gravity is negative, so add positive Y)
        // and add a small buoyancy boost. Also damp horizontal velocity slightly
        // to keep steam from flying out of bounds too fast.
        new_vel.y += 196.0 * 0.7 * dt;
        new_vel.x *= 0.98;
        new_vel.z *= 0.98;
    }

    // Update deformation gradient: F_new = (I + dt * C) * F_old
    let f0 = particle.f_col0.truncate();
    let f1 = particle.f_col1.truncate();
    let f2 = particle.f_col2.truncate();

    // (I + dt*C) applied to each column of F
    let dt_c0 = c_col0 * dt;
    let dt_c1 = c_col1 * dt;
    let dt_c2 = c_col2 * dt;

    let new_f0 = Vec3::new(
        f0.x * (1.0 + dt_c0.x) + f0.y * dt_c1.x + f0.z * dt_c2.x,
        f0.x * dt_c0.y + f0.y * (1.0 + dt_c1.y) + f0.z * dt_c2.y,
        f0.x * dt_c0.z + f0.y * dt_c1.z + f0.z * (1.0 + dt_c2.z),
    );
    let new_f1 = Vec3::new(
        f1.x * (1.0 + dt_c0.x) + f1.y * dt_c1.x + f1.z * dt_c2.x,
        f1.x * dt_c0.y + f1.y * (1.0 + dt_c1.y) + f1.z * dt_c2.y,
        f1.x * dt_c0.z + f1.y * dt_c1.z + f1.z * (1.0 + dt_c2.z),
    );
    let new_f2 = Vec3::new(
        f2.x * (1.0 + dt_c0.x) + f2.y * dt_c1.x + f2.z * dt_c2.x,
        f2.x * dt_c0.y + f2.y * (1.0 + dt_c1.y) + f2.z * dt_c2.y,
        f2.x * dt_c0.z + f2.y * dt_c1.z + f2.z * (1.0 + dt_c2.z),
    );

    // Advect position
    let mut new_pos = pos + new_vel * dt;

    // Clamp position to grid bounds (with small margin)
    let margin = 1.5_f32;
    let upper = grid_size as f32 - margin;
    new_pos.x = new_pos.x.clamp(margin, upper);
    new_pos.y = new_pos.y.clamp(margin, upper);
    new_pos.z = new_pos.z.clamp(margin, upper);

    // Solids: check if the particle has support from a solid voxel below.
    // Supported solids update F and C only (pinned, prevents floor drift #45).
    // Unsupported solids fall through to full position/velocity update.
    if phase == 0 {
        let supported = check_solid_support(pos, voxels, grid_size);
        if supported {
            particle.f_col0 = new_f0.extend(0.0);
            particle.f_col1 = new_f1.extend(0.0);
            particle.f_col2 = new_f2.extend(0.0);
            particle.c_col0 = c_col0.extend(0.0);
            particle.c_col1 = c_col1.extend(0.0);
            particle.c_col2 = c_col2.extend(0.0);
            // Update temperature even for pinned solids (heat conducts through stone)
            particle.vel_temp = Vec4::new(
                particle.vel_temp.x, particle.vel_temp.y, particle.vel_temp.z, new_temp,
            );
            apply_phase_transitions(particle);
            return;
        }
        // Unsupported: fall through to position/velocity update below
    }

    // Write back to particle (liquids and gases only)
    particle.pos_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, particle.pos_mass.w);
    particle.vel_temp = Vec4::new(new_vel.x, new_vel.y, new_vel.z, new_temp);
    particle.f_col0 = new_f0.extend(0.0);
    particle.f_col1 = new_f1.extend(0.0);
    particle.f_col2 = new_f2.extend(0.0);
    particle.c_col0 = c_col0.extend(0.0);
    particle.c_col1 = c_col1.extend(0.0);
    particle.c_col2 = c_col2.extend(0.0);

    // Phase transitions by temperature
    apply_phase_transitions(particle);
}

/// Check and apply phase transitions based on temperature thresholds.
///
/// - Stone (mat=0) T > 1500 -> Liquid (lava, mat=2)
/// - Water (mat=1) T > 100 -> Gas (steam)
/// - Water (mat=1) T < 0 -> Solid (ice, mat=5)
/// - Lava (mat=2) T < 1500 -> Solid (stone, mat=0)
/// - Ice (mat=5) T > 0 -> Liquid (water, mat=1)
///
/// On phase change: reset F = Identity, as per CLAUDE.md trap #8.
pub fn apply_phase_transitions(particle: &mut Particle) {
    let material_id = particle.ids.x;
    let phase = particle.ids.y;
    let temp = particle.vel_temp.w;

    let mut new_phase = phase;
    let mut new_material = material_id;

    match material_id {
        // Stone
        0 => {
            if phase == 0 && temp > 1500.0 {
                new_phase = 1; // solid -> liquid (becomes lava)
                new_material = 2; // MAT_LAVA
            }
        }
        // Water
        1 => {
            if phase == 1 && temp > 100.0 {
                new_phase = 2; // liquid -> gas (steam)
            } else if phase == 1 && temp < 0.0 {
                new_phase = 0; // liquid -> solid (ice)
                new_material = 5; // MAT_ICE
            }
        }
        // Lava
        2 => {
            if phase == 1 && temp < 1500.0 {
                new_phase = 0; // liquid -> solid (becomes stone)
                new_material = 0; // MAT_STONE
            }
        }
        // Ice
        5 => {
            if phase == 0 && temp > 0.0 {
                new_phase = 1; // solid -> liquid (melts to water)
                new_material = 1; // MAT_WATER
            }
        }
        // Gunpowder
        6 => {
            if phase == 0 && temp > 150.0 {
                new_phase = 2; // solid -> gas (explosion)
                // Boost temp for massive EOS pressure
                particle.vel_temp = Vec4::new(
                    particle.vel_temp.x, particle.vel_temp.y, particle.vel_temp.z, 3000.0
                );
                // Reset F (trap #8)
                particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
                particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
                particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
            }
        }
        _ => {}
    }

    if new_phase != phase || new_material != material_id {
        particle.ids.x = new_material;
        particle.ids.y = new_phase;
        // Reset deformation gradient on phase change (CLAUDE.md trap #8)
        particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
}

/// Compute shader entry point: Grid-to-Particle transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read-write).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (read).
/// Descriptor set 0, binding 2: storage buffer of `u32` (voxel grid, 4 per cell, read).
/// Push constants: `G2pPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn g2p(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &G2pPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &[GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] voxels: &[u32],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    gather_particle(&mut particles[idx], grid, voxels, push.grid_size, push.dt);
}
