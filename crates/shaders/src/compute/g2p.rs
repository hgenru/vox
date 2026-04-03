//! Grid-to-Particle (G2P) compute shader.
//!
//! Each thread processes one particle, gathering velocity and temperature
//! from the 27 neighboring grid cells using quadratic B-spline weights.
//! Updates particle velocity (PIC/FLIP blend), temperature, APIC C matrix,
//! deformation gradient F, position, and phase transitions.

use crate::types::{GridCell, Particle, PhaseTransitionRule, FLAG_EXPLOSION, FLAG_RESET_F, MAX_PHASE_RULES};
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
    /// Number of valid phase transition rules.
    pub num_rules: u32,
}

/// PIC/FLIP blend ratio (0.0 = pure FLIP, 1.0 = pure PIC).
/// 0.7 PIC balances stability with fluid-like flow (less viscous damping).
const PIC_RATIO: f32 = 0.5;

/// Number of u32 values per voxel cell (must match voxelize.rs layout).
const U32S_PER_VOXEL: u32 = 4;

/// Check if a voxel cell at (vx, vy, vz) is occupied by a solid (phase==0).
///
/// Returns `true` if the cell is in-bounds, occupied, and has phase==0.
fn is_solid_voxel(vx: u32, vy: u32, vz: u32, voxels: &[u32], grid_size: u32) -> bool {
    if vx >= grid_size || vy >= grid_size || vz >= grid_size {
        return true; // out of bounds = treat as supported (safe default)
    }
    let idx = (vz * grid_size * grid_size + vy * grid_size + vx) as usize;
    let base = idx * U32S_PER_VOXEL as usize;

    // Check occupied flag (offset +3)
    let occupied = voxels[base + 3] > 0;
    if !occupied {
        return false;
    }

    // Unpack phase from packed_material at offset +0: (weight<<24 | mat<<16 | phase<<8 | 0xFF)
    let packed = voxels[base];
    let phase_below = (packed >> 8) & 0xFF;
    phase_below == 0
}

/// Check if a solid particle has support from a solid voxel below it.
///
/// Reads the voxel buffer (from the previous frame's voxelize pass) to see
/// if the cell below this particle is occupied by a solid (phase==0).
/// Checks 2 cells below to avoid stale-read issues where adjacent solids
/// at the same Y level see each other's old positions as support.
/// Particles at the grid floor (vy <= 1) are always considered supported.
pub fn check_solid_support(pos: spirv_std::glam::Vec3, voxels: &[u32], grid_size: u32) -> bool {
    let vx = pos.x as u32;
    let vy = pos.y as u32;
    let vz = pos.z as u32;

    // At bottom of grid = supported by boundary
    if vy <= 1 {
        return true;
    }

    // Check cell directly below (vy - 1)
    if is_solid_voxel(vx, vy - 1, vz, voxels, grid_size) {
        return true;
    }

    // Check 2 cells below (vy - 2) to catch cases where the voxel buffer
    // is stale by one frame and the support has shifted down by 1 cell.
    if vy >= 3 {
        if is_solid_voxel(vx, vy - 2, vz, voxels, grid_size) {
            return true;
        }
    }

    false
}

/// Gather velocity from the grid and update one particle.
///
/// Implements the G2P transfer step:
/// 1. Gather weighted velocity from 27 neighbor cells (PIC velocity)
/// 2. Blend PIC and FLIP velocities
/// 3. Compute APIC affine momentum matrix C
/// 4. Update deformation gradient F
/// 5. Advect position
/// 6. Check phase transitions by temperature (data-driven via reaction table)
///
/// For solid particles (phase==0), checks support from the voxel below.
/// Supported solids update only F/C (pinned). Unsupported solids fall normally.
pub fn gather_particle(
    particle: &mut Particle,
    grid: &[GridCell],
    voxels: &[u32],
    reactions: &[PhaseTransitionRule],
    grid_size: u32,
    dt: f32,
    num_rules: u32,
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

    // Thermal diffusion: blend grid temp with particle temp for gradual transfer.
    // Full grid temp (blend=1.0) = instant equalization (too fast).
    // Blend=0.002 means 0.2% of grid temp difference per step.
    // At 120 steps/sec: lava at 2000C surrounded by cold stone cools to
    // solidification (~1500C) in ~8-12 seconds. Combined with radiative cooling
    // (0.0005/step), total cooling rate is gentle enough for visual gameplay.
    let thermal_blend = 0.002_f32;
    let old_temp = particle.vel_temp.w;
    new_temp = old_temp + thermal_blend * (new_temp - old_temp);

    // Radiative cooling: particles lose heat to environment over time (Newton's cooling law).
    // Without this, total thermal energy only accumulates (each explosion adds 3000C)
    // and chain explosions escalate infinitely. This ensures heat dissipates even
    // without cold neighbors nearby.
    // At 0.0005 per step (120 steps/sec): hot particle loses ~6% of excess heat per second.
    // 2000C lava takes ~8-10s to cool to solidification (1500C). Gentle enough for gameplay.
    new_temp = apply_radiative_cooling(new_temp);

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
    // Unsupported solids fall normally with minimum fall speed to overcome grid friction.
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
            apply_phase_transitions(particle, reactions, num_rules);
            return;
        }
        // Unsupported solid: bypass PIC blend entirely.
        // PIC blend resets velocity to grid average (~0) every frame, preventing
        // gravity from accumulating. Instead, use old particle velocity + gravity
        // directly, so blocks accelerate like real falling objects.
        let old_vel_y = particle.vel_temp.y;
        let gravity = -196.0_f32;
        new_vel = Vec3::new(
            old_vel.x * 0.9, // slight horizontal damping
            old_vel_y + gravity * dt, // accumulate gravity each frame
            old_vel.z * 0.9,
        );
        // Clamp to terminal velocity (~50 grid-units/s) to prevent instability
        if new_vel.y < -50.0 {
            new_vel.y = -50.0;
        }
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

    // Phase transitions by temperature (data-driven)
    apply_phase_transitions(particle, reactions, num_rules);
}

/// Apply radiative cooling (Newton's cooling law) to a temperature value.
///
/// Particles lose heat to the environment at a rate proportional to the
/// temperature difference from ambient (20C). This prevents thermal runaway
/// where explosions accumulate heat indefinitely.
///
/// `cooling_rate = 0.002` per step. At 120 steps/sec, a 3000C particle
/// loses ~21% of excess heat per second.
fn apply_radiative_cooling(temp: f32) -> f32 {
    let ambient_temp = 20.0_f32;
    let cooling_rate = 0.0005_f32;
    temp + cooling_rate * (ambient_temp - temp)
}

/// GPU-friendly pseudo-random hash for a single float.
///
/// Uses integer bit reinterpretation and xorshift-style mixing to produce
/// a value in [-1.0, 1.0] with reasonable distribution, even for
/// nearby integer-ish inputs (e.g., grid-aligned particle positions).
fn hash_f32(x: f32) -> f32 {
    // Reinterpret float bits as integer for mixing
    let mut bits = x.to_bits();
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x45d9f3b);
    bits ^= bits >> 16;
    // Convert to [-1, 1] range
    let normalized = (bits & 0xFFFF) as f32 / 32768.0 - 1.0;
    normalized
}

/// Apply gunpowder explosion: set velocity to pseudo-random outward direction.
///
/// Uses particle position as a seed for the hash to give each particle
/// a unique explosion direction. Speed is calibrated against gravity (-196)
/// so particles travel ~30-60 grid units before falling back.
///
/// Also boosts temperature to 3000C for thermal pressure effects.
fn apply_gunpowder_explosion(particle: &mut Particle) {
    let px = particle.pos_mass.x;
    let py = particle.pos_mass.y;
    let pz = particle.pos_mass.z;

    // Hash each axis with different seeds for decorrelation
    let hx = hash_f32(px * 73.17 + pz * 37.91);
    let hy = hash_f32(py * 127.31 + px * 53.47);
    let hz = hash_f32(pz * 91.53 + py * 67.13);

    // Explosion speed: 200 grid-units/s.
    // Against gravity=-196, max vertical rise = v^2/(2*g) ~ 102 grid units.
    let explosion_speed = 200.0_f32;
    let vx = hx * explosion_speed;
    let vy_exp = hy.abs() * explosion_speed + 100.0; // always some upward bias
    let vz = hz * explosion_speed;
    particle.vel_temp = Vec4::new(vx, vy_exp, vz, 3000.0);
    // Also push position directly — G2P will overwrite velocity on next frame
    // via grid coupling, so the impulse must also move particles spatially.
    // Same approach as explosion.rs (direct position push).
    let push_scale = 0.02_f32;
    particle.pos_mass = Vec4::new(
        particle.pos_mass.x + vx * push_scale,
        particle.pos_mass.y + vy_exp * push_scale,
        particle.pos_mass.z + vz * push_scale,
        particle.pos_mass.w,
    );
    // Reset F (trap #8)
    particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
    particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
    particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
}

/// Check and apply phase transitions using the data-driven reaction table.
///
/// Loops over the reaction rules and applies the first matching rule.
/// This replaces the previous hardcoded match block, avoiding trap #15
/// (rust-gpu drops later branches in long if/else chains).
///
/// On phase change: reset F = Identity, as per CLAUDE.md trap #8.
pub fn apply_phase_transitions(
    particle: &mut Particle,
    reactions: &[PhaseTransitionRule],
    num_rules: u32,
) {
    let material_id = particle.ids.x;
    let phase = particle.ids.y;
    let temp = particle.vel_temp.w;

    let count = if (num_rules as usize) < MAX_PHASE_RULES {
        num_rules
    } else {
        MAX_PHASE_RULES as u32
    };

    let mut i = 0u32;
    while i < count {
        // Copy values out of storage buffer to avoid OpPhi pointer issues
        // in SPIR-V (rust-gpu generates phi nodes for &reactions[i] across loop iterations).
        let rule_materials = reactions[i as usize].materials;
        let rule_conditions = reactions[i as usize].conditions;
        if rule_materials.x == material_id && rule_materials.y == phase {
            let temp_min = rule_conditions.x;
            let temp_max = rule_conditions.y;
            if temp > temp_min && temp < temp_max {
                let flags = rule_conditions.z as u32;
                let new_material = rule_materials.z;
                let new_phase = rule_materials.w;

                // Apply transition
                particle.ids.x = new_material;
                particle.ids.y = new_phase;

                // Reset F if flag is set (trap #8)
                if flags & FLAG_RESET_F != 0 {
                    reset_deformation_gradient(particle);
                }

                // Apply explosion if flag is set
                if flags & FLAG_EXPLOSION != 0 {
                    apply_gunpowder_explosion(particle);
                }

                return;
            }
        }
        i += 1;
    }
}

/// Reset the deformation gradient F to identity.
///
/// Helper function extracted to satisfy trap #4a (entry points must call
/// at least one helper function).
fn reset_deformation_gradient(particle: &mut Particle) {
    particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
    particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
    particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
}

/// Compute shader entry point: Grid-to-Particle transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read-write).
/// Descriptor set 0, binding 1: storage buffer of `GridCell` (read).
/// Descriptor set 0, binding 2: storage buffer of `u32` (voxel grid, 4 per cell, read).
/// Descriptor set 0, binding 3: storage buffer of `PhaseTransitionRule` (read).
/// Push constants: `G2pPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn g2p(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &G2pPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &[GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] voxels: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] reactions: &[PhaseTransitionRule],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    gather_particle(
        &mut particles[idx],
        grid,
        voxels,
        reactions,
        push.grid_size,
        push.dt,
        push.num_rules,
    );
}
