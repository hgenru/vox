//! Particle-to-Grid (P2G) compute shader.
//!
//! Each thread processes one particle, scattering mass, momentum, stress,
//! and temperature to the 27 neighboring grid cells using quadratic B-spline weights.
//! Uses `spirv_std::arch::atomic_f_add()` for concurrent accumulation.
//!
//! The grid buffer is accessed as flat `&mut [f32]` to enable per-component
//! float atomics. Each `GridCell` occupies 12 consecutive f32 values:
//! `[vel_x, vel_y, vel_z, mass, force_x, force_y, force_z, solid_flag, temp, pad, pad, pad]`.

use crate::types::{MaterialParams, Particle};
use spirv_std::glam::{UVec3, Vec3};
use spirv_std::spirv;

use super::quadratic_bspline_weights;

/// Number of f32 values per grid cell (velocity_mass: 4 + force_pad: 4 + temp_pad: 4 = 12).
const FLOATS_PER_CELL: u32 = 12;

/// Push constants for the P2G shader.
///
/// Contains simulation parameters passed per-dispatch.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct P2gPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Padding to maintain alignment.
    pub _pad: u32,
}

/// Compute the stress contribution from the particle's constitutive model.
///
/// For solid (phase=0): simplified fixed corotated elasticity.
/// For liquid (phase=1): equation of state pressure.
/// For gas (phase=2): weak EOS pressure.
///
/// Reads viscosity from the MaterialParams buffer instead of hardcoded values.
///
/// Returns the Cauchy stress tensor columns packed into Vec3s.
pub fn compute_stress(particle: &Particle, materials: &[MaterialParams]) -> [Vec3; 3] {
    let phase = particle.ids.y;
    let f_col0 = particle.f_col0.truncate();
    let f_col1 = particle.f_col1.truncate();
    let f_col2 = particle.f_col2.truncate();

    // Determinant of F (volume ratio)
    let j = f_col0.dot(f_col1.cross(f_col2));

    if phase == 1 || phase == 2 {
        // Liquid / Gas: EOS pressure + viscous stress.
        // Read viscosity from material buffer.
        let material_id = particle.ids.x;
        let mat_count = materials.len() as u32;
        let safe_id = if material_id < mat_count {
            material_id
        } else {
            0
        };
        let mat = &materials[safe_id as usize];

        let (bulk, viscosity) = if phase == 2 {
            // Gas: weak pressure, very low viscosity
            (2.0_f32, 0.001_f32)
        } else {
            // Liquid: read viscosity from material elastic.w
            (20.0_f32, mat.elastic.w)
        };

        let pressure = bulk * (j - 1.0);

        // Viscous stress: approximated from APIC C matrix (velocity gradient).
        // D = 0.5 * (C + C^T), then deviatoric part opposes shearing flow.
        let c0 = particle.c_col0.truncate();
        let c1 = particle.c_col1.truncate();
        let c2 = particle.c_col2.truncate();

        // Symmetric strain rate D = 0.5*(C + C^T)
        let d00 = c0.x;
        let d11 = c1.y;
        let d22 = c2.z;
        let d01 = 0.5 * (c0.y + c1.x);
        let d02 = 0.5 * (c0.z + c2.x);
        let d12 = 0.5 * (c1.z + c2.y);

        // Deviatoric part: D' = D - (tr(D)/3)*I
        let tr_d = d00 + d11 + d22;
        let third_tr = tr_d / 3.0;
        let dev00 = d00 - third_tr;
        let dev11 = d11 - third_tr;
        let dev22 = d22 - third_tr;

        // Cauchy stress = -p*I + 2*eta*D'
        let s00 = -pressure + 2.0 * viscosity * dev00;
        let s11 = -pressure + 2.0 * viscosity * dev11;
        let s22 = -pressure + 2.0 * viscosity * dev22;
        let s01 = 2.0 * viscosity * d01;
        let s02 = 2.0 * viscosity * d02;
        let s12 = 2.0 * viscosity * d12;

        [
            Vec3::new(s00, s01, s02),
            Vec3::new(s01, s11, s12),
            Vec3::new(s02, s12, s22),
        ]
    } else {
        // Solid: simplified neo-Hookean / fixed corotated
        // mu and lambda from Young's modulus E=1e4, nu=0.3
        let mu = 3846.0_f32; // E / (2*(1+nu))
        let lambda = 5769.0_f32; // E*nu / ((1+nu)*(1-2*nu))

        let log_j = if j > 1.0e-6 { ln_approx(j) } else { 0.0 };

        let stress_scale = 1.0 / (j.abs().max(1.0e-6));
        let s0 = mu * (f_col0 - Vec3::X) * stress_scale + Vec3::X * lambda * log_j;
        let s1 = mu * (f_col1 - Vec3::Y) * stress_scale + Vec3::Y * lambda * log_j;
        let s2 = mu * (f_col2 - Vec3::Z) * stress_scale + Vec3::Z * lambda * log_j;
        [s0, s1, s2]
    }
}

/// Approximate natural logarithm for GPU (no std::ln available in no_std).
///
/// Uses a Pade approximant around 1.0. Accurate for values near 1.0,
/// which is typical for J (volume ratio) in MPM.
pub fn ln_approx(x: f32) -> f32 {
    // ln(x) ~ 2 * (x-1)/(x+1) for x near 1
    let xm1 = x - 1.0;
    let xp1 = x + 1.0;
    if xp1.abs() < 1.0e-10 {
        return 0.0;
    }
    let r = xm1 / xp1;
    let r2 = r * r;
    // Taylor: 2*(r + r^3/3 + r^5/5)
    2.0 * r * (1.0 + r2 * (1.0 / 3.0 + r2 * (1.0 / 5.0)))
}

/// Atomically add a value to a single f32 in the grid buffer.
///
/// Uses `spirv_std::arch::atomic_f_add` on SPIR-V targets.
/// Falls back to non-atomic addition on CPU (for testing).
///
/// # Safety
/// The caller must ensure `grid[index]` is a valid location and that
/// concurrent atomic access is properly synchronized via SPIR-V semantics.
#[inline]
pub unsafe fn grid_atomic_add(grid: &mut [f32], index: usize, value: f32) {
    #[cfg(target_arch = "spirv")]
    {
        // SCOPE=1 (Device), SEMANTICS=0x0 (Relaxed)
        unsafe {
            spirv_std::arch::atomic_f_add::<f32, 1u32, 0x0u32>(&mut grid[index], value);
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        grid[index] += value;
    }
}

/// Scatter one particle's contribution to the grid using P2G transfer.
///
/// This is the core MPM transfer step. Each particle deposits mass,
/// momentum, and stress to its 27 neighboring grid cells weighted
/// by quadratic B-spline interpolation.
///
/// The grid buffer is flat f32 with 12 floats per cell:
/// `[vel_x, vel_y, vel_z, mass, force_x, force_y, force_z, solid_flag, temp, pad, pad, pad]`.
pub fn scatter_particle(
    particle: &Particle,
    grid: &mut [f32],
    grid_size: u32,
    dt: f32,
    materials: &[MaterialParams],
) {
    let pos = particle.pos_mass.truncate();
    let vel = particle.vel_temp.truncate();
    let mass = particle.pos_mass.w;
    let temp = particle.vel_temp.w;

    // Base cell index (integer part of position - 0.5, since B-spline support is [-1.5, 1.5])
    let base_x = (pos.x - 0.5).max(0.0) as u32;
    let base_y = (pos.y - 0.5).max(0.0) as u32;
    let base_z = (pos.z - 0.5).max(0.0) as u32;

    // Fractional position within the cell (for weight computation)
    let fx = pos.x - base_x as f32;
    let fy = pos.y - base_y as f32;
    let fz = pos.z - base_z as f32;

    let wx = quadratic_bspline_weights(fx);
    let wy = quadratic_bspline_weights(fy);
    let wz = quadratic_bspline_weights(fz);

    // APIC affine matrix C
    let c_col0 = particle.c_col0.truncate();
    let c_col1 = particle.c_col1.truncate();
    let c_col2 = particle.c_col2.truncate();

    // Compute stress contribution (reads viscosity from material buffer)
    let stress = compute_stress(particle, materials);

    // Scatter to 3x3x3 neighborhood
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

                    // Distance from particle to this grid cell
                    let dx = Vec3::new(ci as f32 - pos.x, cj as f32 - pos.y, ck as f32 - pos.z);

                    // APIC momentum transfer: v_i = v_p + C * dx
                    let affine_vel = vel
                        + Vec3::new(
                            c_col0.dot(dx),
                            c_col1.dot(dx),
                            c_col2.dot(dx),
                        );

                    // Stress force contribution
                    let stress_force = Vec3::new(
                        -(stress[0].x * dx.x + stress[0].y * dx.y + stress[0].z * dx.z),
                        -(stress[1].x * dx.x + stress[1].y * dx.y + stress[1].z * dx.z),
                        -(stress[2].x * dx.x + stress[2].y * dx.y + stress[2].z * dx.z),
                    ) * w * dt;

                    let weighted_mass = w * mass;
                    let momentum_x = affine_vel.x * weighted_mass;
                    let momentum_y = affine_vel.y * weighted_mass;
                    let momentum_z = affine_vel.z * weighted_mass;

                    // Flat index into grid buffer (8 floats per cell)
                    let cell_idx = (ck * grid_size * grid_size + cj * grid_size + ci) as usize;
                    let base = cell_idx * FLOATS_PER_CELL as usize;

                    // Atomically accumulate momentum (vel_x, vel_y, vel_z) and mass
                    // Safety: atomic float adds to the grid buffer
                    unsafe {
                        grid_atomic_add(grid, base, momentum_x);
                        grid_atomic_add(grid, base + 1, momentum_y);
                        grid_atomic_add(grid, base + 2, momentum_z);
                        grid_atomic_add(grid, base + 3, weighted_mass);
                        // Force accumulation
                        grid_atomic_add(grid, base + 4, stress_force.x);
                        grid_atomic_add(grid, base + 5, stress_force.y);
                        grid_atomic_add(grid, base + 6, stress_force.z);
                    }

                    // Mark cells with solid particle contribution (phase == 0)
                    // so grid_update can enforce solid boundary conditions.
                    // Uses the pad slot (index 7) as a solid flag.
                    if particle.ids.y == 0 {
                        unsafe {
                            grid_atomic_add(grid, base + 7, 1.0);
                        }
                    }

                    // Scatter weighted temperature to temp_pad.x (index 8).
                    // Accumulated as temp * mass * weight, normalized by mass
                    // in grid_update to get average temperature per cell.
                    let weighted_temp = temp * weighted_mass;
                    unsafe {
                        grid_atomic_add(grid, base + 8, weighted_temp);
                    }
                }
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }
}

/// Compute shader entry point: Particle-to-Grid transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `f32` (grid, read-write with atomics).
///   Layout: 12 floats per cell `[vel_x, vel_y, vel_z, mass, force_x, force_y, force_z, solid_flag, temp, pad, pad, pad]`.
/// Descriptor set 0, binding 2: storage buffer of `MaterialParams` (material table, read).
/// Push constants: `P2gPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn p2g(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &P2gPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] materials: &[MaterialParams],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    scatter_particle(&particles[idx], grid, push.grid_size, push.dt, materials);
}
