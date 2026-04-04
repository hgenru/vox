//! Sparse Particle-to-Grid (P2G) compute shader using hash grid.
//!
//! Same physics as `p2g.rs` but uses a hash grid instead of a dense flat array.
//! Each thread processes one particle, scattering mass, momentum, stress,
//! and temperature to hash grid slots via `hash_grid_insert`.
//!
//! The hash grid has two parallel arrays:
//! - `keys: [u32; capacity]` — packed (x,y,z) or EMPTY sentinel
//! - `values: [f32; capacity * 12]` — GridCell data as flat f32 (for atomics)

use crate::types::{MaterialParams, Particle};
use spirv_std::glam::{UVec3, Vec3};
use spirv_std::spirv;

use super::hash_grid;
use super::p2g::{self, P2gPushConstants};
use super::quadratic_bspline_weights;

/// Number of f32 values per grid cell (velocity_mass: 4 + force_pad: 4 + temp_pad: 4 = 12).
const FLOATS_PER_CELL: u32 = 12;

/// Push constants for the sparse P2G shader.
///
/// Extends the standard P2G push constants with hash grid capacity.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct P2gSparsePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
    /// Hash grid capacity (number of slots, must be power of 2).
    pub hash_capacity: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
    /// Padding to 16-byte alignment.
    pub _pad2: u32,
}

/// Scatter one particle's contribution to the hash grid.
///
/// Same physics as `p2g::scatter_particle` but uses `hash_grid_insert` to
/// find/create slots instead of direct flat-array indexing.
pub fn scatter_particle_sparse(
    particle: &Particle,
    hash_keys: &mut [u32],
    hash_values: &mut [f32],
    grid_size: u32,
    dt: f32,
    hash_capacity: u32,
    materials: &[MaterialParams],
) {
    let pos = particle.pos_mass.truncate();
    let vel = particle.vel_temp.truncate();
    let mass = particle.pos_mass.w;
    let temp = particle.vel_temp.w;

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

    // APIC affine matrix C
    let c_col0 = particle.c_col0.truncate();
    let c_col1 = particle.c_col1.truncate();
    let c_col2 = particle.c_col2.truncate();

    // Compute stress contribution
    let stress = p2g::compute_stress(particle, materials);

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
                    scatter_to_cell(
                        hash_keys, hash_values, hash_capacity,
                        ci, cj, ck,
                        pos, vel, mass, temp,
                        c_col0, c_col1, c_col2,
                        &stress, dt,
                        wx[di as usize] * wy[dj as usize] * wz[dk as usize],
                        particle.ids.y,
                    );
                }
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }
}

/// Scatter a single cell's contribution using the hash grid.
///
/// Helper extracted to keep branch count per function low (trap #15).
fn scatter_to_cell(
    hash_keys: &mut [u32],
    hash_values: &mut [f32],
    hash_capacity: u32,
    ci: u32, cj: u32, ck: u32,
    pos: Vec3, vel: Vec3, mass: f32, temp: f32,
    c_col0: Vec3, c_col1: Vec3, c_col2: Vec3,
    stress: &[Vec3; 3],
    dt: f32,
    w: f32,
    phase: u32,
) {
    // Insert/find slot in hash grid
    let key = hash_grid::pack_key(ci, cj, ck);
    let slot = unsafe { hash_grid::hash_grid_insert(hash_keys, key, hash_capacity) };
    if slot == hash_grid::HASH_GRID_EMPTY_KEY {
        return; // Table full, skip this cell
    }

    let dx = Vec3::new(ci as f32 - pos.x, cj as f32 - pos.y, ck as f32 - pos.z);

    // APIC momentum transfer
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

    let base = slot as usize * FLOATS_PER_CELL as usize;

    // Atomically accumulate
    unsafe {
        p2g::grid_atomic_add(hash_values, base, momentum_x);
        p2g::grid_atomic_add(hash_values, base + 1, momentum_y);
        p2g::grid_atomic_add(hash_values, base + 2, momentum_z);
        p2g::grid_atomic_add(hash_values, base + 3, weighted_mass);
        p2g::grid_atomic_add(hash_values, base + 4, stress_force.x);
        p2g::grid_atomic_add(hash_values, base + 5, stress_force.y);
        p2g::grid_atomic_add(hash_values, base + 6, stress_force.z);
    }

    // Solid flag
    if phase == 0 {
        unsafe {
            p2g::grid_atomic_add(hash_values, base + 7, 1.0);
        }
    }

    // Temperature
    let weighted_temp = temp * weighted_mass;
    unsafe {
        p2g::grid_atomic_add(hash_values, base + 8, weighted_temp);
    }
}

/// Compute shader entry point: sparse Particle-to-Grid transfer.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (hash_keys, read-write with atomics).
/// Descriptor set 0, binding 2: storage buffer of `f32` (hash_values, read-write with atomics).
/// Descriptor set 0, binding 3: storage buffer of `MaterialParams` (material table, read).
/// Descriptor set 0, binding 4: storage buffer of `u32` (sleep_state per brick, read).
/// Push constants: `P2gSparsePushConstants`.
#[spirv(compute(threads(64)))]
pub fn p2g_sparse(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &P2gSparsePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hash_keys: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hash_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[MaterialParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] sleep_state: &[u32],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }

    // NOTE: In sparse mode, the hash grid itself is the sparsity optimization.
    // Sleep-based brick skipping is NOT used here because record_core_physics_sparse()
    // skips mark_active/compute_activity passes, so sleep_state is never updated.
    // Using should_skip_brick() here causes a deadlock: all bricks sleep -> P2G skips
    // all particles -> grid empty -> G2P reads nothing -> particles freeze.
    // The sleep_state binding is kept to avoid descriptor set layout changes.
    let _sleep_state = sleep_state;

    scatter_particle_sparse(
        &particles[idx],
        hash_keys,
        hash_values,
        push.grid_size,
        push.dt,
        push.hash_capacity,
        materials,
    );
}
