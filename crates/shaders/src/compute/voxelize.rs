//! Voxelize compute shader.
//!
//! Each thread processes one particle, mapping its position to a voxel cell
//! and writing material_id + temperature to a 3D output buffer.
//! Used for BLAS brick occupancy detection and rendering.
//!
//! ## Majority-vote material selection
//!
//! At material boundaries, multiple particles of different materials contribute
//! to the same voxel cell. A naive last-write-wins causes the material_id to
//! oscillate each frame (visible as flickering).
//!
//! To fix this, we pack `(weight, material_id, phase)` into a single `u32` and
//! use `atomicMax` so the particle with the highest trilinear weight wins
//! deterministically. For equal weights, the higher material_id wins (stable).
//!
//! ### Voxel buffer layout (4 u32s per cell, accessed as flat `&mut [u32]`):
//!
//! | Offset | Field | Description |
//! |--------|-------|-------------|
//! | 0 | packed_material | `(weight_8b << 24) \| (material_id_8b << 16) \| (phase_8b << 8) \| 0xFF` — atomicMax |
//! | 1 | temperature | f32 reinterpreted as u32 bits — last-write from winner |
//! | 2 | _reserved | unused (zero) |
//! | 3 | occupied | non-zero if any particle maps here — atomicMax with 1 |

use crate::types::Particle;
use spirv_std::glam::{UVec3, UVec4};
use spirv_std::spirv;

use super::p2g::should_skip_brick;

/// Number of u32 values per voxel cell.
const U32S_PER_VOXEL: u32 = 4;

/// Push constants for the voxelize shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VoxelizePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
    /// Padding for alignment.
    pub _pad0: u32,
}

/// Pack material selection data into a single u32 for atomic competition.
///
/// Layout: `(weight_8b << 24) | (material_id_8b << 16) | (phase_8b << 8) | 0xFF`
///
/// `atomicMax` on this value means the highest-weight particle wins.
/// For equal weights, the higher material_id wins (deterministic tie-break).
pub fn pack_material(weight_f32: f32, material_id: u32, phase: u32) -> u32 {
    // Quantize weight [0.0, 1.0] to [0, 255]
    let w_clamped = if weight_f32 < 0.0 {
        0.0
    } else if weight_f32 > 1.0 {
        1.0
    } else {
        weight_f32
    };
    let w_u8 = (w_clamped * 255.0) as u32;
    let mat_u8 = material_id & 0xFF;
    let phase_u8 = phase & 0xFF;
    (w_u8 << 24) | (mat_u8 << 16) | (phase_u8 << 8) | 0xFF
}

/// Unpack material_id from a packed voxel material u32.
pub fn unpack_material_id(packed: u32) -> u32 {
    (packed >> 16) & 0xFF
}

/// Unpack phase from a packed voxel material u32.
pub fn unpack_phase(packed: u32) -> u32 {
    (packed >> 8) & 0xFF
}

/// Atomically set the voxel's packed material data using max-wins competition.
///
/// Uses `atomic_u_max` on SPIR-V targets. Falls back to simple comparison
/// on CPU (for testing).
///
/// # Safety
/// Caller must ensure `voxels[index]` is valid and properly aligned.
#[inline]
pub unsafe fn voxel_atomic_max(voxels: &mut [u32], index: usize, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        unsafe {
            spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut voxels[index], value)
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let old = voxels[index];
        if value > old {
            voxels[index] = value;
        }
        old
    }
}

/// Map a particle to voxel cells and write material data using atomic competition.
///
/// Splats to the 2x2x2 nearest cells weighted by trilinear interpolation.
/// Uses `atomicMax` on a packed `(weight, material_id, phase)` value so the
/// particle with the highest contribution weight wins deterministically.
/// This eliminates boundary flickering caused by non-deterministic write order.
///
/// The voxel buffer is accessed as flat `[u32]` with 4 values per cell
/// to enable per-component atomics (same pattern as P2G uses flat `[f32]`).
///
/// Separated into a helper function for the rust-gpu linker bug workaround
/// (see CLAUDE.md trap #4a).
pub fn write_voxel(
    particle: &Particle,
    voxels: &mut [u32],
    grid_size: u32,
) {
    let pos = particle.pos_mass.truncate();
    let material_id = particle.ids.x;
    let phase = particle.ids.y;
    let temp_bits = particle.vel_temp.w.to_bits();

    // Base cell (floor of position)
    let bx = pos.x as u32;
    let by = pos.y as u32;
    let bz = pos.z as u32;

    // Fractional part for trilinear weights
    let fx = pos.x - bx as f32;
    let fy = pos.y - by as f32;
    let fz = pos.z - bz as f32;

    // Write to 2x2x2 neighbors with trilinear weights
    let mut di = 0u32;
    while di < 2 {
        let mut dj = 0u32;
        while dj < 2 {
            let mut dk = 0u32;
            while dk < 2 {
                let cx = bx + di;
                let cy = by + dj;
                let cz = bz + dk;

                if cx < grid_size && cy < grid_size && cz < grid_size {
                    let wx = if di == 0 { 1.0 - fx } else { fx };
                    let wy = if dj == 0 { 1.0 - fy } else { fy };
                    let wz = if dk == 0 { 1.0 - fz } else { fz };
                    let w = wx * wy * wz;

                    // Only write if weight is significant
                    if w > 0.1 {
                        let cell_idx =
                            (cz * grid_size * grid_size + cy * grid_size + cx) as usize;
                        let base = cell_idx * U32S_PER_VOXEL as usize;

                        let packed = pack_material(w, material_id, phase);

                        // Atomic max-wins competition: highest weight determines material
                        // Safety: atomic u32 max on the voxel buffer
                        let old = unsafe { voxel_atomic_max(voxels, base, packed) };

                        // If we won (our value >= old), write temperature
                        // Temperature is continuous, so slight races here are invisible
                        if packed >= old {
                            voxels[base + 1] = temp_bits;
                        }

                        // Mark cell as occupied (any non-zero value)
                        // Safety: atomic u32 max on the occupied slot
                        unsafe {
                            voxel_atomic_max(voxels, base + 3, 1);
                        }
                    }
                }
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }
}

/// Clear a single voxel cell to empty (all 4 u32 components).
///
/// Helper for the clear pass before voxelization.
pub fn clear_voxel_cell(voxels: &mut [u32], base: usize) {
    voxels[base] = 0;
    voxels[base + 1] = 0;
    voxels[base + 2] = 0;
    voxels[base + 3] = 0;
}

/// Check whether a voxel at the given grid coordinates belongs to a frozen brick.
///
/// Returns `true` if the brick's tick_period is 0 (frozen), meaning the brick's
/// voxels are unchanged and do not need clearing or re-voxelization.
pub fn is_frozen_brick(x: u32, y: u32, z: u32, sleep_state: &[u32]) -> bool {
    let brick_size: u32 = 8;
    let bricks_per_axis: u32 = 32; // 256 / 8
    let bx = x / brick_size;
    let by = y / brick_size;
    let bz = z / brick_size;
    if bx >= bricks_per_axis || by >= bricks_per_axis || bz >= bricks_per_axis {
        return false; // out of bounds = do not skip
    }
    let brick_idx = bz * bricks_per_axis * bricks_per_axis + by * bricks_per_axis + bx;
    sleep_state[brick_idx as usize] == 0
}

/// Compute shader entry point: clear voxel grid.
///
/// Must be dispatched before `voxelize` to reset the voxel buffer.
/// Skips clearing voxels in frozen bricks (tick_period == 0) since those
/// particles haven't moved and their voxels are still correct.
/// Descriptor set 0, binding 0: storage buffer of `u32` (voxel grid, 4 per cell).
/// Descriptor set 0, binding 1: storage buffer of `u32` (sleep_state per brick, read).
/// Dispatch with `(GRID_SIZE/4, GRID_SIZE/4, GRID_SIZE/4)` workgroups.
#[spirv(compute(threads(4, 4, 4)))]
pub fn clear_voxels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &VoxelizePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] voxels: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sleep_state: &[u32],
) {
    let grid_size = push.grid_size;
    if id.x >= grid_size || id.y >= grid_size || id.z >= grid_size {
        return;
    }
    // Skip clearing voxels in frozen bricks — their data is still valid
    if is_frozen_brick(id.x, id.y, id.z, sleep_state) {
        return;
    }
    let cell_idx = (id.z * grid_size * grid_size + id.y * grid_size + id.x) as usize;
    let base = cell_idx * U32S_PER_VOXEL as usize;
    clear_voxel_cell(voxels, base);
}

/// Compute shader entry point: voxelize particles.
///
/// Skips particles in frozen bricks (tick_period == 0) since their voxels
/// are unchanged from the previous frame.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (voxel grid, 4 per cell, write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (sleep_state per brick, read).
/// Push constants: `VoxelizePushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn voxelize(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &VoxelizePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] voxels: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sleep_state: &[u32],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    // Copy position to locals (trap #21) and check sleep state before writing
    let pos_mass = particles[idx].pos_mass;
    if should_skip_brick(pos_mass.x, pos_mass.y, pos_mass.z, sleep_state, push.frame_number) {
        return;
    }
    write_voxel(&particles[idx], voxels, push.grid_size);
}
