//! Mark active grid cells compute shader.
//!
//! Runs per-particle (1D dispatch, 64 threads/workgroup). For each particle,
//! marks the 3x3x3 neighborhood of grid cells that the particle influences.
//! Uses atomic operations to build a compact list of unique active cell indices.
//!
//! The mark buffer (one u32 per grid cell) tracks which cells have already been
//! added to the active list, avoiding duplicates. An atomic counter tracks the
//! total number of active cells.

use crate::types::Particle;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the mark_active shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct MarkActivePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of particles.
    pub num_particles: u32,
    /// Padding for alignment.
    pub _pad0: u32,
    /// Padding for alignment.
    pub _pad1: u32,
}

/// Atomically try to set a mark buffer entry from 0 to 1, returning the old value.
///
/// Uses `atomic_u_max` with value 1 on SPIR-V targets. Since marks are 0 or 1,
/// max(0, 1) = 1 sets the mark, and the returned old value tells us whether
/// this thread was the first to mark it.
///
/// Falls back to non-atomic read-write on CPU (for testing).
///
/// # Safety
/// Caller must ensure `mark[index]` is valid and properly aligned for atomic access.
#[inline]
pub unsafe fn mark_atomic_set(mark: &mut [u32], index: usize) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { spirv_std::arch::atomic_u_max::<u32, 1u32, 0x0u32>(&mut mark[index], 1) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let old = mark[index];
        mark[index] = 1;
        old
    }
}

/// Atomically increment a u32 counter, returning the old value (used as a slot index).
///
/// Uses `spirv_std::arch::atomic_i_add` on SPIR-V targets (works for unsigned too
/// since the bit pattern addition is the same).
///
/// Falls back to non-atomic increment on CPU (for testing).
///
/// # Safety
/// Caller must ensure `counter[index]` is valid and properly aligned for atomic access.
#[inline]
pub unsafe fn atomic_add_u32(counter: &mut [u32], index: usize, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { spirv_std::arch::atomic_i_add::<u32, 1u32, 0x0u32>(&mut counter[index], value) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let old = counter[index];
        counter[index] = old + value;
        old
    }
}

/// Try to mark a single grid cell as active.
///
/// If the cell coordinates are in bounds and the cell hasn't been marked yet,
/// atomically marks it and appends its flat index to the active_cells list.
///
/// Helper function to avoid long if/else chains in the 3x3x3 loop (trap #15)
/// and to satisfy the entry-point-must-call-helper requirement (trap #4a).
pub fn try_mark_cell(
    ci: i32,
    cj: i32,
    ck: i32,
    grid_size: u32,
    mark: &mut [u32],
    active_cells: &mut [u32],
    active_count: &mut [u32],
) {
    let gs = grid_size as i32;
    if ci < 0 || ci >= gs || cj < 0 || cj >= gs || ck < 0 || ck >= gs {
        return;
    }
    let cell_idx = (ci as u32) * grid_size * grid_size + (cj as u32) * grid_size + (ck as u32);

    // Atomically try to mark this cell. If old value was 0, we are the first thread.
    let old = unsafe { mark_atomic_set(mark, cell_idx as usize) };
    if old == 0 {
        // We won the race — append to active list
        let slot = unsafe { atomic_add_u32(active_count, 0, 1) };
        active_cells[slot as usize] = cell_idx;
    }
}

/// Scatter marks for a single particle's 3x3x3 grid neighborhood.
///
/// Iterates over the 27 neighboring cells and calls `try_mark_cell` for each.
/// Uses while-loops (not for-loops) since rust-gpu doesn't support range iteration.
/// Copies particle fields to locals to avoid buffer references in loops (trap #21).
pub fn scatter_marks(
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    grid_size: u32,
    mark: &mut [u32],
    active_cells: &mut [u32],
    active_count: &mut [u32],
) {
    let base_x = pos_x as i32 - 1;
    let base_y = pos_y as i32 - 1;
    let base_z = pos_z as i32 - 1;

    let mut di = 0i32;
    while di < 3 {
        let mut dj = 0i32;
        while dj < 3 {
            let mut dk = 0i32;
            while dk < 3 {
                try_mark_cell(
                    base_x + di,
                    base_y + dj,
                    base_z + dk,
                    grid_size,
                    mark,
                    active_cells,
                    active_count,
                );
                dk += 1;
            }
            dj += 1;
        }
        di += 1;
    }
}

/// Compute shader entry point: mark active grid cells.
///
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (mark buffer, read-write).
/// Descriptor set 0, binding 2: storage buffer of `u32` (active cells list, write).
/// Descriptor set 0, binding 3: storage buffer of `u32` (active count atomic, read-write).
/// Push constants: `MarkActivePushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn mark_active(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &MarkActivePushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &[Particle],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mark: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_cells: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_count: &mut [u32],
) {
    let idx = id.x;
    if idx >= push.num_particles {
        return;
    }
    // Copy particle position to locals (trap #21: no references to buffer elements in loops)
    let pos_mass = particles[idx as usize].pos_mass;
    scatter_marks(
        pos_mass.x,
        pos_mass.y,
        pos_mass.z,
        push.grid_size,
        mark,
        active_cells,
        active_count,
    );
}
