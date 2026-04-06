//! Column-scan gravity compute shader for the CA substrate.
//!
//! Scans each (x, z) column within dirty chunks and drops non-solid voxels
//! down through air gaps. Processes entire columns for fast multi-voxel falling,
//! unlike Margolus which only moves 1 voxel per step within 2x2x2 blocks.
//!
//! Workgroup layout: (8, 1, 8) = 64 threads per workgroup.
//! Each thread processes one column (x, z) within a chunk.
//! 32/8 = 4 workgroups per axis -> 4*4 = 16 workgroups per chunk.
//! Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the gravity pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaGravityPush {
    /// Current frame number.
    pub frame_number: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Dirty list header size in u32s: [dirty_count, dispatch_x, dispatch_y, dispatch_z].
const DIRTY_LIST_HEADER: u32 = 4;

/// Reads a voxel from the chunk pool. Returns 0 (air) for out-of-bounds.
fn read_voxel(chunk_pool: &[u32], slot_id: u32, x: u32, y: u32, z: u32) -> u32 {
    if y >= ca_types::CA_CHUNK_SIZE {
        return 0;
    }
    let addr = ca_types::voxel_addr(slot_id, x, y, z);
    chunk_pool[addr as usize]
}

/// Writes a voxel to the chunk pool.
fn write_voxel(chunk_pool: &mut [u32], slot_id: u32, x: u32, y: u32, z: u32, voxel: u32) {
    let addr = ca_types::voxel_addr(slot_id, x, y, z);
    chunk_pool[addr as usize] = voxel;
}

/// Checks if a voxel can fall (phase >= 1: powder, liquid, gas).
fn can_fall(materials: &[u32], voxel: u32) -> bool {
    let mat_id = ca_types::voxel_material_id(voxel);
    if mat_id == 0 {
        return false;
    }
    let phase = ca_types::mat_phase(materials, mat_id);
    phase >= 1
}

/// Performs one bottom-to-top sweep of a column, dropping non-solid voxels into air below.
///
/// Returns true if any swap occurred.
fn sweep_column(
    chunk_pool: &mut [u32],
    materials: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) -> bool {
    let mut swapped = false;
    let mut y: u32 = 0;
    loop {
        if y >= 31 {
            return swapped;
        }
        let here = read_voxel(chunk_pool, slot_id, x, y, z);
        let above = read_voxel(chunk_pool, slot_id, x, y + 1, z);

        // Copy to locals to check (trap #21)
        let here_mat = ca_types::voxel_material_id(here);
        let fall = can_fall(materials, above);

        if here_mat == 0 && fall {
            write_voxel(chunk_pool, slot_id, x, y, z, above);
            write_voxel(chunk_pool, slot_id, x, y + 1, z, here);
            swapped = true;
        }

        y += 1;
    }
}

/// Try to move voxel at y=0 of current chunk into y=31 of chunk below.
///
/// Reads neighbor_ids from metadata: -Y neighbor is at index 3 (offset +4+3=7 in meta u32s).
fn try_cross_chunk_fall(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    let here = read_voxel(chunk_pool, slot_id, x, 0, z);
    if !can_fall(materials, here) {
        return;
    }

    // Read -Y neighbor slot from metadata
    // ChunkGpuMeta: [world_pos(4)] [neighbor_ids(6): +X,-X,+Y,-Y,+Z,-Z] [activity] [flags] [pad(4)]
    // -Y neighbor is at offset 4+3 = 7
    let meta_base = slot_id * ca_types::META_SIZE_U32;
    let below_slot_raw = metadata[(meta_base + 7) as usize];
    // -1 (0xFFFFFFFF) means not loaded
    if below_slot_raw == 0xFFFFFFFF {
        return;
    }
    let below_slot = below_slot_raw;

    let below_voxel = read_voxel(chunk_pool, below_slot, x, 31, z);
    let below_mat = ca_types::voxel_material_id(below_voxel);
    if below_mat == 0 {
        // Air below in neighbor chunk: drop!
        write_voxel(chunk_pool, below_slot, x, 31, z, here);
        write_voxel(chunk_pool, slot_id, x, 0, z, 0); // air
    }
}

/// Processes one column with multiple passes for faster cascading.
fn process_column(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    // 4 passes to allow voxels to fall up to 4 cells per dispatch
    let mut pass: u32 = 0;
    loop {
        if pass >= 4 {
            break;
        }
        let did_swap = sweep_column(chunk_pool, materials, slot_id, x, z);
        if !did_swap {
            break;
        }
        pass += 1;
    }

    // Cross-chunk: if voxel at y=0 can fall, move to y=31 of chunk below
    try_cross_chunk_fall(chunk_pool, materials, metadata, slot_id, x, z);
}

/// Main gravity logic: maps thread to chunk + column, then processes.
fn gravity_impl(
    wg_id: UVec3,
    local_id: UVec3,
    chunk_pool: &mut [u32],
    metadata: &[u32],
    dirty_list: &[u32],
    materials: &[u32],
) {
    let wg_per_chunk = ca_types::GRAVITY_WG_PER_CHUNK; // 16
    let chunk_index = wg_id.x / wg_per_chunk;
    let local_wg = wg_id.x % wg_per_chunk;

    // Read dirty count from header
    let dirty_count = dirty_list[0];
    if chunk_index >= dirty_count {
        return;
    }

    // Get slot_id from dirty list (offset by header)
    let slot_id = dirty_list[(DIRTY_LIST_HEADER + chunk_index) as usize];

    // Decompose local_wg into 2D workgroup position (4x4 grid of workgroups)
    let wg_z = local_wg / 4;
    let wg_x = local_wg % 4;

    // Compute column (x, z) within chunk
    let col_x = wg_x * 8 + local_id.x;
    let col_z = wg_z * 8 + local_id.z;

    // Bounds check
    if col_x >= ca_types::CA_CHUNK_SIZE {
        return;
    }
    if col_z >= ca_types::CA_CHUNK_SIZE {
        return;
    }

    process_column(chunk_pool, materials, metadata, slot_id, col_x, col_z);
}

/// Compute shader entry point: column-scan gravity on dirty chunks.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool (voxel gigabuffer, read/write)
/// - binding 1: metadata (ChunkGpuMeta array as `&[u32]`, read)
/// - binding 2: dirty_list (dirty chunk IDs with header, read)
/// - binding 3: materials (MaterialPropertiesCA array as `&[u32]`, read)
///
/// Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.
#[spirv(compute(threads(8, 1, 8)))]
pub fn ca_gravity(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dirty_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
    #[spirv(push_constant)] _push: &CaGravityPush,
) {
    // Entry point must call a helper function (trap #4a)
    gravity_impl(wg_id, local_id, chunk_pool, metadata, dirty_list, materials);
}
