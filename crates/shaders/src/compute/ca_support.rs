//! Structural integrity compute shader for the CA substrate.
//!
//! Runs BEFORE gravity. Scans each (x, z) column bottom to top, tracking
//! support state. Solid voxels (phase=0) that lack support below are
//! converted to rubble (mat_id=10, phase=1) so gravity can drop them.
//!
//! Support rules:
//! - y=0 with no chunk below (world edge): always supported
//! - y=0 with chunk below: supported if the voxel at y=31 of chunk below is solid
//! - y>0: supported if y-1 is solid AND y-1 was itself supported
//! - Non-solid voxels (air, liquid, powder, gas) break the support chain
//!
//! Workgroup layout: (8, 1, 8) = 64 threads per workgroup.
//! Each thread processes one column (x, z) within a chunk.
//! 32/8 = 4 workgroups per axis -> 4*4 = 16 workgroups per chunk.
//! Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the support pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaSupportPush {
    /// Current frame number.
    pub frame_number: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Rubble material ID. Must match the CPU-side `default_ca_materials()` index.
const RUBBLE_MAT_ID: u32 = 10;

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

/// Checks if a voxel is solid (phase == 0) and non-air.
fn is_solid(materials: &[u32], voxel: u32) -> bool {
    let mat_id = ca_types::voxel_material_id(voxel);
    if mat_id == 0 {
        return false;
    }
    let phase = ca_types::mat_phase(materials, mat_id);
    phase == 0
}

/// Determines initial support state from the chunk below.
///
/// If no chunk below exists (world edge), returns true (ground level).
/// Otherwise checks if the voxel at y=31 in the chunk below is solid.
fn initial_support(
    chunk_pool: &[u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) -> bool {
    // -Y neighbor is direction 3
    let below_slot = ca_types::meta_neighbor_id(metadata, slot_id, 3);
    if below_slot == 0xFFFFFFFF {
        return true; // No chunk below = world edge = supported
    }
    let below_voxel = read_voxel(chunk_pool, below_slot, x, 31, z);
    is_solid(materials, below_voxel)
}

/// Converts an unsupported solid voxel to rubble by changing its material ID.
///
/// Preserves temperature, bonds, oxygen, and flags.
fn convert_to_rubble(chunk_pool: &mut [u32], slot_id: u32, x: u32, y: u32, z: u32, voxel: u32) {
    let new_voxel = ca_types::voxel_set_material(voxel, RUBBLE_MAT_ID);
    write_voxel(chunk_pool, slot_id, x, y, z, new_voxel);
}

/// Processes one column, converting unsupported solids to rubble.
fn process_column(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    let mut supported = initial_support(chunk_pool, materials, metadata, slot_id, x, z);

    let mut y: u32 = 0;
    loop {
        if y >= ca_types::CA_CHUNK_SIZE {
            return;
        }
        let voxel = read_voxel(chunk_pool, slot_id, x, y, z);
        let mat_id = ca_types::voxel_material_id(voxel);

        if mat_id == 0 {
            // Air: breaks the support chain
            supported = false;
        } else {
            let phase = ca_types::mat_phase(materials, mat_id);
            let solid = phase == 0;
            supported = update_support(chunk_pool, slot_id, x, y, z, voxel, solid, supported);
        }

        y += 1;
    }
}

/// Updates support state for a single voxel.
///
/// If the voxel is solid and unsupported, converts it to rubble and returns false.
/// If the voxel is solid and supported, returns true (chain continues).
/// If the voxel is non-solid (liquid/powder/gas), returns false (chain breaks).
fn update_support(
    chunk_pool: &mut [u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    solid: bool,
    supported: bool,
) -> bool {
    if solid {
        if !supported {
            convert_to_rubble(chunk_pool, slot_id, x, y, z, voxel);
            return false;
        }
        return true;
    }
    // Non-solid non-air (liquid, powder, gas): breaks support chain
    false
}

/// Main support logic: maps thread to chunk + column, then processes.
fn support_impl(
    wg_id: UVec3,
    local_id: UVec3,
    chunk_pool: &mut [u32],
    metadata: &[u32],
    dirty_list: &[u32],
    materials: &[u32],
) {
    let wg_per_chunk = ca_types::SUPPORT_WG_PER_CHUNK; // 16
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

/// Compute shader entry point: structural integrity check on dirty chunks.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool (voxel gigabuffer, read/write)
/// - binding 1: metadata (ChunkGpuMeta array as `&[u32]`, read)
/// - binding 2: dirty_list (dirty chunk IDs with header, read)
/// - binding 3: materials (MaterialPropertiesCA array as `&[u32]`, read)
///
/// Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.
#[spirv(compute(threads(8, 1, 8)))]
pub fn ca_support(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dirty_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
    #[spirv(push_constant)] _push: &CaSupportPush,
) {
    // Entry point must call a helper function (trap #4a)
    support_impl(wg_id, local_id, chunk_pool, metadata, dirty_list, materials);
}
