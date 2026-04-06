//! Horizontal liquid spread compute shader for the CA substrate.
//!
//! Runs AFTER gravity. For each column (x, z), scans bottom to top.
//! If a liquid voxel is supported (solid/liquid/powder below), tries to
//! spread into one adjacent horizontal air cell.
//!
//! Uses frame-based direction hashing to randomize spread direction per tick.
//! Only ONE spread per column per tick to prevent mass loss.
//!
//! Workgroup layout: (8, 1, 8) = 64 threads per workgroup.
//! Each thread processes one column (x, z) within a chunk.
//! 32/8 = 4 workgroups per axis -> 4*4 = 16 workgroups per chunk.
//! Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the spread pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaSpreadPush {
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

/// Checks if a voxel is liquid (phase == 2).
fn is_liquid(materials: &[u32], voxel: u32) -> bool {
    let mat_id = ca_types::voxel_material_id(voxel);
    if mat_id == 0 {
        return false;
    }
    let phase = ca_types::mat_phase(materials, mat_id);
    phase == 2
}

/// Checks if a voxel is non-air (can support liquid above).
fn is_support(voxel: u32) -> bool {
    ca_types::voxel_material_id(voxel) != 0
}

/// Tries to spread a liquid voxel in the +X direction within the same chunk.
/// Returns true if spread occurred.
fn try_spread_plus_x(
    chunk_pool: &mut [u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    metadata: &[u32],
) -> bool {
    if x + 1 < ca_types::CA_CHUNK_SIZE {
        let neighbor = read_voxel(chunk_pool, slot_id, x + 1, y, z);
        if ca_types::voxel_material_id(neighbor) == 0 {
            write_voxel(chunk_pool, slot_id, x + 1, y, z, voxel);
            write_voxel(chunk_pool, slot_id, x, y, z, 0);
            return true;
        }
        return false;
    }
    // Cross-chunk: +X neighbor is direction 0
    try_spread_cross_chunk(chunk_pool, metadata, slot_id, 0, 0, y, z, x, y, z, voxel)
}

/// Tries to spread a liquid voxel in the -X direction within the same chunk.
/// Returns true if spread occurred.
fn try_spread_minus_x(
    chunk_pool: &mut [u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    metadata: &[u32],
) -> bool {
    if x > 0 {
        let neighbor = read_voxel(chunk_pool, slot_id, x - 1, y, z);
        if ca_types::voxel_material_id(neighbor) == 0 {
            write_voxel(chunk_pool, slot_id, x - 1, y, z, voxel);
            write_voxel(chunk_pool, slot_id, x, y, z, 0);
            return true;
        }
        return false;
    }
    // Cross-chunk: -X neighbor is direction 1
    try_spread_cross_chunk(chunk_pool, metadata, slot_id, 1, 31, y, z, x, y, z, voxel)
}

/// Tries to spread a liquid voxel in the +Z direction within the same chunk.
/// Returns true if spread occurred.
fn try_spread_plus_z(
    chunk_pool: &mut [u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    metadata: &[u32],
) -> bool {
    if z + 1 < ca_types::CA_CHUNK_SIZE {
        let neighbor = read_voxel(chunk_pool, slot_id, x, y, z + 1);
        if ca_types::voxel_material_id(neighbor) == 0 {
            write_voxel(chunk_pool, slot_id, x, y, z + 1, voxel);
            write_voxel(chunk_pool, slot_id, x, y, z, 0);
            return true;
        }
        return false;
    }
    // Cross-chunk: +Z neighbor is direction 4
    try_spread_cross_chunk(chunk_pool, metadata, slot_id, 4, x, y, 0, x, y, z, voxel)
}

/// Tries to spread a liquid voxel in the -Z direction within the same chunk.
/// Returns true if spread occurred.
fn try_spread_minus_z(
    chunk_pool: &mut [u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    metadata: &[u32],
) -> bool {
    if z > 0 {
        let neighbor = read_voxel(chunk_pool, slot_id, x, y, z - 1);
        if ca_types::voxel_material_id(neighbor) == 0 {
            write_voxel(chunk_pool, slot_id, x, y, z - 1, voxel);
            write_voxel(chunk_pool, slot_id, x, y, z, 0);
            return true;
        }
        return false;
    }
    // Cross-chunk: -Z neighbor is direction 5
    try_spread_cross_chunk(chunk_pool, metadata, slot_id, 5, x, y, 31, x, y, z, voxel)
}

/// Attempts a cross-chunk spread in the given direction.
///
/// `direction`: 0=+X, 1=-X, 4=+Z, 5=-Z
/// `dst_x/y/z`: coordinates in the neighbor chunk to write liquid into
/// `src_x/y/z`: coordinates in this chunk to clear (write air)
fn try_spread_cross_chunk(
    chunk_pool: &mut [u32],
    metadata: &[u32],
    slot_id: u32,
    direction: u32,
    dst_x: u32,
    dst_y: u32,
    dst_z: u32,
    src_x: u32,
    src_y: u32,
    src_z: u32,
    voxel: u32,
) -> bool {
    let neighbor_slot = ca_types::meta_neighbor_id(metadata, slot_id, direction);
    if neighbor_slot == 0xFFFFFFFF {
        return false;
    }
    let neighbor_voxel = read_voxel(chunk_pool, neighbor_slot, dst_x, dst_y, dst_z);
    if ca_types::voxel_material_id(neighbor_voxel) == 0 {
        write_voxel(chunk_pool, neighbor_slot, dst_x, dst_y, dst_z, voxel);
        write_voxel(chunk_pool, slot_id, src_x, src_y, src_z, 0);
        return true;
    }
    false
}

/// Checks if a voxel at (x, y, z) is at the liquid surface.
///
/// A liquid is "at the surface" if:
/// - it is supported (solid/liquid/powder below), AND
/// - the voxel directly above is NOT liquid (air, gas, or out-of-bounds).
///
/// This prevents underwater liquid from spreading horizontally, which causes
/// the "boiling" oscillation artifact.
fn is_surface_liquid(
    chunk_pool: &[u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
) -> bool {
    // Must be supported
    let supported = check_support(chunk_pool, metadata, slot_id, x, y, z);
    if !supported {
        return false;
    }

    // Check voxel above: if liquid, we're underwater -> don't spread
    let above = read_voxel_or_neighbor_above(chunk_pool, metadata, slot_id, x, y, z);
    let above_mat = ca_types::voxel_material_id(above);
    if above_mat != 0 {
        let above_phase = ca_types::mat_phase(materials, above_mat);
        // phase 2 = liquid in our material table
        if above_phase == 2 {
            return false;
        }
    }

    true
}

/// Reads the voxel above (x, y+1, z), crossing chunk boundary if needed.
fn read_voxel_or_neighbor_above(
    chunk_pool: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
) -> u32 {
    if y + 1 < ca_types::CA_CHUNK_SIZE {
        return read_voxel(chunk_pool, slot_id, x, y + 1, z);
    }
    // Cross-chunk: +Y neighbor is direction 2
    let above_slot = ca_types::meta_neighbor_id(metadata, slot_id, 2);
    if above_slot == 0xFFFFFFFF {
        return 0; // No chunk above = air
    }
    read_voxel(chunk_pool, above_slot, x, 0, z)
}

/// Processes one column, spreading the first supported surface liquid horizontally.
///
/// Only surface liquid (supported from below, no liquid above) spreads.
/// This prevents internal "boiling" oscillation where underwater voxels
/// bounce back and forth.
fn process_column(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
    frame: u32,
) {
    let dir = ca_types::hash_position(x, 0, z, frame) % 4;
    let mut y: u32 = 0;
    loop {
        if y >= ca_types::CA_CHUNK_SIZE {
            return;
        }
        let voxel = read_voxel(chunk_pool, slot_id, x, y, z);
        let liquid = is_liquid(materials, voxel);

        if liquid {
            // Only spread surface liquid (supported below, no liquid above)
            let surface = is_surface_liquid(chunk_pool, materials, metadata, slot_id, x, y, z);
            if surface {
                let did_spread = try_spread_dir(chunk_pool, metadata, slot_id, x, y, z, voxel, dir);
                if did_spread {
                    return; // One spread per column per tick
                }
            }
        }
        y += 1;
    }
}

/// Checks if a voxel at (x, y, z) is supported (non-air below or y==0).
fn check_support(
    chunk_pool: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
) -> bool {
    if y == 0 {
        // Check cross-chunk: -Y neighbor (direction 3)
        let below_slot = ca_types::meta_neighbor_id(metadata, slot_id, 3);
        if below_slot == 0xFFFFFFFF {
            return true; // Edge of world = supported
        }
        let below_voxel = read_voxel(chunk_pool, below_slot, x, 31, z);
        return is_support(below_voxel);
    }
    let below = read_voxel(chunk_pool, slot_id, x, y - 1, z);
    is_support(below)
}

/// Attempts spread in a given direction (0=+X, 1=-X, 2=+Z, 3=-Z).
fn try_spread_dir(
    chunk_pool: &mut [u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    y: u32,
    z: u32,
    voxel: u32,
    dir: u32,
) -> bool {
    if dir == 0 {
        return try_spread_plus_x(chunk_pool, slot_id, x, y, z, voxel, metadata);
    }
    if dir == 1 {
        return try_spread_minus_x(chunk_pool, slot_id, x, y, z, voxel, metadata);
    }
    if dir == 2 {
        return try_spread_plus_z(chunk_pool, slot_id, x, y, z, voxel, metadata);
    }
    // dir == 3
    try_spread_minus_z(chunk_pool, slot_id, x, y, z, voxel, metadata)
}

/// Main spread logic: maps thread to chunk + column, then processes.
fn spread_impl(
    wg_id: UVec3,
    local_id: UVec3,
    chunk_pool: &mut [u32],
    metadata: &[u32],
    dirty_list: &[u32],
    materials: &[u32],
    frame: u32,
) {
    let wg_per_chunk = ca_types::SPREAD_WG_PER_CHUNK; // 16
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

    process_column(chunk_pool, materials, metadata, slot_id, col_x, col_z, frame);
}

/// Compute shader entry point: horizontal liquid spread on dirty chunks.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool (voxel gigabuffer, read/write)
/// - binding 1: metadata (ChunkGpuMeta array as `&[u32]`, read)
/// - binding 2: dirty_list (dirty chunk IDs with header, read)
/// - binding 3: materials (MaterialPropertiesCA array as `&[u32]`, read)
///
/// Dispatched indirectly: X = dirty_count * 16, Y = 1, Z = 1.
#[spirv(compute(threads(8, 1, 8)))]
pub fn ca_spread(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dirty_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
    #[spirv(push_constant)] push: &CaSpreadPush,
) {
    // Entry point must call a helper function (trap #4a)
    spread_impl(
        wg_id,
        local_id,
        chunk_pool,
        metadata,
        dirty_list,
        materials,
        push.frame_number,
    );
}
