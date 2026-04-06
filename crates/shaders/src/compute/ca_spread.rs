//! Pressure-based liquid spread compute shader for the CA substrate.
//!
//! Runs AFTER gravity. For each column (x, z), counts liquid height and
//! compares with one horizontal neighbor. If this column is taller by >= 2,
//! moves the top liquid voxel to the neighbor column's landing position.
//!
//! This produces natural water equalization: liquid flows from high to low,
//! settling into flat surfaces rather than hopping like sand.
//!
//! Anti-oscillation: only flow when height diff >= 2.
//! Direction: one neighbor per tick, chosen by hash(x, z, frame) % 4.
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
fn is_liquid_phase(materials: &[u32], mat_id: u32) -> bool {
    if mat_id == 0 {
        return false;
    }
    let phase = ca_types::mat_phase(materials, mat_id);
    phase == 2
}

/// Count the height of consecutive liquid voxels at the TOP of a column.
/// Scans down from y=31, skips air, then counts liquid from the top surface.
fn count_liquid_height(
    chunk_pool: &[u32],
    materials: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) -> u32 {
    let mut count = 0u32;
    // Find top non-air voxel
    let mut y = 31u32;
    loop {
        let v = read_voxel(chunk_pool, slot_id, x, y, z);
        let mat = ca_types::voxel_material_id(v);
        if mat != 0 {
            break;
        }
        if y == 0 {
            return 0;
        }
        y -= 1;
    }
    // Count consecutive liquid from top
    loop {
        let v = read_voxel(chunk_pool, slot_id, x, y, z);
        let mat = ca_types::voxel_material_id(v);
        if !is_liquid_phase(materials, mat) {
            break;
        }
        count += 1;
        if y == 0 {
            break;
        }
        y -= 1;
    }
    count
}

/// Find the y-coordinate of the topmost liquid voxel in a column.
/// Returns 0xFFFFFFFF if no liquid found.
fn find_top_liquid(
    chunk_pool: &[u32],
    materials: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) -> u32 {
    let mut y = 31u32;
    loop {
        let v = read_voxel(chunk_pool, slot_id, x, y, z);
        let mat = ca_types::voxel_material_id(v);
        if is_liquid_phase(materials, mat) {
            return y;
        }
        if y == 0 {
            return 0xFFFFFFFF;
        }
        y -= 1;
    }
}

/// Find the landing y-coordinate in a neighbor column.
/// This is the first air voxel above solid/liquid ground.
/// Returns 0xFFFFFFFF if no valid landing spot.
fn find_landing_y(
    chunk_pool: &[u32],
    materials: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) -> u32 {
    // Scan bottom to top: find first air above support
    let mut y = 0u32;
    loop {
        if y >= ca_types::CA_CHUNK_SIZE {
            return 0xFFFFFFFF;
        }
        let v = read_voxel(chunk_pool, slot_id, x, y, z);
        let mat = ca_types::voxel_material_id(v);
        if mat == 0 {
            // Air voxel found. Check if supported.
            return check_landing_support(chunk_pool, slot_id, x, y, z);
        }
        y += 1;
    }
}

/// Check if landing at (x, y, z) is supported (y==0 or non-air below).
/// Returns y if supported, 0xFFFFFFFF otherwise.
fn check_landing_support(
    chunk_pool: &[u32],
    slot_id: u32,
    x: u32,
    y: u32,
    _z: u32,
) -> u32 {
    if y == 0 {
        return y; // Ground level = supported
    }
    let below = read_voxel(chunk_pool, slot_id, x, y - 1, _z);
    let below_mat = ca_types::voxel_material_id(below);
    if below_mat != 0 {
        return y; // Supported by non-air below
    }
    0xFFFFFFFF
}

/// Equalize liquid between this column and a neighbor column within the same chunk.
fn equalize_same_chunk(
    chunk_pool: &mut [u32],
    materials: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
    nx: u32,
    nz: u32,
) {
    let my_height = count_liquid_height(chunk_pool, materials, slot_id, x, z);
    let neighbor_height = count_liquid_height(chunk_pool, materials, slot_id, nx, nz);

    // Only flow if difference >= 2 (prevents oscillation)
    if my_height <= neighbor_height + 1 {
        return;
    }

    let top_y = find_top_liquid(chunk_pool, materials, slot_id, x, z);
    if top_y == 0xFFFFFFFF {
        return;
    }

    let landing_y = find_landing_y(chunk_pool, materials, slot_id, nx, nz);
    if landing_y == 0xFFFFFFFF {
        return;
    }

    // Don't move liquid upward (would look unnatural)
    if landing_y > top_y + 1 {
        return;
    }

    let voxel = read_voxel(chunk_pool, slot_id, x, top_y, z);
    write_voxel(chunk_pool, slot_id, x, top_y, z, 0); // clear source
    write_voxel(chunk_pool, slot_id, nx, landing_y, nz, voxel); // place at landing
}

/// Equalize liquid across a chunk boundary.
fn equalize_cross_chunk(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
    direction: u32,
    dst_x: u32,
    dst_z: u32,
) {
    let neighbor_slot = ca_types::meta_neighbor_id(metadata, slot_id, direction);
    if neighbor_slot == 0xFFFFFFFF {
        return;
    }

    let my_height = count_liquid_height(chunk_pool, materials, slot_id, x, z);
    let neighbor_height = count_liquid_height(chunk_pool, materials, neighbor_slot, dst_x, dst_z);

    if my_height <= neighbor_height + 1 {
        return;
    }

    let top_y = find_top_liquid(chunk_pool, materials, slot_id, x, z);
    if top_y == 0xFFFFFFFF {
        return;
    }

    let landing_y = find_landing_y(chunk_pool, materials, neighbor_slot, dst_x, dst_z);
    if landing_y == 0xFFFFFFFF {
        return;
    }

    if landing_y > top_y + 1 {
        return;
    }

    let voxel = read_voxel(chunk_pool, slot_id, x, top_y, z);
    write_voxel(chunk_pool, slot_id, x, top_y, z, 0);
    write_voxel(chunk_pool, neighbor_slot, dst_x, landing_y, dst_z, voxel);
}

/// Try equalization in +X direction.
fn try_equalize_plus_x(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    if x + 1 < ca_types::CA_CHUNK_SIZE {
        equalize_same_chunk(chunk_pool, materials, slot_id, x, z, x + 1, z);
        return;
    }
    equalize_cross_chunk(chunk_pool, materials, metadata, slot_id, x, z, 0, 0, z);
}

/// Try equalization in -X direction.
fn try_equalize_minus_x(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    if x > 0 {
        equalize_same_chunk(chunk_pool, materials, slot_id, x, z, x - 1, z);
        return;
    }
    equalize_cross_chunk(chunk_pool, materials, metadata, slot_id, x, z, 1, 31, z);
}

/// Try equalization in +Z direction.
fn try_equalize_plus_z(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    if z + 1 < ca_types::CA_CHUNK_SIZE {
        equalize_same_chunk(chunk_pool, materials, slot_id, x, z, x, z + 1);
        return;
    }
    equalize_cross_chunk(chunk_pool, materials, metadata, slot_id, x, z, 4, x, 0);
}

/// Try equalization in -Z direction.
fn try_equalize_minus_z(
    chunk_pool: &mut [u32],
    materials: &[u32],
    metadata: &[u32],
    slot_id: u32,
    x: u32,
    z: u32,
) {
    if z > 0 {
        equalize_same_chunk(chunk_pool, materials, slot_id, x, z, x, z - 1);
        return;
    }
    equalize_cross_chunk(chunk_pool, materials, metadata, slot_id, x, z, 5, x, 31);
}

/// Processes one column: pick one neighbor direction, equalize liquid height.
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
    // Split into separate if-return blocks (trap #15: max 3 if/else branches)
    if dir == 0 {
        try_equalize_plus_x(chunk_pool, materials, metadata, slot_id, x, z);
        return;
    }
    if dir == 1 {
        try_equalize_minus_x(chunk_pool, materials, metadata, slot_id, x, z);
        return;
    }
    if dir == 2 {
        try_equalize_plus_z(chunk_pool, materials, metadata, slot_id, x, z);
        return;
    }
    // dir == 3
    try_equalize_minus_z(chunk_pool, materials, metadata, slot_id, x, z);
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

/// Compute shader entry point: pressure-based liquid spread on dirty chunks.
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
