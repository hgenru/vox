//! Thermal diffusion compute shader for the CA substrate.
//!
//! Processes dirty chunks via indirect dispatch. Each thread handles one voxel,
//! reading 6 face-neighbor temperatures, computing a conductivity-weighted delta,
//! and checking phase transitions (melt/boil/freeze).
//!
//! All arithmetic is integer to ensure determinism across GPU vendors.
//!
//! Workgroup layout: (4, 4, 4) = 64 threads per workgroup.
//! Each chunk needs 8x8x8 = 512 workgroups (one 4x4x4 region each).
//! Dispatched indirectly: X = dirty_count * 512, Y = 1, Z = 1.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the thermal diffusion pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaThermalPush {
    /// Current frame number (for debugging/hashing).
    pub frame_number: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Dirty list header size in u32s: [dirty_count, dispatch_x, dispatch_y, dispatch_z].
const DIRTY_LIST_HEADER: u32 = 4;

// ---- Helper functions (trap #4a and #15 compliance) ----

/// Reads the temperature of a neighbor voxel, handling chunk boundaries.
///
/// For interior neighbors (0 <= coord < 32), reads from the same slot.
/// For boundary neighbors, looks up the neighbor chunk in metadata.
/// If the neighbor chunk is not loaded, returns `CA_AMBIENT_TEMP`.
fn read_neighbor_temp(
    chunk_pool: &[u32],
    metadata: &[u32],
    slot_id: u32,
    lx: i32,
    ly: i32,
    lz: i32,
    direction: u32,
) -> u32 {
    let cs = ca_types::CA_CHUNK_SIZE as i32;

    // Interior case: all coordinates are within [0, 32)
    let in_bounds = lx >= 0 && lx < cs && ly >= 0 && ly < cs && lz >= 0 && lz < cs;
    if in_bounds {
        let addr = ca_types::voxel_addr(slot_id, lx as u32, ly as u32, lz as u32);
        let voxel = chunk_pool[addr as usize];
        return ca_types::voxel_temperature(voxel);
    }

    // Boundary case: read from neighbor chunk
    read_neighbor_temp_cross_chunk(chunk_pool, metadata, slot_id, lx, ly, lz, direction)
}

/// Reads temperature from a cross-chunk neighbor voxel.
///
/// Looks up the neighbor chunk slot in metadata. If not loaded (-1), returns ambient.
fn read_neighbor_temp_cross_chunk(
    chunk_pool: &[u32],
    metadata: &[u32],
    slot_id: u32,
    lx: i32,
    ly: i32,
    lz: i32,
    direction: u32,
) -> u32 {
    let cs = ca_types::CA_CHUNK_SIZE as i32;
    let neighbor_slot = ca_types::meta_neighbor_id(metadata, slot_id, direction);

    // neighbor_slot == 0xFFFFFFFF means not loaded
    if neighbor_slot == 0xFFFFFFFF {
        return ca_types::CA_AMBIENT_TEMP;
    }

    // Wrap coordinates into [0, 32)
    let wx = ((lx % cs) + cs) % cs;
    let wy = ((ly % cs) + cs) % cs;
    let wz = ((lz % cs) + cs) % cs;

    let addr = ca_types::voxel_addr(neighbor_slot, wx as u32, wy as u32, wz as u32);
    let voxel = chunk_pool[addr as usize];
    ca_types::voxel_temperature(voxel)
}

/// Computes thermal delta using integer arithmetic for determinism.
///
/// Returns the signed temperature change to apply.
fn compute_thermal_delta(
    my_temp: u32,
    neighbor_temps: [u32; 6],
    conductivity: u32,
) -> i32 {
    // Average of 6 neighbors (integer division)
    let sum = neighbor_temps[0] + neighbor_temps[1] + neighbor_temps[2]
        + neighbor_temps[3] + neighbor_temps[4] + neighbor_temps[5];
    let avg_temp = sum / 6;

    // Delta = conductivity * (avg - my) >> 8  (fixed-point: conductivity/256)
    let diff = avg_temp as i32 - my_temp as i32;
    (conductivity as i32 * diff) >> 8
}

/// Checks for melting phase transition and returns a new voxel if triggered.
///
/// Returns `Some(new_voxel)` if melted, `None` otherwise.
fn check_melt(materials: &[u32], mat_id: u32, new_temp: u32, voxel: u32) -> u32 {
    let melt_temp = ca_types::mat_melt_temp(materials, mat_id);
    let melt_into = ca_types::mat_melt_into(materials, mat_id);

    if melt_into != 0 && new_temp > melt_temp && melt_temp > 0 {
        let v = ca_types::voxel_set_material(voxel, melt_into);
        // Clear bonds on phase transition
        return ca_types::voxel_set_bonds(v, 0);
    }
    voxel
}

/// Checks for boiling phase transition and returns a new voxel if triggered.
fn check_boil(materials: &[u32], mat_id: u32, new_temp: u32, voxel: u32) -> u32 {
    let boil_temp = ca_types::mat_boil_temp(materials, mat_id);
    let boil_into = ca_types::mat_boil_into(materials, mat_id);

    if boil_into != 0 && new_temp > boil_temp && boil_temp > 0 {
        let v = ca_types::voxel_set_material(voxel, boil_into);
        return ca_types::voxel_set_bonds(v, 0);
    }
    voxel
}

/// Checks for freezing phase transition and returns a new voxel if triggered.
fn check_freeze(materials: &[u32], mat_id: u32, new_temp: u32, voxel: u32) -> u32 {
    let freeze_temp = ca_types::mat_freeze_temp(materials, mat_id);
    let freeze_into = ca_types::mat_freeze_into(materials, mat_id);

    if freeze_into != 0 && new_temp < freeze_temp {
        let v = ca_types::voxel_set_material(voxel, freeze_into);
        return ca_types::voxel_set_bonds(v, 0);
    }
    voxel
}

/// Clamps an i32 to [0, 255] and returns as u32.
fn clamp_temp(t: i32) -> u32 {
    if t < 0 {
        0
    } else if t > 255 {
        255
    } else {
        t as u32
    }
}

/// Main thermal diffusion logic for one voxel.
///
/// Reads neighbors, computes delta, checks phase transitions, writes back.
fn thermal_main(
    chunk_pool: &mut [u32],
    metadata: &[u32],
    materials: &[u32],
    slot_id: u32,
    lx: u32,
    ly: u32,
    lz: u32,
) {
    // Read current voxel
    let addr = ca_types::voxel_addr(slot_id, lx, ly, lz);
    let voxel = chunk_pool[addr as usize];

    // Skip air (material_id == 0)
    let mat_id = ca_types::voxel_material_id(voxel);
    if mat_id == 0 {
        return;
    }

    let my_temp = ca_types::voxel_temperature(voxel);
    let conductivity = ca_types::mat_conductivity(materials, mat_id);

    // Skip insulators (conductivity == 0)
    if conductivity == 0 {
        return;
    }

    // Read 6 face-neighbor temperatures
    // Directions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    let ix = lx as i32;
    let iy = ly as i32;
    let iz = lz as i32;

    let t0 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix + 1, iy, iz, 0);
    let t1 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix - 1, iy, iz, 1);
    let t2 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix, iy + 1, iz, 2);
    let t3 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix, iy - 1, iz, 3);
    let t4 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix, iy, iz + 1, 4);
    let t5 = read_neighbor_temp(chunk_pool, metadata, slot_id, ix, iy, iz - 1, 5);

    let neighbor_temps = [t0, t1, t2, t3, t4, t5];

    // Compute temperature delta
    let delta = compute_thermal_delta(my_temp, neighbor_temps, conductivity);
    let new_temp = clamp_temp(my_temp as i32 + delta);

    // Update temperature in voxel
    let mut updated = ca_types::voxel_set_temperature(voxel, new_temp);

    // Check phase transitions (each in its own function to avoid >3 branches)
    updated = check_melt(materials, mat_id, new_temp, updated);
    // Re-read material in case it changed from melt
    let cur_mat = ca_types::voxel_material_id(updated);
    updated = check_boil(materials, cur_mat, new_temp, updated);
    let cur_mat2 = ca_types::voxel_material_id(updated);
    updated = check_freeze(materials, cur_mat2, new_temp, updated);

    // Write back
    chunk_pool[addr as usize] = updated;
}

/// Compute shader entry point: thermal diffusion on dirty chunks.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool (voxel gigabuffer, read/write)
/// - binding 1: metadata (ChunkGpuMeta array as `&[u32]`, read)
/// - binding 2: dirty_list (dirty chunk IDs with header, read)
/// - binding 3: materials (MaterialPropertiesCA array as `&[u32]`, read)
///
/// Dispatched indirectly: X = dirty_count * 512, Y = 1, Z = 1.
#[spirv(compute(threads(4, 4, 4)))]
pub fn ca_thermal(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dirty_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
    #[spirv(push_constant)] _push: &CaThermalPush,
) {
    // Map workgroup to dirty chunk + region within chunk
    let wg_per_chunk = ca_types::THERMAL_WG_PER_CHUNK; // 512
    let chunk_index = wg_id.x / wg_per_chunk;
    let region_in_chunk = wg_id.x % wg_per_chunk;

    // Read dirty count from header
    let dirty_count = dirty_list[0];
    if chunk_index >= dirty_count {
        return;
    }

    // Get slot_id from dirty list (offset by header)
    let slot_id = dirty_list[(DIRTY_LIST_HEADER + chunk_index) as usize];

    // Decompose region_in_chunk into 3D region coordinates
    // 8 regions per axis (32/4 = 8), so 8x8x8 = 512 regions
    let rz = region_in_chunk / 64;
    let ry = (region_in_chunk / 8) % 8;
    let rx = region_in_chunk % 8;

    // Compute voxel position within chunk
    let local_x = rx * 4 + local_id.x;
    let local_y = ry * 4 + local_id.y;
    let local_z = rz * 4 + local_id.z;

    // Bounds check (should always pass for 32^3 with 4x4x4 workgroups)
    if local_x >= ca_types::CA_CHUNK_SIZE || local_y >= ca_types::CA_CHUNK_SIZE || local_z >= ca_types::CA_CHUNK_SIZE {
        return;
    }

    thermal_main(chunk_pool, metadata, materials, slot_id, local_x, local_y, local_z);
}
