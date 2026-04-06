//! Compute shader that converts CA chunk voxels into the flat render voxel buffer format.
//!
//! This bridge shader reads from the CA gigabuffer (packed u32 voxels per chunk)
//! and writes into the dense flat voxel buffer that the existing render shader expects.
//!
//! ## Voxel buffer layout (4 u32s per cell, matching `voxelize.rs`):
//!
//! | Offset | Field | Description |
//! |--------|-------|-------------|
//! | 0 | packed_material | `(0xFF << 24) \| (material_id << 16) \| (phase << 8) \| 0xFF` |
//! | 1 | temperature | f32 reinterpreted as u32 bits |
//! | 2 | _reserved | unused (zero) |
//! | 3 | occupied | 1 if non-air, 0 otherwise |

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the CA-to-render conversion shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaToRenderPush {
    /// Number of loaded chunks to process.
    pub num_chunks: u32,
    /// Flat render grid dimension (e.g. 128).
    pub grid_size: u32,
    /// World offset of the flat grid origin (X).
    pub grid_origin_x: i32,
    /// World offset of the flat grid origin (Y).
    pub grid_origin_y: i32,
    /// World offset of the flat grid origin (Z).
    pub grid_origin_z: i32,
    /// Padding to 32-byte alignment.
    pub _pad: [u32; 3],
}

/// CA voxel bit layout constants (mirroring shared::voxel).
const MATERIAL_ID_MASK: u32 = 0x3FF; // bits 0-9
const TEMPERATURE_SHIFT: u32 = 10;
const TEMPERATURE_MASK: u32 = 0xFF; // bits 10-17

/// Number of voxels per chunk axis.
const CHUNK_SIZE: u32 = 32;
/// Total voxels per chunk (32^3).
const VOXELS_PER_CHUNK: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Number of u32s per voxel in the flat render buffer.
const U32S_PER_VOXEL: u32 = 4;

/// Metadata stride in u32s. ChunkGpuMeta = 64 bytes = 16 u32s.
const META_U32S: u32 = 16;

/// Slot stride in u32s. Each slot = 135168 bytes = 33792 u32s.
/// (32768 voxels * 4 bytes + 4096 bitmask bytes) / 4.
const SLOT_U32S: u32 = 33792;

/// Extract material_id from a CA voxel.
fn ca_material_id(voxel: u32) -> u32 {
    voxel & MATERIAL_ID_MASK
}

/// Remap CA material ID to render material ID.
///
/// The render shader has hardcoded checks for old material IDs:
///   old MAT_LAVA=2 → emissive glow
///   old MAT_WOOD=3 → emissive when burning
/// CA IDs: 0=air, 1=stone, 2=sand, 3=water, 4=lava, 5=steam, 6=ice
///
/// We remap so the material_render_buffer (indexed by render_mat_id) has
/// correct colors AND the hardcoded emissive checks match.
///
/// Remap: CA 4 (lava) → render 2 (MAT_LAVA), others keep their CA ID
/// but shifted to avoid collision: CA 2 (sand) → render 4 (was ash slot)
fn remap_material_id(ca_mat: u32) -> u32 {
    if ca_mat == 4 { 2 }       // lava → old MAT_LAVA slot (emissive works)
    else if ca_mat == 2 { 4 }  // sand → old MAT_ASH slot (no collision)
    else { ca_mat }             // stone=1→1, water=3→3, steam=5→5, ice=6→6
}

/// Extract temperature from a CA voxel.
fn ca_temperature(voxel: u32) -> u32 {
    (voxel >> TEMPERATURE_SHIFT) & TEMPERATURE_MASK
}

/// Remap CA phase to render phase.
///
/// CA:     0=solid, 1=powder, 2=liquid, 3=gas
/// Render: 0=solid, 1=liquid, 2=gas
fn remap_phase(ca_phase: u32) -> u32 {
    if ca_phase <= 1 {
        0 // solid + powder → render solid
    } else if ca_phase == 2 {
        1 // liquid → render liquid
    } else {
        2 // gas → render gas
    }
}

/// Convert one CA voxel to the flat render buffer format and write it.
///
/// This helper exists so the entry point is not inlined (rust-gpu trap #4a).
fn convert_and_write(
    chunk_idx: u32,
    local_x: u32,
    local_y: u32,
    local_z: u32,
    push: &CaToRenderPush,
    chunk_pool: &[u32],
    metadata: &[u32],
    render_voxels: &mut [u32],
    materials: &[u32],
) {
    if chunk_idx >= push.num_chunks {
        return;
    }

    // Read chunk metadata: world_pos is at offset 0 of the metadata entry (IVec4 = 4 i32s)
    let meta_base = (chunk_idx * META_U32S) as usize;
    // world_pos: x, y, z, w=slot_id stored as i32 in the first 4 u32 slots
    let chunk_wx = metadata[meta_base] as i32;
    let chunk_wy = metadata[meta_base + 1] as i32;
    let chunk_wz = metadata[meta_base + 2] as i32;
    let slot_id = metadata[meta_base + 3];

    // World position of this voxel
    let wx = chunk_wx * CHUNK_SIZE as i32 + local_x as i32;
    let wy = chunk_wy * CHUNK_SIZE as i32 + local_y as i32;
    let wz = chunk_wz * CHUNK_SIZE as i32 + local_z as i32;

    // Convert to flat grid coordinates
    let gx = wx - push.grid_origin_x;
    let gy = wy - push.grid_origin_y;
    let gz = wz - push.grid_origin_z;

    // Bounds check
    let gs = push.grid_size as i32;
    if gx < 0 || gy < 0 || gz < 0 || gx >= gs || gy >= gs || gz >= gs {
        return;
    }

    // Read CA voxel from chunk pool using slot_id
    // Each slot = SLOT_U32S u32s, voxel data is at the start of the slot
    let voxel_offset = slot_id * SLOT_U32S;
    let local_idx = local_z * CHUNK_SIZE * CHUNK_SIZE + local_y * CHUNK_SIZE + local_x;
    let ca_voxel = chunk_pool[(voxel_offset + local_idx) as usize];

    let mat_id = ca_material_id(ca_voxel);

    // Flat grid index
    let flat_idx = (gz as u32 * push.grid_size * push.grid_size
        + gy as u32 * push.grid_size
        + gx as u32) as usize;
    let base = flat_idx * U32S_PER_VOXEL as usize;

    if mat_id == 0 {
        // Air: write zeros
        render_voxels[base] = 0;
        render_voxels[base + 1] = 0;
        render_voxels[base + 2] = 0;
        render_voxels[base + 3] = 0;
        return;
    }

    let temp = ca_temperature(ca_voxel);

    // Look up phase from material table
    // MaterialPropertiesCA is 64 bytes = 16 u32s; phase is the first u32
    let mat_table_base = (mat_id * 16) as usize;
    let ca_phase = if mat_table_base < materials.len() {
        materials[mat_table_base]
    } else {
        0
    };

    // Remap CA phase to render phase:
    // CA:     0=solid, 1=powder, 2=liquid, 3=gas
    // Render: 0=solid, 1=liquid, 2=gas
    let render_phase = remap_phase(ca_phase);

    // Remap CA material ID to render material ID for hardcoded emissive checks
    let render_mat = remap_material_id(mat_id);

    // Pack material: (0xFF << 24) | (material_id << 16) | (phase << 8) | 0xFF
    let packed_material = (0xFF << 24) | ((render_mat & 0xFF) << 16) | ((render_phase & 0xFF) << 8) | 0xFF;

    // Temperature: only emit heat glow for actually hot materials (lava=4).
    // Render shader's apply_emissive triggers on material_id==2 (old MAT_LAVA).
    // We remap CA lava (mat_id=4) to render material_id=4, so we need the
    // emissive check to match. For now: set temp=0 for non-hot materials
    // to suppress false glow, and high temp only for lava.
    let temp_f32 = if mat_id == 4 {
        temp as f32 * 10.0 // Lava: hot, will glow
    } else {
        0.0 // Everything else: no thermal emission
    };
    let temp_bits = temp_f32.to_bits();

    render_voxels[base] = packed_material;
    render_voxels[base + 1] = temp_bits;
    render_voxels[base + 2] = 0;
    render_voxels[base + 3] = 1; // occupied
}

/// Compute shader entry point: convert CA chunk voxels to flat render voxel buffer.
///
/// Each workgroup processes a 4x4x4 region within a chunk.
/// Dispatch with `(num_chunks * 8, 8, 8)` workgroups (8 WGs per axis for 32^3 / 4^3).
///
/// Descriptor set 0:
/// - binding 0: chunk_pool voxel buffer (CA format, read)
/// - binding 1: chunk metadata buffer (read)
/// - binding 2: flat render voxel buffer (write)
/// - binding 3: CA material table (read)
///
/// Push constants: `CaToRenderPush`.
#[spirv(compute(threads(4, 4, 4)))]
pub fn ca_to_render(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(push_constant)] push: &CaToRenderPush,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] render_voxels: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
) {
    // global_id.x covers all chunks * 32 voxels in X
    // Determine which chunk and local position
    let total_x = global_id.x;
    let chunk_idx = total_x / CHUNK_SIZE;
    let local_x = total_x % CHUNK_SIZE;
    let local_y = global_id.y;
    let local_z = global_id.z;

    if local_y >= CHUNK_SIZE || local_z >= CHUNK_SIZE {
        return;
    }

    convert_and_write(
        chunk_idx,
        local_x,
        local_y,
        local_z,
        push,
        chunk_pool,
        metadata,
        render_voxels,
        materials,
    );
}
