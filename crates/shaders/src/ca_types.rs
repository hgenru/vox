//! GPU-side type definitions for the CA (Cellular Automata) substrate.
//!
//! These mirror the CPU-side types in `shared::voxel`, `shared::chunk_gpu`,
//! `shared::material_ca`, and `shared::reaction_ca`. Uses `spirv_std::glam`
//! for SPIR-V compatibility. Memory layouts MUST be kept in sync manually.
//!
//! The voxel is a packed u32 with the following bit layout:
//! - `material_id`: bits 0-9 (10 bits, 0-1023)
//! - `temperature`: bits 10-17 (8 bits, 0-255)
//! - `bonds`: bits 18-23 (6 bits, +X/-X/+Y/-Y/+Z/-Z)
//! - `oxygen`: bits 24-27 (4 bits, 0-15)
//! - `flags`: bits 28-31 (4 bits: dirty, burning, wet, poisoned)

// ---- Constants (must match shared::constants) ----

/// CA chunk size in voxels per axis.
pub const CA_CHUNK_SIZE: u32 = 32;

/// Total voxels per CA chunk (32^3).
pub const CA_CHUNK_VOXELS: u32 = 32 * 32 * 32; // 32768

/// Default ambient temperature for unloaded neighbors (game units).
pub const CA_AMBIENT_TEMP: u32 = 128;

/// Maximum number of material types in the CA system.
pub const CA_MAX_MATERIALS: u32 = 1024;

/// Maximum number of chemical reactions.
pub const CA_MAX_REACTIONS: u32 = 256;

/// Size of one slot in u32 units: voxels (32768) + dirty bitmask (1024).
pub const SLOT_SIZE_U32: u32 = 33792;

/// Number of u32 voxel entries per slot.
pub const VOXELS_PER_SLOT_U32: u32 = 32768;

/// Number of u32s in chunk metadata (64 bytes / 4).
pub const META_SIZE_U32: u32 = 16;

/// Number of u32s per MaterialPropertiesCA (64 bytes / 4).
pub const MATERIAL_SIZE_U32: u32 = 16;

/// Number of u32s per ReactionEntry (32 bytes / 4).
pub const REACTION_SIZE_U32: u32 = 8;

/// Workgroups per chunk for thermal pass (32/4 = 8 regions per axis, 8^3 = 512).
pub const THERMAL_WG_PER_CHUNK: u32 = 512;

/// Workgroups per chunk for Margolus pass (16^3 blocks / 64 threads = 64).
pub const MARGOLUS_WG_PER_CHUNK: u32 = 64;

/// Workgroups per chunk for gravity pass (32/8 = 4 per axis, 4*4 = 16).
pub const GRAVITY_WG_PER_CHUNK: u32 = 16;

// ---- Voxel pack/unpack (must match shared::voxel::Voxel bit layout) ----

/// Extracts the material ID (bits 0-9) from a packed voxel.
pub fn voxel_material_id(v: u32) -> u32 {
    v & 0x3FF
}

/// Extracts the temperature (bits 10-17) from a packed voxel.
pub fn voxel_temperature(v: u32) -> u32 {
    (v >> 10) & 0xFF
}

/// Extracts the bond flags (bits 18-23) from a packed voxel.
pub fn voxel_bonds(v: u32) -> u32 {
    (v >> 18) & 0x3F
}

/// Extracts the oxygen level (bits 24-27) from a packed voxel.
pub fn voxel_oxygen(v: u32) -> u32 {
    (v >> 24) & 0xF
}

/// Extracts the flag bits (bits 28-31) from a packed voxel.
pub fn voxel_flags(v: u32) -> u32 {
    (v >> 28) & 0xF
}

/// Packs voxel fields into a single u32.
///
/// Values are masked to their respective bit widths.
pub fn voxel_pack(material_id: u32, temperature: u32, bonds: u32, oxygen: u32, flags: u32) -> u32 {
    (material_id & 0x3FF)
        | ((temperature & 0xFF) << 10)
        | ((bonds & 0x3F) << 18)
        | ((oxygen & 0xF) << 24)
        | ((flags & 0xF) << 28)
}

/// Returns a voxel with only the temperature field changed.
pub fn voxel_set_temperature(v: u32, temp: u32) -> u32 {
    (v & !(0xFF << 10)) | ((temp & 0xFF) << 10)
}

/// Returns a voxel with only the material ID field changed.
pub fn voxel_set_material(v: u32, mat: u32) -> u32 {
    (v & !0x3FF) | (mat & 0x3FF)
}

/// Returns a voxel with only the bonds field changed.
pub fn voxel_set_bonds(v: u32, bonds: u32) -> u32 {
    (v & !(0x3F << 18)) | ((bonds & 0x3F) << 18)
}

/// Returns a voxel with only the flags field changed.
pub fn voxel_set_flags(v: u32, flags: u32) -> u32 {
    (v & !(0xF << 28)) | ((flags & 0xF) << 28)
}

// ---- Slot addressing ----

/// Computes the u32 index into the chunk pool for a voxel at (x, y, z) in the given slot.
pub fn voxel_addr(slot_id: u32, local_x: u32, local_y: u32, local_z: u32) -> u32 {
    let voxel_idx = local_z * CA_CHUNK_SIZE * CA_CHUNK_SIZE + local_y * CA_CHUNK_SIZE + local_x;
    slot_id * SLOT_SIZE_U32 + voxel_idx
}

// ---- Material table access ----
// MaterialPropertiesCA layout (16 u32s, must match shared::material_ca):
//  [0] phase       [1] density      [2] viscosity    [3] conductivity
//  [4] melt_temp   [5] boil_temp    [6] freeze_temp  [7] melt_into
//  [8] boil_into   [9] freeze_into  [10] flammability [11] ignition_temp
//  [12] burn_into  [13] burn_heat   [14] strength    [15] max_load

/// Returns the phase (0=solid, 1=powder, 2=liquid, 3=gas) for a material.
pub fn mat_phase(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32) as usize]
}

/// Returns the density for a material.
pub fn mat_density(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 1) as usize]
}

/// Returns the thermal conductivity for a material.
pub fn mat_conductivity(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 3) as usize]
}

/// Returns the melt temperature for a material.
pub fn mat_melt_temp(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 4) as usize]
}

/// Returns the boil temperature for a material.
pub fn mat_boil_temp(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 5) as usize]
}

/// Returns the freeze temperature for a material.
pub fn mat_freeze_temp(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 6) as usize]
}

/// Returns the material ID to become when melting.
pub fn mat_melt_into(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 7) as usize]
}

/// Returns the material ID to become when boiling.
pub fn mat_boil_into(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 8) as usize]
}

/// Returns the material ID to become when freezing.
pub fn mat_freeze_into(materials: &[u32], mat_id: u32) -> u32 {
    materials[(mat_id * MATERIAL_SIZE_U32 + 9) as usize]
}

// ---- Chunk metadata access ----
// ChunkGpuMeta layout (16 u32s, must match shared::chunk_gpu):
//  [0..3]  world_pos (IVec4: x,y,z,slot_id)
//  [4..9]  neighbor_ids[6] (+X,-X,+Y,-Y,+Z,-Z as i32)
//  [10]    activity
//  [11]    flags
//  [12..15] _pad

/// Returns the neighbor slot ID for a given direction (0..5: +X,-X,+Y,-Y,+Z,-Z).
///
/// Returns -1 (as u32 = 0xFFFFFFFF) if the neighbor is not loaded.
pub fn meta_neighbor_id(metadata: &[u32], slot_id: u32, direction: u32) -> u32 {
    metadata[(slot_id * META_SIZE_U32 + 4 + direction) as usize]
}

/// Returns the flags field from chunk metadata.
pub fn meta_flags(metadata: &[u32], slot_id: u32) -> u32 {
    metadata[(slot_id * META_SIZE_U32 + 11) as usize]
}

/// Returns the activity level from chunk metadata.
pub fn meta_activity(metadata: &[u32], slot_id: u32) -> u32 {
    metadata[(slot_id * META_SIZE_U32 + 10) as usize]
}

// ---- Reaction table access ----
// ReactionEntry layout (8 u32s, must match shared::reaction_ca):
//  [0] input_a    [1] input_b    [2] output_a   [3] output_b
//  [4] heat_delta [5] condition  [6] probability [7] impulse

/// Returns the input_a material ID for a reaction.
pub fn reaction_input_a(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32) as usize]
}

/// Returns the input_b material ID for a reaction.
pub fn reaction_input_b(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32 + 1) as usize]
}

/// Returns the output_a material ID for a reaction.
pub fn reaction_output_a(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32 + 2) as usize]
}

/// Returns the output_b material ID for a reaction.
pub fn reaction_output_b(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32 + 3) as usize]
}

/// Returns the heat_delta for a reaction (reinterpreted as i32).
pub fn reaction_heat_delta(reactions: &[u32], idx: u32) -> i32 {
    reactions[(idx * REACTION_SIZE_U32 + 4) as usize] as i32
}

/// Returns the condition for a reaction (0=none, 1=needs_oxygen, 2=needs_high_temp).
pub fn reaction_condition(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32 + 5) as usize]
}

/// Returns the probability (0-255) for a reaction.
pub fn reaction_probability(reactions: &[u32], idx: u32) -> u32 {
    reactions[(idx * REACTION_SIZE_U32 + 6) as usize]
}

// ---- Deterministic hash / RNG ----

/// Deterministic spatial hash for pseudo-random number generation.
///
/// Returns a uniform-ish u32 from position and frame number.
pub fn hash_position(x: u32, y: u32, z: u32, frame: u32) -> u32 {
    let mut h = x.wrapping_mul(73856093)
        ^ y.wrapping_mul(19349663)
        ^ z.wrapping_mul(83492791)
        ^ frame.wrapping_mul(48611);
    h = h.wrapping_mul(2654435761);
    h ^= h >> 16;
    h
}
