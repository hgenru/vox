//! Margolus 2x2x2 block automaton compute shader for falling sand + chemical reactions.
//!
//! Called twice per CA step: once with even offset (0,0,0) and once with odd offset (1,1,1).
//! Each thread processes one 2x2x2 block within a dirty chunk.
//!
//! Workgroup layout: (4, 4, 4) = 64 threads per workgroup.
//! Each chunk has 16x16x16 = 4096 blocks, needing 4096/64 = 64 workgroups.
//! Dispatched indirectly: X = dirty_count * 64, Y = 1, Z = 1.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the Margolus pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaMargolusPush {
    /// Current frame number (for deterministic RNG).
    pub frame_number: u32,
    /// Block offset X (0 for even pass, 1 for odd pass).
    pub offset_x: u32,
    /// Block offset Y (0 for even, 1 for odd).
    pub offset_y: u32,
    /// Block offset Z (0 for even, 1 for odd).
    pub offset_z: u32,
    /// Number of active reactions in the reaction table.
    pub num_reactions: u32,
    /// Padding to 32-byte alignment.
    pub _pad: [u32; 3],
}

/// Dirty list header size in u32s.
const DIRTY_LIST_HEADER: u32 = 4;

// ---- Voxel read/write helpers ----

/// Reads a voxel from the chunk pool at (x, y, z) in the given slot.
///
/// Returns 0 (air) if coordinates are out of bounds.
fn read_voxel(chunk_pool: &[u32], slot_id: u32, x: u32, y: u32, z: u32) -> u32 {
    if x >= ca_types::CA_CHUNK_SIZE || y >= ca_types::CA_CHUNK_SIZE || z >= ca_types::CA_CHUNK_SIZE {
        return 0;
    }
    let addr = ca_types::voxel_addr(slot_id, x, y, z);
    chunk_pool[addr as usize]
}

/// Writes a voxel to the chunk pool at (x, y, z) in the given slot.
fn write_voxel(chunk_pool: &mut [u32], slot_id: u32, x: u32, y: u32, z: u32, voxel: u32) {
    if x >= ca_types::CA_CHUNK_SIZE || y >= ca_types::CA_CHUNK_SIZE || z >= ca_types::CA_CHUNK_SIZE {
        return;
    }
    let addr = ca_types::voxel_addr(slot_id, x, y, z);
    chunk_pool[addr as usize] = voxel;
}

// ---- Falling sand helpers ----

/// Determines whether two voxels should swap positions (top falls into bottom).
///
/// Returns true if the top voxel is heavier and should sink.
fn should_swap_falling(materials: &[u32], top_mat: u32, bot_mat: u32) -> bool {
    // Air (material 0) is always displaced
    if top_mat == 0 {
        return false;
    }
    if bot_mat == 0 {
        // Non-air on top, air on bottom: fall if top is powder/liquid/gas
        let phase_top = ca_types::mat_phase(materials, top_mat);
        // powder=1, liquid=2, gas=3 can all fall into air
        return phase_top >= 1;
    }

    should_swap_by_density(materials, top_mat, bot_mat)
}

/// Checks density-based swapping for non-air voxels.
///
/// Powder sinks in liquid, liquid sinks in gas, etc.
fn should_swap_by_density(materials: &[u32], top_mat: u32, bot_mat: u32) -> bool {
    let phase_top = ca_types::mat_phase(materials, top_mat);
    let phase_bot = ca_types::mat_phase(materials, bot_mat);

    // Solids (phase 0) never move in CA
    if phase_top == 0 {
        return false;
    }
    // Can't displace a solid
    if phase_bot == 0 {
        return false;
    }

    let density_top = ca_types::mat_density(materials, top_mat);
    let density_bot = ca_types::mat_density(materials, bot_mat);

    // Heavier sinks: top is heavier than bottom
    density_top > density_bot
}

/// Applies falling-sand rules to a vertical column pair within the 2x2x2 block.
///
/// Swaps top and bottom if the top is heavier. Returns (new_top, new_bot).
fn apply_gravity_pair(materials: &[u32], top: u32, bot: u32) -> (u32, u32) {
    let top_mat = ca_types::voxel_material_id(top);
    let bot_mat = ca_types::voxel_material_id(bot);

    if should_swap_falling(materials, top_mat, bot_mat) {
        (bot, top)
    } else {
        (top, bot)
    }
}

// ---- Gas buoyancy helper ----

/// Applies gas buoyancy: gas rises through heavier materials.
///
/// If bottom is gas and top is heavier, swap. Returns (new_top, new_bot).
fn apply_buoyancy_pair(materials: &[u32], top: u32, bot: u32) -> (u32, u32) {
    let bot_mat = ca_types::voxel_material_id(bot);
    if bot_mat == 0 {
        return (top, bot);
    }

    let top_mat = ca_types::voxel_material_id(top);
    if top_mat == 0 {
        // Top is air, bottom is something: gas can rise into air
        let phase_bot = ca_types::mat_phase(materials, bot_mat);
        if phase_bot == 3 {
            return (bot, top);
        }
        return (top, bot);
    }

    // Both non-air: gas phase rises through heavier
    let phase_bot = ca_types::mat_phase(materials, bot_mat);
    if phase_bot == 3 {
        let density_top = ca_types::mat_density(materials, top_mat);
        let density_bot = ca_types::mat_density(materials, bot_mat);
        if density_top > density_bot {
            return (bot, top);
        }
    }

    (top, bot)
}

// ---- Reaction helpers ----

/// Tries to apply a chemical reaction between two adjacent voxels.
///
/// Linear scans the reaction table for a match. On match, applies outputs
/// and heat delta. Returns (new_a, new_b, reacted).
fn try_reactions(
    reactions: &[u32],
    num_reactions: u32,
    va: u32,
    vb: u32,
    rng: u32,
) -> (u32, u32, bool) {
    let mat_a = ca_types::voxel_material_id(va);
    let mat_b = ca_types::voxel_material_id(vb);

    // Skip if either is air
    if mat_a == 0 {
        return (va, vb, false);
    }
    if mat_b == 0 {
        return (va, vb, false);
    }

    scan_reactions_forward(reactions, num_reactions, va, vb, mat_a, mat_b, rng)
}

/// Scans the reaction table for a matching pair (forward: a=input_a, b=input_b).
fn scan_reactions_forward(
    reactions: &[u32],
    num_reactions: u32,
    va: u32,
    vb: u32,
    mat_a: u32,
    mat_b: u32,
    rng: u32,
) -> (u32, u32, bool) {
    let mut i: u32 = 0;
    loop {
        if i >= num_reactions {
            // Also try reversed order
            return scan_reactions_reversed(reactions, num_reactions, va, vb, mat_a, mat_b, rng);
        }

        // Copy to locals to avoid &buffer[i] in loop (trap #21)
        let r_input_a = ca_types::reaction_input_a(reactions, i);
        let r_input_b = ca_types::reaction_input_b(reactions, i);

        if r_input_a == mat_a && r_input_b == mat_b {
            return apply_reaction_if_probable(reactions, i, va, vb, rng, false);
        }

        i += 1;
    }
}

/// Scans the reaction table for a matching pair (reversed: a=input_b, b=input_a).
fn scan_reactions_reversed(
    reactions: &[u32],
    num_reactions: u32,
    va: u32,
    vb: u32,
    mat_a: u32,
    mat_b: u32,
    rng: u32,
) -> (u32, u32, bool) {
    let mut i: u32 = 0;
    loop {
        if i >= num_reactions {
            return (va, vb, false);
        }

        let r_input_a = ca_types::reaction_input_a(reactions, i);
        let r_input_b = ca_types::reaction_input_b(reactions, i);

        if r_input_a == mat_b && r_input_b == mat_a {
            return apply_reaction_if_probable(reactions, i, vb, va, rng, true);
        }

        i += 1;
    }
}

/// Applies a reaction if the RNG passes the probability check.
///
/// `swapped` indicates whether the inputs were provided in reversed order.
fn apply_reaction_if_probable(
    reactions: &[u32],
    idx: u32,
    va_matched: u32,
    vb_matched: u32,
    rng: u32,
    swapped: bool,
) -> (u32, u32, bool) {
    let prob = ca_types::reaction_probability(reactions, idx);
    if (rng & 0xFF) >= prob {
        if swapped {
            return (vb_matched, va_matched, false);
        }
        return (va_matched, vb_matched, false);
    }

    let output_a_mat = ca_types::reaction_output_a(reactions, idx);
    let output_b_mat = ca_types::reaction_output_b(reactions, idx);
    let heat_delta = ca_types::reaction_heat_delta(reactions, idx);

    let new_a = apply_reaction_outputs(va_matched, output_a_mat, heat_delta);
    let new_b = apply_reaction_outputs(vb_matched, output_b_mat, heat_delta);

    if swapped {
        (new_b, new_a, true)
    } else {
        (new_a, new_b, true)
    }
}

/// Applies reaction outputs to a single voxel: sets material and adjusts temperature.
fn apply_reaction_outputs(voxel: u32, new_mat: u32, heat_delta: i32) -> u32 {
    let mut v = ca_types::voxel_set_material(voxel, new_mat);
    // Clear bonds on material change
    v = ca_types::voxel_set_bonds(v, 0);
    // Apply heat delta
    let old_temp = ca_types::voxel_temperature(v) as i32;
    let new_temp = old_temp + heat_delta;
    let clamped = if new_temp < 0 {
        0u32
    } else if new_temp > 255 {
        255u32
    } else {
        new_temp as u32
    };
    ca_types::voxel_set_temperature(v, clamped)
}

// ---- Horizontal spreading helpers ----

/// Checks if a voxel is a fluid (liquid or gas) that can spread laterally.
fn is_spreadable(materials: &[u32], mat_id: u32) -> bool {
    if mat_id == 0 {
        return false;
    }
    let phase = ca_types::mat_phase(materials, mat_id);
    // liquid=2, gas=3 can spread; powder=1 can also spread (sand slides)
    phase >= 1
}

/// Tries to swap two horizontal neighbors: spreads fluid into air.
///
/// Returns (new_a, new_b). Swaps if `a` is spreadable and `b` is air,
/// or if `a` is heavier fluid and `b` is lighter fluid.
fn try_horizontal_swap(materials: &[u32], a: u32, b: u32) -> (u32, u32) {
    let mat_a = ca_types::voxel_material_id(a);
    let mat_b = ca_types::voxel_material_id(b);

    // Spread into air: fluid moves into adjacent air cell
    // Only one direction: a is fluid, b is air → a moves into b
    if mat_b == 0 && is_spreadable(materials, mat_a) {
        let phase_a = ca_types::mat_phase(materials, mat_a);
        if phase_a >= 2 {
            return (b, a); // swap: fluid goes to b, air goes to a
        }
    }
    // Reverse: b is fluid, a is air → b moves into a
    if mat_a == 0 && is_spreadable(materials, mat_b) {
        let phase_b = ca_types::mat_phase(materials, mat_b);
        if phase_b >= 2 {
            return (b, a); // swap: fluid goes to a, air goes to b
        }
    }

    (a, b)
}

/// Applies horizontal spreading in the bottom layer of the 2x2x2 block.
///
/// Separated to keep branch count low per function (trap #15).
/// Uses a deterministic RNG to choose spread direction, preventing bias.
fn apply_horizontal_spread_bottom(
    materials: &[u32],
    v000: &mut u32,
    v100: &mut u32,
    v001: &mut u32,
    v101: &mut u32,
    rng: u32,
) {
    // Alternate between X-axis and Z-axis spreading based on rng
    if (rng & 1) == 0 {
        // X-axis first, then Z-axis
        let (a, b) = try_horizontal_swap(materials, *v000, *v100);
        *v000 = a;
        *v100 = b;
        let (a, b) = try_horizontal_swap(materials, *v001, *v101);
        *v001 = a;
        *v101 = b;
        // Z-axis
        let (a, b) = try_horizontal_swap(materials, *v000, *v001);
        *v000 = a;
        *v001 = b;
        let (a, b) = try_horizontal_swap(materials, *v100, *v101);
        *v100 = a;
        *v101 = b;
    } else {
        // Z-axis first, then X-axis
        let (a, b) = try_horizontal_swap(materials, *v000, *v001);
        *v000 = a;
        *v001 = b;
        let (a, b) = try_horizontal_swap(materials, *v100, *v101);
        *v100 = a;
        *v101 = b;
        // X-axis
        let (a, b) = try_horizontal_swap(materials, *v000, *v100);
        *v000 = a;
        *v100 = b;
        let (a, b) = try_horizontal_swap(materials, *v001, *v101);
        *v001 = a;
        *v101 = b;
    }
}

/// Applies horizontal spreading in the top layer of the 2x2x2 block.
fn apply_horizontal_spread_top(
    materials: &[u32],
    v010: &mut u32,
    v110: &mut u32,
    v011: &mut u32,
    v111: &mut u32,
    rng: u32,
) {
    if (rng & 2) == 0 {
        let (a, b) = try_horizontal_swap(materials, *v010, *v110);
        *v010 = a;
        *v110 = b;
        let (a, b) = try_horizontal_swap(materials, *v011, *v111);
        *v011 = a;
        *v111 = b;
        let (a, b) = try_horizontal_swap(materials, *v010, *v011);
        *v010 = a;
        *v011 = b;
        let (a, b) = try_horizontal_swap(materials, *v110, *v111);
        *v110 = a;
        *v111 = b;
    } else {
        let (a, b) = try_horizontal_swap(materials, *v010, *v011);
        *v010 = a;
        *v011 = b;
        let (a, b) = try_horizontal_swap(materials, *v110, *v111);
        *v110 = a;
        *v111 = b;
        let (a, b) = try_horizontal_swap(materials, *v010, *v110);
        *v010 = a;
        *v110 = b;
        let (a, b) = try_horizontal_swap(materials, *v011, *v111);
        *v011 = a;
        *v111 = b;
    }
}

// ---- Main block processing ----

/// Processes one 2x2x2 Margolus block: applies gravity, spreading, and reactions.
///
/// This is the main orchestration function called from the entry point.
fn margolus_main(
    chunk_pool: &mut [u32],
    materials: &[u32],
    reactions: &[u32],
    slot_id: u32,
    bx: u32,
    by: u32,
    bz: u32,
    frame: u32,
    num_reactions: u32,
) {
    // Read 8 voxels of the 2x2x2 block (separate variables, not array — trap #21)
    let mut v000 = read_voxel(chunk_pool, slot_id, bx, by, bz);
    let mut v100 = read_voxel(chunk_pool, slot_id, bx + 1, by, bz);
    let mut v010 = read_voxel(chunk_pool, slot_id, bx, by + 1, bz);
    let mut v110 = read_voxel(chunk_pool, slot_id, bx + 1, by + 1, bz);
    let mut v001 = read_voxel(chunk_pool, slot_id, bx, by, bz + 1);
    let mut v101 = read_voxel(chunk_pool, slot_id, bx + 1, by, bz + 1);
    let mut v011 = read_voxel(chunk_pool, slot_id, bx, by + 1, bz + 1);
    let mut v111 = read_voxel(chunk_pool, slot_id, bx + 1, by + 1, bz + 1);

    // --- Step 1: Apply gravity (falling sand) ---
    // Process 4 vertical column pairs: y=1 (top) vs y=0 (bottom)
    let (new_top, new_bot) = apply_gravity_pair(materials, v010, v000);
    v010 = new_top;
    v000 = new_bot;

    let (new_top, new_bot) = apply_gravity_pair(materials, v110, v100);
    v110 = new_top;
    v100 = new_bot;

    let (new_top, new_bot) = apply_gravity_pair(materials, v011, v001);
    v011 = new_top;
    v001 = new_bot;

    let (new_top, new_bot) = apply_gravity_pair(materials, v111, v101);
    v111 = new_top;
    v101 = new_bot;

    // --- Step 2: Apply gas buoyancy ---
    let (new_top, new_bot) = apply_buoyancy_pair(materials, v010, v000);
    v010 = new_top;
    v000 = new_bot;

    let (new_top, new_bot) = apply_buoyancy_pair(materials, v110, v100);
    v110 = new_top;
    v100 = new_bot;

    let (new_top, new_bot) = apply_buoyancy_pair(materials, v011, v001);
    v011 = new_top;
    v001 = new_bot;

    let (new_top, new_bot) = apply_buoyancy_pair(materials, v111, v101);
    v111 = new_top;
    v101 = new_bot;

    // --- Step 3: Horizontal spreading (disabled: causes water loss bug) ---
    // TODO: fix conservation issue in horizontal spread before re-enabling
    // let rng = ca_types::hash_position(bx, by, bz, frame);
    // apply_horizontal_spread_bottom(materials, &mut v000, &mut v100, &mut v001, &mut v101, rng);
    // apply_horizontal_spread_top(materials, &mut v010, &mut v110, &mut v011, &mut v111, rng);
    let _ = ca_types::hash_position(bx, by, bz, frame); // suppress unused warning

    // --- Step 4: Apply chemical reactions on edges ---
    apply_block_reactions(
        &mut v000, &mut v100, &mut v010, &mut v110,
        &mut v001, &mut v101, &mut v011, &mut v111,
        reactions, num_reactions, bx, by, bz, frame,
    );

    // --- Write back 8 voxels ---
    write_voxel(chunk_pool, slot_id, bx, by, bz, v000);
    write_voxel(chunk_pool, slot_id, bx + 1, by, bz, v100);
    write_voxel(chunk_pool, slot_id, bx, by + 1, bz, v010);
    write_voxel(chunk_pool, slot_id, bx + 1, by + 1, bz, v110);
    write_voxel(chunk_pool, slot_id, bx, by, bz + 1, v001);
    write_voxel(chunk_pool, slot_id, bx + 1, by, bz + 1, v101);
    write_voxel(chunk_pool, slot_id, bx, by + 1, bz + 1, v011);
    write_voxel(chunk_pool, slot_id, bx + 1, by + 1, bz + 1, v111);
}

/// Applies chemical reactions along the 12 edges of the 2x2x2 block.
///
/// Separated to keep branch count low per function (trap #15).
#[allow(clippy::too_many_arguments)]
fn apply_block_reactions(
    v000: &mut u32, v100: &mut u32, v010: &mut u32, v110: &mut u32,
    v001: &mut u32, v101: &mut u32, v011: &mut u32, v111: &mut u32,
    reactions: &[u32],
    num_reactions: u32,
    bx: u32,
    by: u32,
    bz: u32,
    frame: u32,
) {
    if num_reactions == 0 {
        return;
    }

    // X-axis edges (4 edges)
    apply_edge_reactions_x(v000, v100, reactions, num_reactions, bx, by, bz, frame);
    apply_edge_reactions_x(v010, v110, reactions, num_reactions, bx, by + 1, bz, frame);
    apply_edge_reactions_x(v001, v101, reactions, num_reactions, bx, by, bz + 1, frame);
    apply_edge_reactions_x(v011, v111, reactions, num_reactions, bx, by + 1, bz + 1, frame);

    // Y-axis edges (4 edges)
    apply_edge_reactions_y(v000, v010, reactions, num_reactions, bx, by, bz, frame);
    apply_edge_reactions_y(v100, v110, reactions, num_reactions, bx + 1, by, bz, frame);
    apply_edge_reactions_y(v001, v011, reactions, num_reactions, bx, by, bz + 1, frame);
    apply_edge_reactions_y(v101, v111, reactions, num_reactions, bx + 1, by, bz + 1, frame);

    // Z-axis edges (4 edges)
    apply_edge_reactions_z(v000, v001, reactions, num_reactions, bx, by, bz, frame);
    apply_edge_reactions_z(v100, v101, reactions, num_reactions, bx + 1, by, bz, frame);
    apply_edge_reactions_z(v010, v011, reactions, num_reactions, bx, by + 1, bz, frame);
    apply_edge_reactions_z(v110, v111, reactions, num_reactions, bx + 1, by + 1, bz, frame);
}

/// Applies reactions between two X-adjacent voxels.
fn apply_edge_reactions_x(
    va: &mut u32, vb: &mut u32,
    reactions: &[u32], num_reactions: u32,
    x: u32, y: u32, z: u32, frame: u32,
) {
    let rng = ca_types::hash_position(x, y, z, frame);
    let (new_a, new_b, _) = try_reactions(reactions, num_reactions, *va, *vb, rng);
    *va = new_a;
    *vb = new_b;
}

/// Applies reactions between two Y-adjacent voxels.
fn apply_edge_reactions_y(
    va: &mut u32, vb: &mut u32,
    reactions: &[u32], num_reactions: u32,
    x: u32, y: u32, z: u32, frame: u32,
) {
    let rng = ca_types::hash_position(x, y, z, frame.wrapping_add(7919));
    let (new_a, new_b, _) = try_reactions(reactions, num_reactions, *va, *vb, rng);
    *va = new_a;
    *vb = new_b;
}

/// Applies reactions between two Z-adjacent voxels.
fn apply_edge_reactions_z(
    va: &mut u32, vb: &mut u32,
    reactions: &[u32], num_reactions: u32,
    x: u32, y: u32, z: u32, frame: u32,
) {
    let rng = ca_types::hash_position(x, y, z, frame.wrapping_add(15727));
    let (new_a, new_b, _) = try_reactions(reactions, num_reactions, *va, *vb, rng);
    *va = new_a;
    *vb = new_b;
}

/// Compute shader entry point: Margolus block automaton for falling sand + reactions.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool (voxel gigabuffer, read/write)
/// - binding 1: metadata (ChunkGpuMeta array as `&[u32]`, read)
/// - binding 2: dirty_list (dirty chunk IDs with header, read)
/// - binding 3: materials (MaterialPropertiesCA array as `&[u32]`, read)
/// - binding 4: reactions (ReactionEntry array as `&[u32]`, read)
///
/// Dispatched indirectly: X = dirty_count * 64, Y = 1, Z = 1.
#[spirv(compute(threads(4, 4, 4)))]
pub fn ca_margolus(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dirty_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] reactions: &[u32],
    #[spirv(push_constant)] push: &CaMargolusPush,
) {
    margolus_entry(
        local_id, wg_id,
        chunk_pool, metadata, dirty_list, materials, reactions,
        push.frame_number, push.offset_x, push.offset_y, push.offset_z,
        push.num_reactions,
    );
}

/// Entry-point orchestration (helper for trap #4a compliance).
#[allow(clippy::too_many_arguments)]
fn margolus_entry(
    local_id: UVec3,
    wg_id: UVec3,
    chunk_pool: &mut [u32],
    metadata: &[u32],
    dirty_list: &[u32],
    materials: &[u32],
    reactions: &[u32],
    frame_number: u32,
    offset_x: u32,
    offset_y: u32,
    offset_z: u32,
    num_reactions: u32,
) {
    let wg_per_chunk = ca_types::MARGOLUS_WG_PER_CHUNK; // 64
    let chunk_index = wg_id.x / wg_per_chunk;
    let local_wg = wg_id.x % wg_per_chunk;

    // Read dirty count from header
    let dirty_count = dirty_list[0];
    if chunk_index >= dirty_count {
        return;
    }

    // Get slot_id
    let slot_id = dirty_list[(DIRTY_LIST_HEADER + chunk_index) as usize];

    // Decompose local_wg into 3D workgroup position within chunk
    // 64 workgroups = 4x4x4 workgroups, each covering 4x4x4 threads = 4x4x4 blocks
    let wg_z = local_wg / 16;
    let wg_y = (local_wg / 4) % 4;
    let wg_x = local_wg % 4;

    // Each thread handles one 2x2x2 block
    // Block coordinates within chunk (in voxel units, even-aligned)
    let block_x = (wg_x * 4 + local_id.x) * 2 + offset_x;
    let block_y = (wg_y * 4 + local_id.y) * 2 + offset_y;
    let block_z = (wg_z * 4 + local_id.z) * 2 + offset_z;

    // Bounds check: with odd offset, some blocks extend past chunk boundary
    // Need block + 1 < 32, so block must be <= 30
    if block_x + 1 >= ca_types::CA_CHUNK_SIZE {
        return;
    }
    if block_y + 1 >= ca_types::CA_CHUNK_SIZE {
        return;
    }
    if block_z + 1 >= ca_types::CA_CHUNK_SIZE {
        return;
    }

    // Suppress unused metadata warning (reserved for cross-chunk Margolus in future)
    let _ = metadata[0];

    margolus_main(
        chunk_pool, materials, reactions,
        slot_id, block_x, block_y, block_z,
        frame_number, num_reactions,
    );
}
