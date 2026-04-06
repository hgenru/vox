//! CPU reference implementation of Cellular Automata operations.
//!
//! [`CaGrid`] is the test oracle for GPU CA shaders. It implements thermal
//! diffusion, Margolus falling-sand, chemical reactions, and phase transitions
//! using the exact same integer arithmetic the GPU will use, so results must
//! be bit-identical for the same input.

use shared::constants::{CA_AMBIENT_TEMP, CA_CHUNK_SIZE, CA_CHUNK_VOXELS};
use shared::material_ca::MaterialPropertiesCA;
use shared::reaction_ca::ReactionEntry;
use shared::voxel::Voxel;
use std::collections::HashMap;

/// CPU-side CA world for testing. Stores chunks as arrays of packed voxels.
pub struct CaGrid {
    /// Chunk data: coord -> 32^3 packed voxels.
    chunks: HashMap<[i32; 3], Vec<u32>>,
    /// Material properties table.
    materials: Vec<MaterialPropertiesCA>,
    /// Chemical reaction table.
    reactions: Vec<ReactionEntry>,
}

impl CaGrid {
    /// Creates a new CA grid with the given material and reaction tables.
    pub fn new(materials: Vec<MaterialPropertiesCA>, reactions: Vec<ReactionEntry>) -> Self {
        Self {
            chunks: HashMap::new(),
            materials,
            reactions,
        }
    }

    /// Returns the voxel at the given world position.
    ///
    /// Returns air (0) if the containing chunk is not loaded.
    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32) -> Voxel {
        let (chunk, idx) = Self::world_to_chunk_local(wx, wy, wz);
        match self.chunks.get(&chunk) {
            Some(data) => Voxel(data[idx]),
            None => Voxel::air(),
        }
    }

    /// Sets the voxel at the given world position.
    ///
    /// Creates the chunk (filled with air) if it does not exist yet.
    pub fn set_voxel(&mut self, wx: i32, wy: i32, wz: i32, voxel: Voxel) {
        let (chunk, idx) = Self::world_to_chunk_local(wx, wy, wz);
        let data = self
            .chunks
            .entry(chunk)
            .or_insert_with(|| vec![0u32; CA_CHUNK_VOXELS]);
        data[idx] = voxel.0;
    }

    /// Loads a chunk from raw packed-u32 data.
    ///
    /// `data` must contain exactly `CA_CHUNK_VOXELS` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != CA_CHUNK_VOXELS`.
    pub fn load_chunk(&mut self, cx: i32, cy: i32, cz: i32, data: Vec<u32>) {
        assert_eq!(
            data.len(),
            CA_CHUNK_VOXELS,
            "chunk data must be exactly {} u32s",
            CA_CHUNK_VOXELS
        );
        self.chunks.insert([cx, cy, cz], data);
    }

    /// Returns the raw packed data for the given chunk, if loaded.
    pub fn get_chunk_data(&self, cx: i32, cy: i32, cz: i32) -> Option<&[u32]> {
        self.chunks.get(&[cx, cy, cz]).map(|v| v.as_slice())
    }

    /// Converts world coordinates to (chunk_coord, local_flat_index).
    ///
    /// Uses Euclidean division so negative coordinates are handled correctly.
    /// Local index is Z-major: `lz * 32 * 32 + ly * 32 + lx`.
    fn world_to_chunk_local(wx: i32, wy: i32, wz: i32) -> ([i32; 3], usize) {
        let cs = CA_CHUNK_SIZE as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        let idx = lz * CA_CHUNK_SIZE * CA_CHUNK_SIZE + ly * CA_CHUNK_SIZE + lx;
        ([cx, cy, cz], idx)
    }

    /// Runs one full CA step: thermal diffusion, then even Margolus, then odd Margolus.
    pub fn step(&mut self, frame: u32) {
        self.thermal_step();
        self.margolus_step([0, 0, 0], frame);
        self.margolus_step([1, 1, 1], frame);
    }

    /// Diffuses temperature across all loaded chunks using integer math.
    ///
    /// Reads from a snapshot of the old state and writes new temperatures in-place.
    /// Also applies phase transitions after updating temperature.
    pub fn thermal_step(&mut self) {
        // Snapshot the old state for double-buffering reads.
        let old_chunks: HashMap<[i32; 3], Vec<u32>> = self.chunks.clone();

        // Helper: read voxel from old state.
        let get_old = |wx: i32, wy: i32, wz: i32| -> Voxel {
            let (chunk, idx) = Self::world_to_chunk_local(wx, wy, wz);
            match old_chunks.get(&chunk) {
                Some(data) => Voxel(data[idx]),
                None => {
                    // Unloaded neighbor: air at ambient temperature.
                    Voxel::new(0, CA_AMBIENT_TEMP, 0, 0, 0)
                }
            }
        };

        // Collect chunk keys so we can iterate without borrowing self.
        let chunk_keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();

        for chunk_coord in chunk_keys {
            let cs = CA_CHUNK_SIZE as i32;
            let base_x = chunk_coord[0] * cs;
            let base_y = chunk_coord[1] * cs;
            let base_z = chunk_coord[2] * cs;

            for lz in 0..CA_CHUNK_SIZE {
                for ly in 0..CA_CHUNK_SIZE {
                    for lx in 0..CA_CHUNK_SIZE {
                        let wx = base_x + lx as i32;
                        let wy = base_y + ly as i32;
                        let wz = base_z + lz as i32;

                        let old_voxel = get_old(wx, wy, wz);
                        if old_voxel.is_air() {
                            continue;
                        }

                        let mat_id = old_voxel.material_id() as usize;
                        if mat_id >= self.materials.len() {
                            continue;
                        }

                        let conductivity = self.materials[mat_id].conductivity as i32;
                        if conductivity == 0 {
                            // Insulator: skip diffusion but still check phase transitions.
                            let my_temp = old_voxel.temperature();
                            self.apply_phase_transition(chunk_coord, lz, ly, lx, my_temp);
                            continue;
                        }

                        let my_temp = old_voxel.temperature() as i32;

                        // Read 6 neighbors from old state.
                        let n0 = get_old(wx - 1, wy, wz).temperature() as i32;
                        let n1 = get_old(wx + 1, wy, wz).temperature() as i32;
                        let n2 = get_old(wx, wy - 1, wz).temperature() as i32;
                        let n3 = get_old(wx, wy + 1, wz).temperature() as i32;
                        let n4 = get_old(wx, wy, wz - 1).temperature() as i32;
                        let n5 = get_old(wx, wy, wz + 1).temperature() as i32;

                        let avg_temp = (n0 + n1 + n2 + n3 + n4 + n5) / 6; // integer division
                        let delta = (conductivity * (avg_temp - my_temp)) >> 8; // arithmetic shift
                        let new_temp = (my_temp + delta).clamp(0, 255) as u8;

                        let idx = lz * CA_CHUNK_SIZE * CA_CHUNK_SIZE + ly * CA_CHUNK_SIZE + lx;
                        let data = self.chunks.get_mut(&chunk_coord).expect("chunk must exist");
                        let mut voxel = Voxel(data[idx]);
                        voxel.set_temperature(new_temp);
                        data[idx] = voxel.0;

                        // Phase transitions based on new temperature.
                        self.apply_phase_transition(chunk_coord, lz, ly, lx, new_temp);
                    }
                }
            }
        }
    }

    /// Processes all 2x2x2 Margolus blocks at the given offset.
    ///
    /// `offset` is `[0,0,0]` for even frames and `[1,1,1]` for odd frames.
    pub fn margolus_step(&mut self, offset: [i32; 3], frame: u32) {
        // Collect all world-space block origins that overlap loaded chunks.
        let chunk_keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();
        let mut block_origins: Vec<[i32; 3]> = Vec::new();

        let cs = CA_CHUNK_SIZE as i32;
        for chunk_coord in &chunk_keys {
            let base_x = chunk_coord[0] * cs;
            let base_y = chunk_coord[1] * cs;
            let base_z = chunk_coord[2] * cs;

            // Blocks start at offset within each chunk, stepping by 2.
            let mut bz = base_z + offset[2];
            while bz + 1 < base_z + cs {
                let mut by = base_y + offset[1];
                while by + 1 < base_y + cs {
                    let mut bx = base_x + offset[0];
                    while bx + 1 < base_x + cs {
                        block_origins.push([bx, by, bz]);
                        bx += 2;
                    }
                    by += 2;
                }
                bz += 2;
            }
        }

        // Sort for determinism.
        block_origins.sort();
        block_origins.dedup();

        for origin in block_origins {
            self.process_margolus_block(origin, frame);
        }
    }

    /// Processes a single 2x2x2 Margolus block.
    ///
    /// Applies gravity/falling-sand rules, then chemical reactions.
    fn process_margolus_block(&mut self, origin: [i32; 3], frame: u32) {
        let [ox, oy, oz] = origin;

        // The 8 positions in the 2x2x2 block (x,y,z offsets from origin).
        // Listed bottom-to-top (y=0 first, then y=1) for gravity processing.
        let offsets: [[i32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
        ];

        // Read 8 voxels into local array.
        let mut voxels: [Voxel; 8] = [Voxel::air(); 8];
        let mut positions: [[i32; 3]; 8] = [[0; 3]; 8];
        for (i, off) in offsets.iter().enumerate() {
            let wx = ox + off[0];
            let wy = oy + off[1];
            let wz = oz + off[2];
            voxels[i] = self.get_voxel(wx, wy, wz);
            positions[i] = [wx, wy, wz];
        }

        // --- Gravity / falling sand ---
        // Process pairs: each voxel in top layer (y=1, indices 4-7) tries to swap
        // with the one directly below (y=0, indices 0-3).
        for i in 0..4 {
            let top_idx = i + 4; // y=1
            let bot_idx = i; // y=0
            self.try_gravity_swap(&mut voxels, top_idx, bot_idx);
        }

        // --- Chemical reactions ---
        // 12 edges in a 2x2x2 cube.
        let edges: [[usize; 2]; 12] = [
            // Bottom face (y=0)
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            // Top face (y=1)
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            // Vertical edges
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];

        for [a, b] in edges {
            self.try_reaction(&mut voxels, a, b, &positions, frame);
        }

        // Write 8 voxels back.
        for (i, off) in offsets.iter().enumerate() {
            let wx = ox + off[0];
            let wy = oy + off[1];
            let wz = oz + off[2];
            self.set_voxel(wx, wy, wz, voxels[i]);
        }
    }

    /// Tries to swap two voxels based on gravity rules.
    ///
    /// `top` is the index of the upper voxel, `bot` the lower one.
    /// - Powder over air/lighter-liquid: swap (sand falls).
    /// - Liquid over air: swap (water falls).
    /// - Gas under air/lighter-liquid: swap (steam rises).
    fn try_gravity_swap(&self, voxels: &mut [Voxel; 8], top: usize, bot: usize) {
        let top_v = voxels[top];
        let bot_v = voxels[bot];

        let top_mat = top_v.material_id() as usize;
        let bot_mat = bot_v.material_id() as usize;

        let top_phase = self.get_phase(top_mat);
        let bot_phase = self.get_phase(bot_mat);
        let top_density = self.get_density(top_mat);
        let bot_density = self.get_density(bot_mat);

        // Powder (phase=1) falls through air (mat=0) or lighter liquid.
        if top_phase == 1 && (bot_v.is_air() || (bot_phase == 2 && bot_density < top_density)) {
            voxels[top] = bot_v;
            voxels[bot] = top_v;
            return;
        }

        // Liquid (phase=2) falls through air.
        if top_phase == 2 && bot_v.is_air() {
            voxels[top] = bot_v;
            voxels[bot] = top_v;
            return;
        }

        // Liquid falls through lighter liquid.
        if top_phase == 2 && bot_phase == 2 && top_density > bot_density {
            voxels[top] = bot_v;
            voxels[bot] = top_v;
            return;
        }

        // Gas (phase=3) rises through air or lighter material above it.
        // Here top is above bot, so gas rising means: bot is gas, top is air/lighter.
        if bot_phase == 3
            && (top_v.is_air() || (top_phase == 2 && top_density > bot_density))
        {
            voxels[top] = bot_v;
            voxels[bot] = top_v;
        }
    }

    /// Attempts a chemical reaction between two adjacent voxels in a Margolus block.
    fn try_reaction(
        &self,
        voxels: &mut [Voxel; 8],
        a: usize,
        b: usize,
        positions: &[[i32; 3]; 8],
        frame: u32,
    ) {
        let va = voxels[a];
        let vb = voxels[b];
        let mat_a = va.material_id() as u32;
        let mat_b = vb.material_id() as u32;

        // Look up reaction in both orderings.
        for reaction in &self.reactions {
            let (idx_a, idx_b, _flipped) =
                if reaction.input_a == mat_a && reaction.input_b == mat_b {
                    (a, b, false)
                } else if reaction.input_a == mat_b && reaction.input_b == mat_a {
                    (b, a, true)
                } else {
                    continue;
                };

            let va_actual = voxels[idx_a];
            let vb_actual = voxels[idx_b];

            // Check condition on voxel A (the one matching input_a).
            let condition_met = match reaction.condition {
                0 => true,
                1 => va_actual.oxygen() > 0,
                2 => {
                    let mat_id = va_actual.material_id() as usize;
                    if mat_id < self.materials.len() {
                        (va_actual.temperature() as u32) > self.materials[mat_id].ignition_temp
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if !condition_met {
                continue;
            }

            // Stochastic check.
            let [px, py, pz] = positions[a];
            let h = hash_position(px, py, pz, frame);
            if (h % 256) >= reaction.probability {
                continue;
            }

            // Apply reaction.
            let mut new_a = va_actual;
            new_a.set_material_id(reaction.output_a as u16);

            let mut new_b = vb_actual;
            new_b.set_material_id(reaction.output_b as u16);

            // Apply heat delta to both voxels.
            if reaction.heat_delta != 0 {
                let ta = new_a.temperature() as i32 + reaction.heat_delta;
                new_a.set_temperature(ta.clamp(0, 255) as u8);
                let tb = new_b.temperature() as i32 + reaction.heat_delta;
                new_b.set_temperature(tb.clamp(0, 255) as u8);
            }

            // If reaction consumed oxygen, decrement.
            if reaction.condition == 1 {
                let oxy = new_a.oxygen().saturating_sub(1);
                new_a.set_oxygen(oxy);
            }

            voxels[idx_a] = new_a;
            voxels[idx_b] = new_b;

            // Only apply one reaction per edge per step.
            return;
        }
    }

    /// Returns the phase of a material (0 for air/unknown).
    fn get_phase(&self, mat_id: usize) -> u32 {
        if mat_id == 0 || mat_id >= self.materials.len() {
            return 0;
        }
        self.materials[mat_id].phase
    }

    /// Returns the density of a material (0 for air/unknown).
    fn get_density(&self, mat_id: usize) -> u32 {
        if mat_id == 0 || mat_id >= self.materials.len() {
            return 0;
        }
        self.materials[mat_id].density
    }

    /// Applies phase transition rules to a voxel after its temperature was updated.
    ///
    /// Checks melt, boil, and freeze thresholds. A threshold of 0 means "disabled".
    fn apply_phase_transition(
        &mut self,
        chunk_coord: [i32; 3],
        lz: usize,
        ly: usize,
        lx: usize,
        _new_temp: u8,
    ) {
        let idx = lz * CA_CHUNK_SIZE * CA_CHUNK_SIZE + ly * CA_CHUNK_SIZE + lx;
        let data = self.chunks.get_mut(&chunk_coord).expect("chunk must exist");
        let mut voxel = Voxel(data[idx]);
        let mat_id = voxel.material_id() as usize;
        if mat_id == 0 || mat_id >= self.materials.len() {
            return;
        }
        let mat = &self.materials[mat_id];
        let temp = voxel.temperature() as u32;

        // Melting.
        if mat.melt_temp > 0 && temp > mat.melt_temp && mat.melt_into > 0 {
            voxel.set_material_id(mat.melt_into as u16);
            voxel.set_bonds(0);
            data[idx] = voxel.0;
            return;
        }

        // Boiling.
        if mat.boil_temp > 0 && temp > mat.boil_temp && mat.boil_into > 0 {
            voxel.set_material_id(mat.boil_into as u16);
            voxel.set_bonds(0);
            data[idx] = voxel.0;
            return;
        }

        // Freezing.
        if mat.freeze_temp > 0 && temp < mat.freeze_temp && mat.freeze_into > 0 {
            voxel.set_material_id(mat.freeze_into as u16);
            data[idx] = voxel.0;
        }
    }
}

/// Deterministic spatial hash for stochastic reaction firing.
///
/// Must be identical on CPU and GPU for bit-exact results.
pub fn hash_position(x: i32, y: i32, z: i32, frame: u32) -> u32 {
    let mut h = (x as u32)
        .wrapping_mul(73856093)
        ^ (y as u32).wrapping_mul(19349663)
        ^ (z as u32).wrapping_mul(83492791)
        ^ frame.wrapping_mul(48611);
    h = h.wrapping_mul(2654435761);
    h ^= h >> 16;
    h
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a zeroed `MaterialPropertiesCA`.
    fn zeroed_mat() -> MaterialPropertiesCA {
        MaterialPropertiesCA {
            phase: 0,
            density: 0,
            viscosity: 0,
            conductivity: 0,
            melt_temp: 0,
            boil_temp: 0,
            freeze_temp: 0,
            melt_into: 0,
            boil_into: 0,
            freeze_into: 0,
            flammability: 0,
            ignition_temp: 0,
            burn_into: 0,
            burn_heat: 0,
            strength: 0,
            max_load: 0,
        }
    }

    /// Creates a test material table with common materials.
    ///
    /// Index 0 = Air, 1 = Stone, 2 = Sand, 3 = Water, 4 = Lava,
    /// 5 = Steam, 6 = Ice, 7 = Wood, 8 = Fire, 9 = Ash.
    fn test_materials() -> Vec<MaterialPropertiesCA> {
        let mut mats = vec![zeroed_mat(); 10];

        // 0 = Air (all zeros, default)

        // 1 = Stone (solid, high melt_temp)
        mats[1].phase = 0;
        mats[1].density = 200;
        mats[1].conductivity = 2;
        mats[1].melt_temp = 200;
        mats[1].melt_into = 4; // becomes lava
        mats[1].strength = 100;

        // 2 = Sand (powder)
        mats[2].phase = 1;
        mats[2].density = 150;
        mats[2].conductivity = 1;
        mats[2].melt_temp = 220;
        mats[2].melt_into = 3; // becomes "glass"/water for simplicity

        // 3 = Water (liquid)
        mats[3].phase = 2;
        mats[3].density = 100;
        mats[3].conductivity = 4;
        mats[3].freeze_temp = 50;
        mats[3].freeze_into = 6; // becomes ice
        mats[3].boil_temp = 200;
        mats[3].boil_into = 5; // becomes steam

        // 4 = Lava (liquid, high temp)
        mats[4].phase = 2;
        mats[4].density = 200;
        mats[4].conductivity = 8;
        mats[4].freeze_temp = 150;
        mats[4].freeze_into = 1; // becomes stone

        // 5 = Steam (gas)
        mats[5].phase = 3;
        mats[5].density = 10;
        mats[5].conductivity = 1;
        mats[5].freeze_temp = 100;
        mats[5].freeze_into = 3; // becomes water

        // 6 = Ice (solid)
        mats[6].phase = 0;
        mats[6].density = 90;
        mats[6].conductivity = 3;
        mats[6].melt_temp = 50;
        mats[6].melt_into = 3; // becomes water

        // 7 = Wood (solid, flammable)
        mats[7].phase = 0;
        mats[7].density = 80;
        mats[7].conductivity = 1;
        mats[7].flammability = 200;
        mats[7].ignition_temp = 150;
        mats[7].burn_into = 9; // becomes ash
        mats[7].burn_heat = 20;

        // 8 = Fire (gas)
        mats[8].phase = 3;
        mats[8].density = 5;
        mats[8].conductivity = 10;
        mats[8].burn_heat = 20;

        // 9 = Ash (powder)
        mats[9].phase = 1;
        mats[9].density = 50;
        mats[9].conductivity = 1;

        mats
    }

    /// Creates a reaction table for testing.
    fn test_reactions() -> Vec<ReactionEntry> {
        vec![
            // Water + Lava -> Stone + Steam (always, 100% probability)
            ReactionEntry {
                input_a: 3,  // water
                input_b: 4,  // lava
                output_a: 1, // stone
                output_b: 5, // steam
                heat_delta: -20,
                condition: 0,    // always
                probability: 255, // ~100%
                impulse: 0,
            },
        ]
    }

    #[test]
    fn test_world_to_chunk_local() {
        // Positive coords.
        let (chunk, idx) = CaGrid::world_to_chunk_local(0, 0, 0);
        assert_eq!(chunk, [0, 0, 0]);
        assert_eq!(idx, 0);

        let (chunk, idx) = CaGrid::world_to_chunk_local(31, 31, 31);
        assert_eq!(chunk, [0, 0, 0]);
        assert_eq!(idx, 31 * 32 * 32 + 31 * 32 + 31);

        let (chunk, _) = CaGrid::world_to_chunk_local(32, 0, 0);
        assert_eq!(chunk, [1, 0, 0]);

        // Negative coords: -1 should be in chunk [-1, -1, -1], local [31,31,31].
        let (chunk, idx) = CaGrid::world_to_chunk_local(-1, -1, -1);
        assert_eq!(chunk, [-1, -1, -1]);
        assert_eq!(idx, 31 * 32 * 32 + 31 * 32 + 31);
    }

    #[test]
    fn test_get_set_voxel_roundtrip() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());
        let v = Voxel::new(3, 128, 0, 5, 0);
        grid.set_voxel(10, 20, 30, v);
        assert_eq!(grid.get_voxel(10, 20, 30), v);
    }

    #[test]
    fn test_get_voxel_unloaded_is_air() {
        let grid = CaGrid::new(test_materials(), test_reactions());
        assert!(grid.get_voxel(100, 200, 300).is_air());
    }

    #[test]
    fn test_thermal_diffusion_basic() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Place a hot stone voxel at origin, surrounded by cold stone.
        let hot = Voxel::new(1, 255, 0, 0, 0); // stone, temp=255
        let cold = Voxel::new(1, 0, 0, 0, 0); // stone, temp=0
        grid.set_voxel(1, 1, 1, hot);

        // Place cold neighbors.
        grid.set_voxel(0, 1, 1, cold);
        grid.set_voxel(2, 1, 1, cold);
        grid.set_voxel(1, 0, 1, cold);
        grid.set_voxel(1, 2, 1, cold);
        grid.set_voxel(1, 1, 0, cold);
        grid.set_voxel(1, 1, 2, cold);

        grid.thermal_step();

        // Center should have cooled down.
        let center = grid.get_voxel(1, 1, 1);
        assert!(
            center.temperature() < 255,
            "center should cool: got {}",
            center.temperature()
        );

        // Verify exact values. Stone conductivity=2.
        // Center: avg_temp of 6 neighbors = 0. delta = (2*(0-255))>>8 = (-510)>>8 = -2
        // (Rust arithmetic right shift of -510 by 8 = -2, since -510 = -2*256 + 2, floor division)
        // new_temp = 255 + (-2) = 253
        assert_eq!(center.temperature(), 253);

        // Neighbor at (0,1,1): conductivity=2 is too low for the >>8 shift to
        // produce a non-zero delta from its neighbors, so it stays at 0. This is
        // correct integer behavior — verified by hand calculation:
        // avg = (128 + 255 + 0 + 0 + 0 + 0)/6 = 63, delta = (2*63)>>8 = 0.
        let neighbor = grid.get_voxel(0, 1, 1);
        assert_eq!(neighbor.temperature(), 0);
    }

    #[test]
    fn test_thermal_diffusion_cross_chunk() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Hot stone at (31, 0, 0) in chunk [0,0,0], cold stone at (32, 0, 0) in chunk [1,0,0].
        let hot = Voxel::new(1, 200, 0, 0, 0);
        let cold = Voxel::new(1, 50, 0, 0, 0);
        grid.set_voxel(31, 0, 0, hot);
        grid.set_voxel(32, 0, 0, cold);

        grid.thermal_step();

        let left = grid.get_voxel(31, 0, 0);
        let right = grid.get_voxel(32, 0, 0);

        // Both should move toward each other.
        // Hot voxel at (31,0,0): neighbors include (32,0,0)=50 and 5 others.
        // (30,0,0)=air in chunk[0,0,0]=temp0, (31,-1,0)=air in chunk[0,-1,0] unloaded=128,
        // (31,1,0)=air in chunk[0,0,0]=0, (31,0,-1)=air in chunk[0,0,-1] unloaded=128,
        // (31,0,1)=air in chunk[0,0,0]=0
        // avg = (0 + 50 + 128 + 0 + 128 + 0)/6 = 306/6 = 51
        // delta = (2 * (51-200))>>8 = (-298)>>8 = -2
        // new = 200-2 = 198
        assert_eq!(left.temperature(), 198);

        // Cold voxel at (32,0,0): neighbors include (31,0,0)=200 and 5 others.
        // (33,0,0)=air in chunk[1,0,0]=0, (32,-1,0)=air in chunk[1,-1,0] unloaded=128,
        // (32,1,0)=air in chunk[1,0,0]=0, (32,0,-1)=air in chunk[1,0,-1] unloaded=128,
        // (32,0,1)=air in chunk[1,0,0]=0
        // avg = (200 + 0 + 128 + 0 + 128 + 0)/6 = 456/6 = 76
        // delta = (2 * (76-50))>>8 = 52>>8 = 0
        // new = 50
        assert_eq!(right.temperature(), 50);
    }

    #[test]
    fn test_thermal_diffusion_unloaded_neighbor() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Place a stone voxel at the edge of a chunk. All 6 neighbors are either
        // air in-chunk or in unloaded chunks (ambient temp = 128).
        // Use (0, 0, 0) with only this single voxel.
        // Neighbors: (-1,0,0) unloaded=128, (1,0,0) air=0, (0,-1,0) unloaded via chunk [0,-1,0]=128,
        // (0,1,0) air=0, (0,0,-1) unloaded=128, (0,0,1) air=0
        // Wait, (0,-1,0): chunk coord = [0, -1, 0], not loaded. So ambient 128.
        // And (0,1,0) is in chunk [0,0,0] which IS loaded -> air -> temp=0.
        let v = Voxel::new(1, 100, 0, 0, 0); // stone, temp=100
        grid.set_voxel(0, 0, 0, v);

        grid.thermal_step();

        // avg = (128 + 0 + 128 + 0 + 128 + 0)/6 = 384/6 = 64
        // delta = (2*(64-100))>>8 = (-72)>>8 = -1
        // new = 100-1 = 99
        let after = grid.get_voxel(0, 0, 0);
        assert_eq!(after.temperature(), 99);
    }

    #[test]
    fn test_falling_sand() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Sand at y=1, air at y=0. After margolus step, sand should fall.
        let sand = Voxel::new(2, 128, 0, 0, 0);
        grid.set_voxel(0, 1, 0, sand);
        // Ensure chunk exists for y=0 level.
        grid.set_voxel(0, 0, 0, Voxel::air());

        // Even offset [0,0,0]: block at origin covers (0,0,0) to (1,1,1).
        grid.margolus_step([0, 0, 0], 0);

        let below = grid.get_voxel(0, 0, 0);
        let above = grid.get_voxel(0, 1, 0);
        assert_eq!(below.material_id(), 2, "sand should have fallen to y=0");
        assert!(above.is_air(), "y=1 should now be air");
    }

    #[test]
    fn test_water_falls() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        let water = Voxel::new(3, 128, 0, 0, 0);
        grid.set_voxel(0, 1, 0, water);
        grid.set_voxel(0, 0, 0, Voxel::air());

        grid.margolus_step([0, 0, 0], 0);

        assert_eq!(grid.get_voxel(0, 0, 0).material_id(), 3);
        assert!(grid.get_voxel(0, 1, 0).is_air());
    }

    #[test]
    fn test_density_ordering() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Sand (density=150) above water (density=100). Sand should sink.
        let sand = Voxel::new(2, 128, 0, 0, 0);
        let water = Voxel::new(3, 128, 0, 0, 0);
        grid.set_voxel(0, 1, 0, sand);
        grid.set_voxel(0, 0, 0, water);

        grid.margolus_step([0, 0, 0], 0);

        assert_eq!(
            grid.get_voxel(0, 0, 0).material_id(),
            2,
            "sand should sink below water"
        );
        assert_eq!(
            grid.get_voxel(0, 1, 0).material_id(),
            3,
            "water should rise above sand"
        );
    }

    #[test]
    fn test_gas_rises() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Steam at y=0, air at y=1. Steam should rise.
        let steam = Voxel::new(5, 128, 0, 0, 0);
        grid.set_voxel(0, 0, 0, steam);
        grid.set_voxel(0, 1, 0, Voxel::air());

        grid.margolus_step([0, 0, 0], 0);

        assert_eq!(
            grid.get_voxel(0, 1, 0).material_id(),
            5,
            "steam should rise to y=1"
        );
        assert!(
            grid.get_voxel(0, 0, 0).is_air(),
            "y=0 should be air after steam rises"
        );
    }

    #[test]
    fn test_reaction_water_lava() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Water at (0,0,0), lava at (1,0,0). They are adjacent in a 2x2x2 block.
        let water = Voxel::new(3, 128, 0, 0, 0);
        let lava = Voxel::new(4, 250, 0, 0, 0);
        grid.set_voxel(0, 0, 0, water);
        grid.set_voxel(1, 0, 0, lava);

        grid.margolus_step([0, 0, 0], 0);

        let v0 = grid.get_voxel(0, 0, 0);
        let v1 = grid.get_voxel(1, 0, 0);

        // Water -> stone, lava -> steam.
        assert_eq!(v0.material_id(), 1, "water should become stone");
        assert_eq!(v1.material_id(), 5, "lava should become steam");

        // Heat delta = -20 applied to both.
        assert_eq!(v0.temperature(), 108); // 128 - 20
        assert_eq!(v1.temperature(), 230); // 250 - 20
    }

    #[test]
    fn test_phase_transition_ice_melts() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Ice (mat_id=6) with temp above melt_temp (50). Should become water (3).
        let ice = Voxel::new(6, 60, 0, 0, 0);
        grid.set_voxel(0, 0, 0, ice);

        grid.thermal_step();

        let v = grid.get_voxel(0, 0, 0);
        assert_eq!(v.material_id(), 3, "ice should melt into water");
        assert_eq!(v.bonds(), 0, "melted voxel should have no bonds");
    }

    #[test]
    fn test_phase_transition_water_boils() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Water (mat_id=3) with temp above boil_temp (200). Should become steam (5).
        let water = Voxel::new(3, 210, 0, 0, 0);
        grid.set_voxel(0, 0, 0, water);

        grid.thermal_step();

        let v = grid.get_voxel(0, 0, 0);
        assert_eq!(v.material_id(), 5, "water should boil into steam");
    }

    #[test]
    fn test_determinism() {
        let materials = test_materials();
        let reactions = test_reactions();

        // Build identical worlds.
        let build_world = || {
            let mut grid = CaGrid::new(materials.clone(), reactions.clone());
            grid.set_voxel(0, 0, 0, Voxel::new(3, 128, 0, 0, 0)); // water
            grid.set_voxel(1, 0, 0, Voxel::new(4, 250, 0, 0, 0)); // lava
            grid.set_voxel(0, 1, 0, Voxel::new(2, 100, 0, 0, 0)); // sand
            grid.set_voxel(1, 1, 0, Voxel::new(5, 150, 0, 0, 0)); // steam
            grid.set_voxel(0, 0, 1, Voxel::new(1, 50, 0, 0, 0)); // stone
            grid
        };

        let mut grid_a = build_world();
        let mut grid_b = build_world();

        // Run 5 full steps.
        for frame in 0..5 {
            grid_a.step(frame);
            grid_b.step(frame);
        }

        // Compare all chunk data.
        for (key, data_a) in &grid_a.chunks {
            let data_b = grid_b
                .chunks
                .get(key)
                .expect("grid_b should have same chunks");
            assert_eq!(data_a, data_b, "chunk {:?} differs", key);
        }
        assert_eq!(grid_a.chunks.len(), grid_b.chunks.len());
    }

    #[test]
    fn test_step_full_cycle() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        // Set up a small scene across 2 chunks.
        grid.set_voxel(0, 0, 0, Voxel::new(1, 100, 0, 0, 0)); // stone
        grid.set_voxel(1, 0, 0, Voxel::new(3, 128, 0, 0, 0)); // water
        grid.set_voxel(0, 1, 0, Voxel::new(2, 80, 0, 0, 0)); // sand
        grid.set_voxel(32, 0, 0, Voxel::new(4, 250, 0, 0, 0)); // lava in chunk [1,0,0]

        // Run a full step. Should not panic.
        grid.step(0);

        // Verify some voxels changed or stayed.
        let stone = grid.get_voxel(0, 0, 0);
        assert_eq!(stone.material_id(), 1, "stone should remain stone");

        // Sand was at y=1, air at y=0 below it in block... depends on block alignment.
        // Just verify no panic and all voxels are valid.
        for wx in 0..2 {
            for wy in 0..2 {
                for wz in 0..2 {
                    let v = grid.get_voxel(wx, wy, wz);
                    assert!(
                        (v.material_id() as usize) < test_materials().len() || v.is_air(),
                        "voxel at ({},{},{}) has invalid material {}",
                        wx,
                        wy,
                        wz,
                        v.material_id()
                    );
                }
            }
        }
    }

    #[test]
    fn test_load_chunk_and_get_chunk_data() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());
        let mut data = vec![0u32; CA_CHUNK_VOXELS];
        data[0] = Voxel::new(1, 100, 0, 0, 0).0;
        grid.load_chunk(0, 0, 0, data.clone());

        let loaded = grid.get_chunk_data(0, 0, 0).expect("chunk should exist");
        assert_eq!(loaded.len(), CA_CHUNK_VOXELS);
        assert_eq!(loaded[0], data[0]);

        assert!(grid.get_chunk_data(1, 1, 1).is_none());
    }

    #[test]
    fn test_hash_position_deterministic() {
        let h1 = hash_position(10, 20, 30, 42);
        let h2 = hash_position(10, 20, 30, 42);
        assert_eq!(h1, h2);

        // Different inputs give different hashes (with very high probability).
        let h3 = hash_position(10, 20, 30, 43);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_negative_coords() {
        let mut grid = CaGrid::new(test_materials(), test_reactions());

        let v = Voxel::new(2, 100, 0, 0, 0);
        grid.set_voxel(-5, -10, -20, v);
        assert_eq!(grid.get_voxel(-5, -10, -20), v);

        // Verify chunk coord.
        let (chunk, _) = CaGrid::world_to_chunk_local(-5, -10, -20);
        assert_eq!(chunk, [-1, -1, -1]);
    }
}
