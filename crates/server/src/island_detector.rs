//! CPU BFS island detection for rigid body detachment.
//!
//! After an explosion carves a crater, [`detect_islands`] runs BFS from boundary
//! positions to find disconnected solid pieces that are no longer grounded.
//! Each ungrounded piece (up to [`MAX_ISLAND_VOXELS`]) becomes an [`IslandResult`]
//! that can be activated as a rigid body PB-MPM zone.

use std::collections::{HashMap, HashSet, VecDeque};

use shared::material_ca::MaterialPropertiesCA;
use shared::voxel::Voxel;

/// Maximum number of voxels in an island before it is discarded.
const MAX_ISLAND_VOXELS: usize = 2000;

/// Chunk size (voxels per axis).
const CHUNK_SIZE: i32 = 32;

/// Result of a detected floating island.
#[derive(Debug, Clone)]
pub struct IslandResult {
    /// World positions of all voxels in the island.
    pub voxels: Vec<[i32; 3]>,
    /// Center of mass (world coordinates).
    pub center_of_mass: [f32; 3],
    /// Axis-aligned bounding box minimum (world coordinates).
    pub aabb_min: [i32; 3],
    /// Axis-aligned bounding box maximum (world coordinates).
    pub aabb_max: [i32; 3],
    /// Packed voxel data for each voxel in the island (same order as `voxels`).
    pub voxel_data: Vec<u32>,
    /// PB-MPM grid size (next power of two fitting the AABB + padding).
    pub grid_size: u32,
}

/// Read a single voxel from the world chunk data.
///
/// Returns the raw packed u32 voxel value, or 0 (air) if the chunk is not loaded.
pub fn read_world_voxel(chunks: &HashMap<[i32; 3], Vec<u32>>, wx: i32, wy: i32, wz: i32) -> u32 {
    let cs = CHUNK_SIZE;
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    chunks
        .get(&[cx, cy, cz])
        .map(|d| d[lz * 32 * 32 + ly * 32 + lx])
        .unwrap_or(0)
}

/// Check if a voxel is solid (phase == 0) based on its material properties.
fn is_solid_voxel(raw: u32, materials: &[MaterialPropertiesCA]) -> bool {
    let v = Voxel(raw);
    if v.is_air() {
        return false;
    }
    let mat_id = v.material_id() as usize;
    if mat_id < materials.len() {
        materials[mat_id].phase == 0
    } else {
        // Unknown material — treat as solid
        false
    }
}

/// 6-connected face neighbors.
const FACE_NEIGHBORS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// Check if a voxel has an unbroken support column to y=0.
///
/// A voxel is grounded if there exists an unbroken chain of solid voxels
/// directly below it all the way to world y=0.
fn is_grounded(
    chunks: &HashMap<[i32; 3], Vec<u32>>,
    wx: i32,
    wy: i32,
    wz: i32,
    materials: &[MaterialPropertiesCA],
) -> bool {
    for y in 0..wy {
        let raw = read_world_voxel(chunks, wx, y, wz);
        if !is_solid_voxel(raw, materials) {
            return false;
        }
    }
    // If wy == 0, the voxel is at ground level → grounded
    true
}

/// Detect floating islands from boundary positions after crater carving.
///
/// For each boundary position, checks 6 face neighbors that are still solid,
/// then runs BFS/flood fill to find the connected solid component. If the
/// component is ungrounded (no voxel has a solid column to y=0) and has
/// at most [`MAX_ISLAND_VOXELS`] voxels, it is returned as an [`IslandResult`].
///
/// # Arguments
/// * `chunks` — chunk coordinate to voxel data mapping
/// * `boundary` — positions at the crater edge
/// * `materials` — material property table for phase lookup
pub fn detect_islands(
    chunks: &HashMap<[i32; 3], Vec<u32>>,
    boundary: &[[i32; 3]],
    materials: &[MaterialPropertiesCA],
) -> Vec<IslandResult> {
    let mut visited: HashSet<[i32; 3]> = HashSet::new();
    let mut results = Vec::new();

    // Collect BFS start positions: solid neighbors of boundary voxels
    let mut seeds: Vec<[i32; 3]> = Vec::new();
    for &pos in boundary {
        for &offset in &FACE_NEIGHBORS {
            let neighbor = [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]];
            let raw = read_world_voxel(chunks, neighbor[0], neighbor[1], neighbor[2]);
            if is_solid_voxel(raw, materials) && !visited.contains(&neighbor) {
                seeds.push(neighbor);
            }
        }
    }

    for seed in seeds {
        if visited.contains(&seed) {
            continue;
        }

        // BFS flood fill
        let mut queue = VecDeque::new();
        let mut island_voxels: Vec<[i32; 3]> = Vec::new();
        let mut island_data: Vec<u32> = Vec::new();
        let mut grounded = false;
        let mut too_large = false;

        queue.push_back(seed);
        visited.insert(seed);

        while let Some(pos) = queue.pop_front() {
            let raw = read_world_voxel(chunks, pos[0], pos[1], pos[2]);
            if !is_solid_voxel(raw, materials) {
                continue;
            }

            island_voxels.push(pos);
            island_data.push(raw);

            if island_voxels.len() > MAX_ISLAND_VOXELS {
                too_large = true;
                break;
            }

            // Ground check: if any voxel in the island is at y=0, it's grounded
            if pos[1] == 0 {
                grounded = true;
            }

            // Check if this voxel has a support column to ground
            if !grounded && is_grounded(chunks, pos[0], pos[1], pos[2], materials) {
                grounded = true;
            }

            // Expand to face neighbors
            for &offset in &FACE_NEIGHBORS {
                let neighbor = [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]];
                if !visited.contains(&neighbor) {
                    let nraw = read_world_voxel(chunks, neighbor[0], neighbor[1], neighbor[2]);
                    if is_solid_voxel(nraw, materials) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Skip if grounded, too large, or empty
        if grounded || too_large || island_voxels.is_empty() {
            continue;
        }

        // Compute AABB
        let mut aabb_min = island_voxels[0];
        let mut aabb_max = island_voxels[0];
        let mut com = [0.0f32; 3];

        for &pos in &island_voxels {
            for i in 0..3 {
                aabb_min[i] = aabb_min[i].min(pos[i]);
                aabb_max[i] = aabb_max[i].max(pos[i]);
            }
            com[0] += pos[0] as f32;
            com[1] += pos[1] as f32;
            com[2] += pos[2] as f32;
        }

        let count = island_voxels.len() as f32;
        com[0] /= count;
        com[1] /= count;
        com[2] /= count;

        // Grid size: next power of two that fits the AABB extent + 4 padding, clamped to [16, 64]
        let extent_x = (aabb_max[0] - aabb_min[0] + 1) as u32;
        let extent_y = (aabb_max[1] - aabb_min[1] + 1) as u32;
        let extent_z = (aabb_max[2] - aabb_min[2] + 1) as u32;
        let max_extent = extent_x.max(extent_y).max(extent_z) + 4;
        let grid_size = max_extent.next_power_of_two().clamp(16, 64);

        results.push(IslandResult {
            voxels: island_voxels,
            center_of_mass: com,
            aabb_min,
            aabb_max,
            voxel_data: island_data,
            grid_size,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    /// Helper: create a stone material (phase=0) at index 1.
    fn test_materials() -> Vec<MaterialPropertiesCA> {
        vec![
            MaterialPropertiesCA::zeroed(), // 0 = air (unused, air is material_id=0)
            {
                let mut m = MaterialPropertiesCA::zeroed();
                m.phase = 0; // solid
                m.density = 100;
                m
            },
        ]
    }

    /// Helper: build chunk data with solid voxels at specific world positions.
    fn build_chunks(positions: &[[i32; 3]], mat_id: u16) -> HashMap<[i32; 3], Vec<u32>> {
        let mut chunks: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for &pos in positions {
            let cx = pos[0].div_euclid(32);
            let cy = pos[1].div_euclid(32);
            let cz = pos[2].div_euclid(32);
            let lx = pos[0].rem_euclid(32) as usize;
            let ly = pos[1].rem_euclid(32) as usize;
            let lz = pos[2].rem_euclid(32) as usize;
            let chunk = chunks
                .entry([cx, cy, cz])
                .or_insert_with(|| vec![0u32; 32 * 32 * 32]);
            let idx = lz * 32 * 32 + ly * 32 + lx;
            chunk[idx] = Voxel::new(mat_id, 20, 0b111111, 0, 0).0;
        }
        chunks
    }

    #[test]
    fn test_bfs_finds_floating_island() {
        let materials = test_materials();

        // Create a 5x5x5 cube floating at y=10..14
        let mut positions = Vec::new();
        for z in 10..15 {
            for y in 10..15 {
                for x in 10..15 {
                    positions.push([x, y, z]);
                }
            }
        }
        let chunks = build_chunks(&positions, 1);

        // Boundary: a position just below the cube (air underneath)
        let boundary = vec![[10, 9, 10]];

        let islands = detect_islands(&chunks, &boundary, &materials);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].voxels.len(), 125); // 5^3
        assert!(islands[0].grid_size >= 16);
    }

    #[test]
    fn test_grounded_island_skipped() {
        let materials = test_materials();

        // Create a column from y=0 to y=4 — grounded
        let mut positions = Vec::new();
        for y in 0..5 {
            positions.push([10, y, 10]);
        }
        let chunks = build_chunks(&positions, 1);

        // Boundary: beside the column
        let boundary = vec![[11, 2, 10]];

        let islands = detect_islands(&chunks, &boundary, &materials);
        assert!(islands.is_empty(), "Grounded islands should be skipped");
    }

    #[test]
    fn test_max_voxel_limit() {
        let materials = test_materials();

        // Create a large block (13x13x13 = 2197 > 2000)
        let mut positions = Vec::new();
        for z in 5..18 {
            for y in 5..18 {
                for x in 5..18 {
                    positions.push([x, y, z]);
                }
            }
        }
        let chunks = build_chunks(&positions, 1);

        // Boundary next to the block
        let boundary = vec![[4, 10, 10]];

        let islands = detect_islands(&chunks, &boundary, &materials);
        assert!(
            islands.is_empty(),
            "Islands exceeding MAX_ISLAND_VOXELS should be skipped"
        );
    }

    #[test]
    fn test_read_world_voxel() {
        let mut chunks: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        let mut data = vec![0u32; 32 * 32 * 32];
        // Set voxel at local (5, 10, 15) in chunk (0, 0, 0)
        let idx = 15 * 32 * 32 + 10 * 32 + 5;
        data[idx] = Voxel::new(1, 100, 0, 0, 0).0;
        chunks.insert([0, 0, 0], data);

        assert_ne!(read_world_voxel(&chunks, 5, 10, 15), 0);
        assert_eq!(read_world_voxel(&chunks, 0, 0, 0), 0);
        // Non-existent chunk
        assert_eq!(read_world_voxel(&chunks, 100, 100, 100), 0);
    }

    #[test]
    fn test_empty_boundary_returns_empty() {
        let materials = test_materials();
        let chunks: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        let boundary: Vec<[i32; 3]> = Vec::new();

        let islands = detect_islands(&chunks, &boundary, &materials);
        assert!(islands.is_empty());
    }

    #[test]
    fn test_island_aabb_and_com() {
        let materials = test_materials();

        // 3x1x1 bar at y=5
        let positions = vec![[10, 5, 10], [11, 5, 10], [12, 5, 10]];
        let chunks = build_chunks(&positions, 1);

        let boundary = vec![[9, 5, 10]];

        let islands = detect_islands(&chunks, &boundary, &materials);
        assert_eq!(islands.len(), 1);
        let island = &islands[0];
        assert_eq!(island.aabb_min, [10, 5, 10]);
        assert_eq!(island.aabb_max, [12, 5, 10]);
        // Center of mass: (11, 5, 10)
        assert!((island.center_of_mass[0] - 11.0).abs() < 0.01);
        assert!((island.center_of_mass[1] - 5.0).abs() < 0.01);
        assert!((island.center_of_mass[2] - 10.0).abs() < 0.01);
    }
}
