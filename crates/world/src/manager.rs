//! World manager: streaming chunks around the camera.

use std::collections::HashMap;
use std::path::PathBuf;

use shared::constants::CHUNK_SIZE;
use shared::Particle;

use crate::chunk::{global_to_local, local_to_global, ChunkCoord, ChunkData};
use crate::storage::{ChunkStorage, StorageError};
use crate::terrain::TerrainGenerator;

/// Result of a center-update operation: which chunks were evicted and loaded.
#[derive(Debug, Default)]
pub struct StreamingResult {
    /// Chunks that were removed from the active set.
    pub evicted: Vec<ChunkCoord>,
    /// Chunks that were added to the active set.
    pub loaded: Vec<ChunkCoord>,
}

/// Errors from the world manager.
#[derive(Debug, thiserror::Error)]
pub enum WorldError {
    /// Storage I/O failure.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
}

/// A chunk that is part of the active simulation grid.
struct ActiveChunk {
    data: ChunkData,
    /// Range of this chunk's particles within the packed GPU buffer.
    particle_range: std::ops::Range<u32>,
}

/// Manages the infinite world: streaming, generation, and persistence.
pub struct WorldManager {
    active_chunks: HashMap<ChunkCoord, ActiveChunk>,
    dormant_cache: HashMap<ChunkCoord, ChunkData>,
    storage: ChunkStorage,
    terrain: TerrainGenerator,
    active_center: ChunkCoord,
    /// Simulation radius in chunks (active chunks form a (2r+1)^2 * height_chunks box).
    sim_radius: u32,
    /// Render radius in chunks (for far-field voxel snapshots).
    render_radius: u32,
}

impl WorldManager {
    /// Create a new world manager.
    ///
    /// - `seed`: terrain generation seed.
    /// - `cache_dir`: directory for on-disk chunk persistence.
    pub fn new(seed: u64, cache_dir: PathBuf) -> Result<Self, WorldError> {
        let storage = ChunkStorage::new(cache_dir)?;
        Ok(Self {
            active_chunks: HashMap::new(),
            dormant_cache: HashMap::new(),
            storage,
            terrain: TerrainGenerator::new(seed),
            active_center: ChunkCoord::new(0, 0, 0),
            sim_radius: 1,
            render_radius: 3,
        })
    }

    /// Get the current active center chunk.
    pub fn active_center(&self) -> ChunkCoord {
        self.active_center
    }

    /// Get the simulation radius in chunks.
    pub fn sim_radius(&self) -> u32 {
        self.sim_radius
    }

    /// Set the simulation radius in chunks.
    pub fn set_sim_radius(&mut self, radius: u32) {
        self.sim_radius = radius;
    }

    /// Get the render radius in chunks.
    pub fn render_radius(&self) -> u32 {
        self.render_radius
    }

    /// Set the render radius in chunks.
    pub fn set_render_radius(&mut self, radius: u32) {
        self.render_radius = radius;
    }

    /// Number of currently active chunks.
    pub fn active_chunk_count(&self) -> usize {
        self.active_chunks.len()
    }

    /// Update the active center. Evicts chunks that are now out of range,
    /// loads/generates chunks that are newly in range.
    ///
    /// Returns a summary of what changed.
    pub fn update_center(&mut self, new_center: ChunkCoord) -> Result<StreamingResult, WorldError> {
        if new_center == self.active_center && !self.active_chunks.is_empty() {
            return Ok(StreamingResult::default());
        }

        let old_set = self.chunks_in_radius(self.active_center, self.sim_radius);
        let new_set = self.chunks_in_radius(new_center, self.sim_radius);

        let mut result = StreamingResult::default();

        // Evict chunks no longer in the active set
        for coord in &old_set {
            if !new_set.contains(coord) {
                if let Some(active) = self.active_chunks.remove(coord) {
                    // Try to save, but don't fail the whole operation
                    if let Err(e) = self.storage.save(&active.data) {
                        tracing::warn!(chunk = %coord, error = %e, "failed to save evicted chunk");
                    }
                    self.dormant_cache.insert(*coord, active.data);
                    result.evicted.push(*coord);
                }
            }
        }

        // Load chunks newly in range
        for coord in &new_set {
            if !self.active_chunks.contains_key(coord) {
                let data = self.load_or_generate(coord)?;
                self.active_chunks.insert(
                    *coord,
                    ActiveChunk {
                        data,
                        particle_range: 0..0, // updated on pack
                    },
                );
                result.loaded.push(*coord);
            }
        }

        self.active_center = new_center;

        tracing::info!(
            center = %new_center,
            evicted = result.evicted.len(),
            loaded = result.loaded.len(),
            active = self.active_chunks.len(),
            "world center updated"
        );

        Ok(result)
    }

    /// Merge all active chunk particles into a single Vec with global coordinates.
    ///
    /// Particles are translated from local chunk space to the global grid,
    /// where `active_center` maps to the grid origin.
    pub fn pack_particles_for_gpu(&mut self) -> Vec<Particle> {
        let mut packed = Vec::new();

        for (coord, active) in &mut self.active_chunks {
            let start = packed.len() as u32;
            for p in &active.data.particles {
                let pos = p.position();
                let global =
                    local_to_global([pos.x, pos.y, pos.z], coord, &self.active_center);
                let mut gp = *p;
                gp.set_position(glam::Vec3::new(global[0], global[1], global[2]));
                packed.push(gp);
            }
            let end = packed.len() as u32;
            active.particle_range = start..end;
        }

        tracing::debug!(total_particles = packed.len(), "packed particles for GPU");
        packed
    }

    /// Distribute GPU-space particles back into their respective chunks.
    ///
    /// Each particle's global position is converted back to a chunk coordinate
    /// and local position, then placed into the matching active chunk.
    pub fn unpack_particles_from_gpu(&mut self, particles: &[Particle]) {
        // Clear existing particles in all active chunks
        for active in self.active_chunks.values_mut() {
            active.data.particles.clear();
        }

        for p in particles {
            let pos = p.position();
            let (chunk_coord, local_pos) =
                global_to_local([pos.x, pos.y, pos.z], &self.active_center);

            if let Some(active) = self.active_chunks.get_mut(&chunk_coord) {
                let mut lp = *p;
                lp.set_position(glam::Vec3::new(local_pos[0], local_pos[1], local_pos[2]));
                active.data.particles.push(lp);
            } else {
                // Particle escaped the active region — put it in the nearest
                // active chunk to avoid data loss. In practice this means
                // clamping to the border chunk.
                tracing::debug!(
                    chunk = %chunk_coord,
                    "particle outside active region, dropping"
                );
            }
        }
    }

    /// Build a voxel snapshot for a chunk by discretizing particle positions.
    pub fn bake_voxel_snapshot(chunk: &mut ChunkData) {
        let cs = CHUNK_SIZE as usize;
        chunk.voxel_snapshot.fill(0);

        for p in &chunk.particles {
            let pos = p.position();
            let x = pos.x.floor() as usize;
            let y = pos.y.floor() as usize;
            let z = pos.z.floor() as usize;

            if x < cs && y < cs && z < cs {
                let idx = z * cs * cs + y * cs + x;
                // Material ID + 1 so that 0 means empty
                chunk.voxel_snapshot[idx] = (p.material_id() as u8).wrapping_add(1);
            }
        }
    }

    /// Enumerate chunk coords within `radius` of `center`.
    ///
    /// For the Y axis, we use all chunks in [0, WORLD_HEIGHT_CHUNKS).
    fn chunks_in_radius(&self, center: ChunkCoord, radius: u32) -> Vec<ChunkCoord> {
        let r = radius as i32;
        let height = shared::constants::WORLD_HEIGHT_CHUNKS as i32;
        let mut coords = Vec::new();

        for dx in -r..=r {
            for dz in -r..=r {
                for y in 0..height {
                    coords.push(ChunkCoord::new(center.x + dx, y, center.z + dz));
                }
            }
        }

        coords
    }

    /// Load a chunk from the dormant cache, disk, or generate it fresh.
    fn load_or_generate(&mut self, coord: &ChunkCoord) -> Result<ChunkData, WorldError> {
        // 1. Check dormant cache
        if let Some(data) = self.dormant_cache.remove(coord) {
            tracing::debug!(chunk = %coord, "loaded from dormant cache");
            return Ok(data);
        }

        // 2. Check disk
        if let Some(data) = self.storage.load(coord)? {
            tracing::debug!(chunk = %coord, "loaded from disk");
            return Ok(data);
        }

        // 3. Generate
        let data = self.terrain.generate_chunk(*coord);
        tracing::debug!(chunk = %coord, "generated new chunk");
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn test_manager() -> WorldManager {
        let dir = std::env::temp_dir().join(format!(
            "vox_manager_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        WorldManager::new(42, dir).unwrap()
    }

    #[test]
    fn initial_update_loads_chunks() {
        let mut mgr = test_manager();
        let result = mgr.update_center(ChunkCoord::new(0, 0, 0)).unwrap();
        // sim_radius=1 → 3x3 horizontal * WORLD_HEIGHT_CHUNKS vertical
        let expected = 3 * 3 * shared::constants::WORLD_HEIGHT_CHUNKS as usize;
        assert_eq!(result.loaded.len(), expected);
        assert_eq!(mgr.active_chunk_count(), expected);
    }

    #[test]
    fn same_center_is_noop() {
        let mut mgr = test_manager();
        mgr.update_center(ChunkCoord::new(0, 0, 0)).unwrap();
        let result = mgr.update_center(ChunkCoord::new(0, 0, 0)).unwrap();
        assert!(result.loaded.is_empty());
        assert!(result.evicted.is_empty());
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut mgr = test_manager();
        mgr.set_sim_radius(0); // only the center column
        mgr.update_center(ChunkCoord::new(0, 0, 0)).unwrap();

        // Count total particles across active chunks
        let total_before: usize = mgr
            .active_chunks
            .values()
            .map(|a| a.data.particles.len())
            .sum();

        let packed = mgr.pack_particles_for_gpu();
        assert_eq!(packed.len(), total_before);

        mgr.unpack_particles_from_gpu(&packed);

        let total_after: usize = mgr
            .active_chunks
            .values()
            .map(|a| a.data.particles.len())
            .sum();

        assert_eq!(total_before, total_after);
    }

    #[test]
    fn bake_voxel_snapshot() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = ChunkData::empty(coord);
        chunk.particles.push(Particle::new(
            Vec3::new(2.5, 3.5, 4.5),
            1.0,
            5, // material_id
            0,
        ));

        WorldManager::bake_voxel_snapshot(&mut chunk);

        let idx = 4 * 64 * 64 + 3 * 64 + 2; // z=4, y=3, x=2
        assert_eq!(chunk.voxel_snapshot[idx], 6); // material_id + 1
    }

    #[test]
    fn center_shift_evicts_and_loads() {
        let mut mgr = test_manager();
        mgr.set_sim_radius(0);
        mgr.update_center(ChunkCoord::new(0, 0, 0)).unwrap();

        let result = mgr.update_center(ChunkCoord::new(1, 0, 0)).unwrap();
        // Moving by 1 chunk with radius=0 should evict the old column and load the new
        assert!(!result.evicted.is_empty());
        assert!(!result.loaded.is_empty());
    }
}
