//! Chunk save/load to disk.
//!
//! File format (per chunk):
//! - 4 bytes: particle count (little-endian u32)
//! - N * 144 bytes: raw `Particle` data (bytemuck cast)
//! - 64^3 bytes: voxel snapshot

use std::fs;
use std::path::PathBuf;

use bytemuck::{self, Zeroable};
use shared::constants::CHUNK_SIZE;
use shared::Particle;

use crate::chunk::{ChunkCoord, ChunkData};

/// Errors that can occur during chunk storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// I/O error during read or write.
    #[error("chunk storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File is too small or has invalid structure.
    #[error("corrupt chunk file: {reason}")]
    Corrupt { reason: String },
}

/// On-disk chunk persistence backed by a directory of `.chunk` files.
pub struct ChunkStorage {
    base_dir: PathBuf,
}

impl ChunkStorage {
    /// Create a new storage instance rooted at `base_dir`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(base_dir: PathBuf) -> Result<Self, StorageError> {
        fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    /// Save a chunk to disk, overwriting any previous data.
    pub fn save(&self, chunk: &ChunkData) -> Result<(), StorageError> {
        let path = self.chunk_path(&chunk.coord);
        let particle_bytes: &[u8] = bytemuck::cast_slice(&chunk.particles);
        let count = chunk.particles.len() as u32;

        let mut data = Vec::with_capacity(4 + particle_bytes.len() + chunk.voxel_snapshot.len());
        data.extend_from_slice(&count.to_le_bytes());
        data.extend_from_slice(particle_bytes);
        data.extend_from_slice(&chunk.voxel_snapshot);

        fs::write(&path, &data)?;
        tracing::debug!(
            path = %path.display(),
            particles = count,
            "saved chunk"
        );
        Ok(())
    }

    /// Load a chunk from disk, returning `None` if the file does not exist.
    pub fn load(&self, coord: &ChunkCoord) -> Result<Option<ChunkData>, StorageError> {
        let path = self.chunk_path(coord);
        if !path.exists() {
            return Ok(None);
        }

        let data = fs::read(&path)?;
        let snapshot_size = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
        let particle_size = core::mem::size_of::<Particle>();

        if data.len() < 4 {
            return Err(StorageError::Corrupt {
                reason: "file too small for header".into(),
            });
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_len = 4 + count * particle_size + snapshot_size;
        if data.len() != expected_len {
            return Err(StorageError::Corrupt {
                reason: format!(
                    "expected {} bytes, got {} (count={})",
                    expected_len,
                    data.len(),
                    count
                ),
            });
        }

        let particle_bytes = &data[4..4 + count * particle_size];
        // We can't use cast_slice directly because the Vec<u8> from fs::read
        // may not be 16-byte aligned. Copy particle-by-particle instead.
        let mut particles = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * particle_size;
            let mut p = Particle::zeroed();
            let dst: &mut [u8] = bytemuck::bytes_of_mut(&mut p);
            dst.copy_from_slice(&particle_bytes[offset..offset + particle_size]);
            particles.push(p);
        }

        let snapshot_start = 4 + count * particle_size;
        let voxel_snapshot = data[snapshot_start..snapshot_start + snapshot_size].to_vec();

        tracing::debug!(
            path = %path.display(),
            particles = count,
            "loaded chunk"
        );

        Ok(Some(ChunkData {
            coord: *coord,
            particles,
            voxel_snapshot,
            is_generated: true,
        }))
    }

    /// Build the file path for a chunk coordinate.
    fn chunk_path(&self, coord: &ChunkCoord) -> PathBuf {
        self.base_dir
            .join(format!("{}_{}_{}.chunk", coord.x, coord.y, coord.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join("vox_storage_test_roundtrip");
        let _ = fs::remove_dir_all(&dir);
        let storage = ChunkStorage::new(dir.clone()).unwrap();

        let coord = ChunkCoord::new(1, 0, -3);
        let mut chunk = ChunkData::empty(coord);
        chunk.is_generated = true;
        chunk.particles.push(Particle::new(Vec3::new(10.0, 20.0, 30.0), 1.0, 0, 0));
        chunk.particles.push(Particle::new(Vec3::new(5.0, 3.0, 7.0), 0.5, 1, 1));
        chunk.voxel_snapshot[0] = 42;
        chunk.voxel_snapshot[1000] = 7;

        storage.save(&chunk).unwrap();
        let loaded = storage.load(&coord).unwrap().expect("should exist");

        assert_eq!(loaded.particles.len(), 2);
        assert_eq!(loaded.particles[0].position(), Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(loaded.particles[1].material_id(), 1);
        assert_eq!(loaded.voxel_snapshot[0], 42);
        assert_eq!(loaded.voxel_snapshot[1000], 7);
        assert!(loaded.is_generated);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = std::env::temp_dir().join("vox_storage_test_none");
        let _ = fs::remove_dir_all(&dir);
        let storage = ChunkStorage::new(dir.clone()).unwrap();

        let result = storage.load(&ChunkCoord::new(99, 0, 99)).unwrap();
        assert!(result.is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
