//! Procedural terrain generation.
//!
//! Deterministic: same seed + same chunk coordinate = same output, always.

use glam::Vec3;
use shared::constants::CHUNK_SIZE;
use shared::Particle;

use crate::chunk::{ChunkCoord, ChunkData};

/// Procedural terrain generator with a fixed seed.
pub struct TerrainGenerator {
    seed: u64,
}

/// Base terrain height in voxels (from bottom of chunk y=0).
const BASE_HEIGHT: f32 = 32.0;

/// Amplitude of the tallest noise octave.
const NOISE_AMPLITUDE: f32 = 16.0;

/// Sea level in voxels (water fills below this if above terrain).
const WATER_LEVEL: f32 = 28.0;

/// Cave threshold: 3D noise below this value carves empty space.
const CAVE_THRESHOLD: f32 = -0.3;

/// Material ID for stone.
const MAT_STONE: u32 = 0;

/// Material ID for water.
const MAT_WATER: u32 = 1;

impl TerrainGenerator {
    /// Create a new terrain generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate a full chunk of terrain data.
    ///
    /// The returned `ChunkData` has `is_generated = true` and particles
    /// in local coordinates (0..CHUNK_SIZE).
    pub fn generate_chunk(&self, coord: ChunkCoord) -> ChunkData {
        let cs = CHUNK_SIZE as usize;
        let mut particles = Vec::new();

        for lz in 0..cs {
            for lx in 0..cs {
                // World-space xz for noise sampling
                let wx = coord.x as f32 * CHUNK_SIZE as f32 + lx as f32;
                let wz = coord.z as f32 * CHUNK_SIZE as f32 + lz as f32;

                let height = self.terrain_height(wx, wz);

                for ly in 0..cs {
                    let wy = coord.y as f32 * CHUNK_SIZE as f32 + ly as f32;
                    let local = [lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5];

                    if wy < height {
                        // Check cave carving
                        let cave_val = self.noise_3d(wx, wy, wz);
                        if cave_val < CAVE_THRESHOLD {
                            continue; // carved out
                        }
                        particles.push(Particle::new(
                            Vec3::new(local[0], local[1], local[2]),
                            1.0,
                            MAT_STONE,
                            0, // solid
                        ));
                    } else if wy < WATER_LEVEL {
                        particles.push(Particle::new(
                            Vec3::new(local[0], local[1], local[2]),
                            1.0,
                            MAT_WATER,
                            1, // liquid
                        ));
                    }
                }
            }
        }

        tracing::debug!(
            chunk = %coord,
            particle_count = particles.len(),
            "generated terrain chunk"
        );

        ChunkData {
            coord,
            particles,
            voxel_snapshot: vec![0u8; cs * cs * cs],
            is_generated: true,
        }
    }

    /// Compute terrain height at world-space (x, z) using multi-octave noise.
    fn terrain_height(&self, x: f32, z: f32) -> f32 {
        let mut height = BASE_HEIGHT;
        let mut amp = NOISE_AMPLITUDE;
        let mut freq = 0.02_f32;

        for octave in 0..4u32 {
            height += self.noise_2d(x * freq, z * freq, octave) * amp;
            amp *= 0.5;
            freq *= 2.0;
        }

        height
    }

    /// Simple hash-based 2D noise in range [-1, 1].
    fn noise_2d(&self, x: f32, z: f32, octave: u32) -> f32 {
        // Integer cell
        let ix = x.floor() as i32;
        let iz = z.floor() as i32;
        let fx = x - x.floor();
        let fz = z - z.floor();

        // Smoothstep
        let ux = fx * fx * (3.0 - 2.0 * fx);
        let uz = fz * fz * (3.0 - 2.0 * fz);

        let v00 = self.hash_to_float(ix, 0, iz, octave);
        let v10 = self.hash_to_float(ix + 1, 0, iz, octave);
        let v01 = self.hash_to_float(ix, 0, iz + 1, octave);
        let v11 = self.hash_to_float(ix + 1, 0, iz + 1, octave);

        let a = v00 + (v10 - v00) * ux;
        let b = v01 + (v11 - v01) * ux;
        a + (b - a) * uz
    }

    /// Simple hash-based 3D noise in range [-1, 1] for cave carving.
    fn noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let freq = 0.05_f32;
        let sx = x * freq;
        let sy = y * freq;
        let sz = z * freq;

        let ix = sx.floor() as i32;
        let iy = sy.floor() as i32;
        let iz = sz.floor() as i32;
        let fx = sx - sx.floor();
        let fy = sy - sy.floor();
        let fz = sz - sz.floor();

        let ux = fx * fx * (3.0 - 2.0 * fx);
        let uy = fy * fy * (3.0 - 2.0 * fy);
        let uz = fz * fz * (3.0 - 2.0 * fz);

        // Trilinear interpolation of 8 corner hashes
        let v000 = self.hash_to_float(ix, iy, iz, 99);
        let v100 = self.hash_to_float(ix + 1, iy, iz, 99);
        let v010 = self.hash_to_float(ix, iy + 1, iz, 99);
        let v110 = self.hash_to_float(ix + 1, iy + 1, iz, 99);
        let v001 = self.hash_to_float(ix, iy, iz + 1, 99);
        let v101 = self.hash_to_float(ix + 1, iy, iz + 1, 99);
        let v011 = self.hash_to_float(ix, iy + 1, iz + 1, 99);
        let v111 = self.hash_to_float(ix + 1, iy + 1, iz + 1, 99);

        let a00 = v000 + (v100 - v000) * ux;
        let a10 = v010 + (v110 - v010) * ux;
        let a01 = v001 + (v101 - v001) * ux;
        let a11 = v011 + (v111 - v011) * ux;

        let b0 = a00 + (a10 - a00) * uy;
        let b1 = a01 + (a11 - a01) * uy;

        b0 + (b1 - b0) * uz
    }

    /// Hash (ix, iy, iz, octave, seed) to a float in [-1, 1].
    fn hash_to_float(&self, ix: i32, iy: i32, iz: i32, octave: u32) -> f32 {
        let h = self.hash(ix, iy, iz, octave);
        // Map u64 to [-1, 1]
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Deterministic bit-mixing hash.
    fn hash(&self, ix: i32, iy: i32, iz: i32, octave: u32) -> u32 {
        let mut h = self.seed;
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(ix as u64);
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(iy as u64);
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(iz as u64);
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(octave as u64);
        // Final mix
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_determinism() {
        let gen = TerrainGenerator::new(42);
        let coord = ChunkCoord::new(3, 0, -2);
        let a = gen.generate_chunk(coord);
        let b = gen.generate_chunk(coord);

        assert_eq!(a.particles.len(), b.particles.len());
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert_eq!(pa.position(), pb.position());
            assert_eq!(pa.material_id(), pb.material_id());
            assert_eq!(pa.phase(), pb.phase());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let coord = ChunkCoord::new(0, 0, 0);
        let a = TerrainGenerator::new(1).generate_chunk(coord);
        let b = TerrainGenerator::new(999).generate_chunk(coord);
        // Very unlikely to have the exact same particle count
        // (possible but astronomically unlikely for different seeds)
        assert_ne!(a.particles.len(), b.particles.len());
    }

    #[test]
    fn generated_chunk_is_marked() {
        let gen = TerrainGenerator::new(0);
        let c = gen.generate_chunk(ChunkCoord::new(0, 0, 0));
        assert!(c.is_generated);
    }

    #[test]
    fn noise_range() {
        let gen = TerrainGenerator::new(123);
        for i in 0..100 {
            let v = gen.noise_2d(i as f32 * 0.37, i as f32 * 0.53, 0);
            assert!((-1.0..=1.0).contains(&v), "noise out of range: {v}");
        }
    }
}
