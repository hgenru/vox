//! Chunk types and coordinate conversion utilities.

use shared::constants::CHUNK_SIZE;
use shared::Particle;

/// Integer coordinate identifying a chunk in the infinite world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    /// Create a new chunk coordinate.
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl core::fmt::Display for ChunkCoord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// Data for a single chunk: particles in local coordinates plus a voxel snapshot.
pub struct ChunkData {
    /// Which chunk this data belongs to.
    pub coord: ChunkCoord,
    /// Particles with positions in local chunk space (0..CHUNK_SIZE).
    pub particles: Vec<Particle>,
    /// 64^3 material-palette snapshot for far-field rendering.
    /// Index = z * 64*64 + y * 64 + x.
    pub voxel_snapshot: Vec<u8>,
    /// Whether this chunk has been generated (vs. default-constructed).
    pub is_generated: bool,
}

impl ChunkData {
    /// Create an empty (not yet generated) chunk.
    pub fn empty(coord: ChunkCoord) -> Self {
        Self {
            coord,
            particles: Vec::new(),
            voxel_snapshot: vec![0u8; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
            is_generated: false,
        }
    }
}

/// Determine which chunk a world-space XZ position falls in.
///
/// Y is bounded to the active grid height, so we return y=0 always
/// (vertical chunking uses the full `WORLD_HEIGHT_CHUNKS` range).
pub fn chunk_coord_for_position(world_x: f32, world_z: f32) -> ChunkCoord {
    let cs = CHUNK_SIZE as f32;
    ChunkCoord {
        x: (world_x / cs).floor() as i32,
        y: 0,
        z: (world_z / cs).floor() as i32,
    }
}

/// Convert a local-space position (0..CHUNK_SIZE) inside `chunk` to a
/// global grid position relative to `active_center`.
///
/// The active center chunk maps to grid origin (0,0,0). Neighbouring
/// chunks are offset by CHUNK_SIZE in each axis.
pub fn local_to_global(
    local_pos: [f32; 3],
    chunk: &ChunkCoord,
    active_center: &ChunkCoord,
) -> [f32; 3] {
    let cs = CHUNK_SIZE as f32;
    let dx = (chunk.x - active_center.x) as f32 * cs;
    let dy = (chunk.y - active_center.y) as f32 * cs;
    let dz = (chunk.z - active_center.z) as f32 * cs;
    [local_pos[0] + dx, local_pos[1] + dy, local_pos[2] + dz]
}

/// Convert a global grid position (relative to `active_center` at origin)
/// back to a chunk coordinate and local position.
pub fn global_to_local(
    global_pos: [f32; 3],
    active_center: &ChunkCoord,
) -> (ChunkCoord, [f32; 3]) {
    let cs = CHUNK_SIZE as f32;
    let chunk_x = (global_pos[0] / cs).floor() as i32 + active_center.x;
    let chunk_y = (global_pos[1] / cs).floor() as i32 + active_center.y;
    let chunk_z = (global_pos[2] / cs).floor() as i32 + active_center.z;

    let local_x = global_pos[0] - (chunk_x - active_center.x) as f32 * cs;
    let local_y = global_pos[1] - (chunk_y - active_center.y) as f32 * cs;
    let local_z = global_pos[2] - (chunk_z - active_center.z) as f32 * cs;

    (
        ChunkCoord::new(chunk_x, chunk_y, chunk_z),
        [local_x, local_y, local_z],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn chunk_coord_for_positive_positions() {
        let c = chunk_coord_for_position(70.0, 130.0);
        assert_eq!(c.x, 1); // 70 / 64 = 1.09 → floor = 1
        assert_eq!(c.z, 2); // 130 / 64 = 2.03 → floor = 2
    }

    #[test]
    fn chunk_coord_for_negative_positions() {
        let c = chunk_coord_for_position(-10.0, -130.0);
        assert_eq!(c.x, -1); // -10 / 64 = -0.15 → floor = -1
        assert_eq!(c.z, -3); // -130 / 64 = -2.03 → floor = -3
    }

    #[test]
    fn local_global_roundtrip() {
        let center = ChunkCoord::new(5, 0, 3);
        let chunk = ChunkCoord::new(6, 0, 4);
        let local = [10.0, 20.0, 30.0];

        let global = local_to_global(local, &chunk, &center);
        let (back_chunk, back_local) = global_to_local(global, &center);

        assert_eq!(back_chunk, chunk);
        assert_abs_diff_eq!(back_local[0], local[0], epsilon = 1e-4);
        assert_abs_diff_eq!(back_local[1], local[1], epsilon = 1e-4);
        assert_abs_diff_eq!(back_local[2], local[2], epsilon = 1e-4);
    }

    #[test]
    fn local_global_roundtrip_negative_chunk() {
        let center = ChunkCoord::new(-2, 0, -1);
        let chunk = ChunkCoord::new(-3, 0, -2);
        let local = [5.0, 10.0, 50.0];

        let global = local_to_global(local, &chunk, &center);
        let (back_chunk, back_local) = global_to_local(global, &center);

        assert_eq!(back_chunk, chunk);
        assert_abs_diff_eq!(back_local[0], local[0], epsilon = 1e-4);
        assert_abs_diff_eq!(back_local[1], local[1], epsilon = 1e-4);
        assert_abs_diff_eq!(back_local[2], local[2], epsilon = 1e-4);
    }

    #[test]
    fn empty_chunk_has_correct_snapshot_size() {
        let c = ChunkData::empty(ChunkCoord::new(0, 0, 0));
        assert_eq!(c.voxel_snapshot.len(), 64 * 64 * 64);
        assert!(!c.is_generated);
    }
}
