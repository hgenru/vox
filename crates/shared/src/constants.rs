//! Simulation constants.

/// Grid dimension (cells per axis). 256³ with 5cm voxels = 12.8m world.
pub const GRID_SIZE: u32 = 256;

/// Total number of grid cells.
pub const GRID_CELL_COUNT: u32 = GRID_SIZE * GRID_SIZE * GRID_SIZE;

/// Fixed simulation timestep (seconds).
///
/// This is the default value. At runtime, prefer [`WorldConfig::dt()`](crate::WorldConfig::dt)
/// for per-world configuration.
pub const DT: f32 = 0.001;

/// Gravity acceleration (grid-units/s², negative Y).
/// Scaled 20x from 9.81 for 256³ grid with 5cm voxels.
///
/// This is the default value. At runtime, prefer [`WorldConfig::gravity`](crate::WorldConfig)
/// for per-world configuration.
pub const GRAVITY: f32 = -196.0;

/// Brick size for BLAS acceleration structure (voxels per axis).
pub const BRICK_SIZE: u32 = 8;

/// Number of bricks per axis (GRID_SIZE / BRICK_SIZE).
pub const BRICKS_PER_AXIS: u32 = GRID_SIZE / BRICK_SIZE;

/// Render resolution width.
pub const RENDER_WIDTH: u32 = 1280;

/// Render resolution height.
pub const RENDER_HEIGHT: u32 = 720;

/// Maximum number of particles for MVP.
pub const MAX_PARTICLES: u32 = 4_000_000;

/// Physics tick rate (Hz).
pub const PHYSICS_HZ: u32 = 60;

/// Number of frames in flight for GPU synchronization.
pub const FRAMES_IN_FLIGHT: u32 = 2;

/// Upper bound for the active cell list (one quarter of total grid cells).
pub const MAX_ACTIVE_CELLS: u32 = GRID_CELL_COUNT / 4;

/// Workgroup size for 1D sparse dispatches over active cell lists.
pub const SPARSE_WORKGROUP_SIZE: u32 = 64;

/// Total number of bricks in the grid (BRICKS_PER_AXIS³).
pub const TOTAL_BRICKS: u32 = BRICKS_PER_AXIS * BRICKS_PER_AXIS * BRICKS_PER_AXIS;

/// Tile size in pixels for dirty-tile rendering optimization.
pub const DIRTY_TILE_SIZE: u32 = 16;

/// Number of tiles across X for dirty-tile rendering.
pub const DIRTY_TILES_X: u32 = (RENDER_WIDTH + DIRTY_TILE_SIZE - 1) / DIRTY_TILE_SIZE;

/// Number of tiles across Y for dirty-tile rendering.
pub const DIRTY_TILES_Y: u32 = (RENDER_HEIGHT + DIRTY_TILE_SIZE - 1) / DIRTY_TILE_SIZE;

/// Total number of dirty tiles.
pub const DIRTY_TILE_COUNT: u32 = DIRTY_TILES_X * DIRTY_TILES_Y;

/// Default hash grid capacity (number of slots).
/// Should be ~2x the expected max active cells for low collision rate.
/// Must be a power of 2 for fast modulo via bitwise AND.
pub const HASH_GRID_DEFAULT_CAPACITY: u32 = 1 << 21; // 2M slots = ~104MB

/// Empty sentinel for hash grid keys.
/// Represents an unoccupied slot in the hash grid.
pub const HASH_GRID_EMPTY_KEY: u32 = 0xFFFF_FFFF;

/// Maximum number of linear probes before giving up on hash grid insert/lookup.
pub const HASH_GRID_MAX_PROBES: u32 = 128;

/// Super-brick size: number of bricks per super-brick axis (8 bricks = 64 voxels).
pub const SUPER_BRICK_SIZE: u32 = 8;

/// Number of super-bricks per axis (BRICKS_PER_AXIS / SUPER_BRICK_SIZE).
pub const SUPER_BRICKS_PER_AXIS: u32 = BRICKS_PER_AXIS / SUPER_BRICK_SIZE;

/// Total number of super-bricks in the grid (SUPER_BRICKS_PER_AXIS³).
pub const TOTAL_SUPER_BRICKS: u32 = SUPER_BRICKS_PER_AXIS * SUPER_BRICKS_PER_AXIS * SUPER_BRICKS_PER_AXIS;

/// Velocity² threshold below which a particle is considered inactive.
/// Particles with speed² <= this won't increment the activity counter.
pub const ACTIVITY_VELOCITY_THRESHOLD_SQ: f32 = 0.01;

/// Frames of zero activity before a brick enters sleep state.
/// At 60 Hz physics, 30 frames = 0.5 seconds of calm before sleeping.
pub const SLEEP_THRESHOLD: u32 = 30;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_bricks_matches_grid_layout() {
        // With GRID_SIZE=256 and BRICK_SIZE=8, BRICKS_PER_AXIS=32, so 32³=32768.
        assert_eq!(TOTAL_BRICKS, 32_768);
        assert_eq!(BRICKS_PER_AXIS, 32);
        assert_eq!(TOTAL_BRICKS, BRICKS_PER_AXIS.pow(3));
    }

    #[test]
    fn sleep_threshold_is_positive() {
        assert!(SLEEP_THRESHOLD > 0);
    }

    #[test]
    fn dirty_tile_constants() {
        // 1280 / 16 = 80 tiles, 720 / 16 = 45 tiles
        assert_eq!(DIRTY_TILES_X, 80);
        assert_eq!(DIRTY_TILES_Y, 45);
        assert_eq!(DIRTY_TILE_COUNT, 3600);
    }

    #[test]
    fn hash_grid_capacity_is_power_of_two() {
        assert!(HASH_GRID_DEFAULT_CAPACITY.is_power_of_two());
        assert!(HASH_GRID_DEFAULT_CAPACITY > 0);
        assert_eq!(HASH_GRID_DEFAULT_CAPACITY, 1 << 21); // 2M slots
    }

    #[test]
    fn hash_grid_empty_key_is_max_u32() {
        assert_eq!(HASH_GRID_EMPTY_KEY, u32::MAX);
    }

    #[test]
    fn super_brick_constants_consistent() {
        assert_eq!(SUPER_BRICKS_PER_AXIS, 4);
        assert_eq!(TOTAL_SUPER_BRICKS, 64);
        assert_eq!(TOTAL_SUPER_BRICKS, SUPER_BRICKS_PER_AXIS.pow(3));
    }
}
