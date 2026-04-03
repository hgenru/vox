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
pub const MAX_PARTICLES: u32 = 2_000_000;

/// Physics tick rate (Hz).
pub const PHYSICS_HZ: u32 = 60;

/// Number of frames in flight for GPU synchronization.
pub const FRAMES_IN_FLIGHT: u32 = 2;

/// Upper bound for the active cell list (one quarter of total grid cells).
pub const MAX_ACTIVE_CELLS: u32 = GRID_CELL_COUNT / 4;

/// Workgroup size for 1D sparse dispatches over active cell lists.
pub const SPARSE_WORKGROUP_SIZE: u32 = 64;
