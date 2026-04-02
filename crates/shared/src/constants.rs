//! Simulation constants.

/// Grid dimension (cells per axis). 32³ for iteration-0.
pub const GRID_SIZE: u32 = 32;

/// Total number of grid cells.
pub const GRID_CELL_COUNT: u32 = GRID_SIZE * GRID_SIZE * GRID_SIZE;

/// Fixed simulation timestep (seconds).
pub const DT: f32 = 0.001;

/// Gravity acceleration (m/s², negative Y).
pub const GRAVITY: f32 = -9.81;

/// Brick size for BLAS acceleration structure (voxels per axis).
pub const BRICK_SIZE: u32 = 8;

/// Number of bricks per axis (GRID_SIZE / BRICK_SIZE).
pub const BRICKS_PER_AXIS: u32 = GRID_SIZE / BRICK_SIZE;

/// Render resolution width.
pub const RENDER_WIDTH: u32 = 1280;

/// Render resolution height.
pub const RENDER_HEIGHT: u32 = 720;

/// Maximum number of particles for MVP.
pub const MAX_PARTICLES: u32 = 50_000;

/// Physics tick rate (Hz).
pub const PHYSICS_HZ: u32 = 60;

/// Number of frames in flight for GPU synchronization.
pub const FRAMES_IN_FLIGHT: u32 = 2;
