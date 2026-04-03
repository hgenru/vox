//! Push constant structs that mirror shader-side definitions.
//!
//! Every struct here is `#[repr(C)]` + [`bytemuck::Pod`] + [`bytemuck::Zeroable`]
//! so it can be cast to raw bytes for `vkCmdPushConstants`.

use bytemuck::Pod;

/// Push constants for the P2G shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct P2gPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
}

/// Push constants for the grid update shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct GridUpdatePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y).
    pub gravity: f32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the G2P shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct G2pPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Number of valid phase transition rules.
    pub num_rules: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
    /// Padding to 32-byte alignment.
    pub _pad0: u32,
    /// Padding to 32-byte alignment.
    pub _pad1: u32,
    /// Padding to 32-byte alignment.
    pub _pad2: u32,
}

/// Push constants for the voxelize / clear_voxels shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct VoxelizePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Current simulation frame number (used with tick_period for graduated sleep).
    pub frame_number: u32,
    /// Padding.
    pub _pad0: u32,
}

/// Push constants for the render shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct RenderPushConstants {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Padding.
    pub _pad: u32,
    /// Camera eye position (xyz) + padding (w).
    pub eye: glam::Vec4,
    /// Camera target position (xyz) + padding (w).
    pub target: glam::Vec4,
}

/// Push constants for the react shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ReactPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

/// Push constants for the explosion shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ExplosionPushConstants {
    /// Explosion center (xyz), w unused.
    pub center: glam::Vec4,
    /// x = radius, y = strength, z = dt, w = num_particles as f32.
    pub params: glam::Vec4,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Push constants for the mark_active compute shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct MarkActivePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
}

/// Push constants for the sparse grid update shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct GridUpdateSparsePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y).
    pub gravity: f32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the compute_activity shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ComputeActivityPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the compute_occupancy shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ComputeOccupancyPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Number of bricks per axis (grid_size / brick_size).
    pub bricks_per_axis: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the update_sleep shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct UpdateSleepPushConstants {
    /// Total number of bricks in the grid.
    pub total_bricks: u32,
    /// Frames of zero activity before a brick enters sleep state.
    pub sleep_threshold: u32,
    /// Number of bricks per axis (grid_size / brick_size).
    pub bricks_per_axis: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the count_per_brick shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct CountPerBrickPushConstants {
    /// Total number of active particles.
    pub num_particles: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the prefix_sum shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct PrefixSumPushConstants {
    /// Total number of bricks.
    pub total_bricks: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
    /// Padding to 16-byte alignment.
    pub _pad2: u32,
}

/// Push constants for the scatter_particles shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ScatterParticlesPushConstants {
    /// Total number of active particles.
    pub num_particles: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Brick size (voxels per brick axis, e.g. 8).
    pub brick_size: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the compact_active_bricks shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct CompactActiveBricksPushConstants {
    /// Total number of bricks.
    pub total_bricks: u32,
    /// Padding to 16-byte alignment.
    pub _pad0: u32,
    /// Padding to 16-byte alignment.
    pub _pad1: u32,
    /// Padding to 16-byte alignment.
    pub _pad2: u32,
}

/// Maximum number of material slots in the toolbar.
pub const TOOLBAR_MAX_MATERIALS: usize = 8;

/// Push constants for the toolbar overlay shader.
///
/// Must match `ToolbarParams` in `shaders/src/compute/toolbar_overlay.rs`.
/// Total size: 16 bytes (header) + 128 bytes (8 x Vec4 colors) = 144 bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ToolbarPushConstants {
    /// Render target width in pixels.
    pub screen_width: u32,
    /// Render target height in pixels.
    pub screen_height: u32,
    /// Index of the currently selected material slot (0-based).
    pub selected_index: u32,
    /// Number of material slots to display.
    pub material_count: u32,
    /// RGBA colors for each material slot (up to 8).
    pub colors: [glam::Vec4; TOOLBAR_MAX_MATERIALS],
}
