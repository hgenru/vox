//! Compute shader entry points for the MPM simulation pipeline and HUD overlays.
//!
//! Pipeline order: clear_grid -> P2G -> grid_update -> G2P -> voxelize
//! HUD overlays (toolbar_overlay) run after the render pass.
//!
//! The grid buffer has two binding modes:
//! - `&mut [GridCell]` for clear_grid, grid_update, G2P (one thread per cell, no contention)
//! - `&mut [f32]` for P2G (float atomics for concurrent particle scatter)
//!
//! Both views alias the same GPU buffer. Each `GridCell` = 12 contiguous f32s.

pub mod ca_compact;
pub mod ca_margolus;
pub mod ca_thermal;
pub mod clear_grid;
pub mod clear_grid_sparse;
pub mod clear_hash_grid;
pub mod compact_active_bricks;
pub mod compute_activity;
pub mod compute_dirty_tiles;
pub mod compute_occupancy;
pub mod count_per_brick;
pub mod explosion;
pub mod g2p;
pub mod g2p_sparse;
pub mod grid_update;
pub mod grid_update_hash;
pub mod grid_update_sparse;
pub mod hash_grid;
pub mod mark_active;
pub mod p2g;
pub mod p2g_sparse;
pub mod prefix_sum;
pub mod prepare_indirect;
pub mod react;
pub mod render;
pub mod render_far_field;
pub mod scatter_particles;
pub mod toolbar_overlay;
pub mod update_sleep;
pub mod voxelize;

/// Compute 1D quadratic B-spline weights for a particle at fractional position `fx`.
///
/// Returns weights for the 3 neighboring cells relative to the base cell.
/// Used by both P2G and G2P transfers.
pub fn quadratic_bspline_weights(fx: f32) -> [f32; 3] {
    let w0 = 0.5 * (1.5 - fx) * (1.5 - fx);
    let w1 = 0.75 - (fx - 1.0) * (fx - 1.0);
    let w2 = 0.5 * (fx - 0.5) * (fx - 0.5);
    [w0, w1, w2]
}
