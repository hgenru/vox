//! # server
//!
//! GPU compute simulation orchestrator for the VOX voxel physics engine.
//!
//! Owns [`GpuSimulation`], which manages GPU buffers, compute pipelines,
//! and the dispatch chain for the MPM simulation:
//! `clear_grid -> P2G -> grid_update -> G2P -> (clear_voxels -> voxelize)`.
//!
//! All shaders are compiled to a single SPIR-V module at build time via
//! `spirv-builder` in `build.rs`.

mod passes;
pub mod push_constants;
mod readback;
pub mod simulation;

// ---------------------------------------------------------------------------
// Re-exports — preserve the original flat public API
// ---------------------------------------------------------------------------

pub use push_constants::{
    ComputeActivityPushConstants, ExplosionPushConstants, G2pPushConstants,
    GridUpdatePushConstants, GridUpdateSparsePushConstants, MarkActivePushConstants,
    P2gPushConstants, ReactPushConstants, RenderPushConstants, ToolbarPushConstants,
    VoxelizePushConstants, TOOLBAR_MAX_MATERIALS,
};
pub use simulation::GpuSimulation;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in the server simulation.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// GPU-core error.
    #[error("GPU error: {0}")]
    Gpu(#[from] gpu_core::GpuError),

    /// Too many particles.
    #[error("Particle count {0} exceeds maximum {MAX_PARTICLES}")]
    TooManyParticles(usize),
}

use shared::MAX_PARTICLES;

/// Result type alias for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::mem;

    use glam::Vec3;

    use gpu_core::VulkanContext;
    use shared::{MAX_PARTICLES, Particle};

    use crate::{
        GpuSimulation,
        push_constants::*,
    };

    fn init_tracing() {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    }

    #[test]
    fn push_constant_sizes() {
        assert_eq!(mem::size_of::<P2gPushConstants>(), 16);
        assert_eq!(mem::size_of::<GridUpdatePushConstants>(), 16);
        assert_eq!(mem::size_of::<G2pPushConstants>(), 32);
        assert_eq!(mem::size_of::<VoxelizePushConstants>(), 16);
        assert_eq!(mem::size_of::<ReactPushConstants>(), 16);
        // ExplosionPushConstants: 2 Vec4s (32 bytes) + 1 u32 + 3 u32 pad (16 bytes) = 48 bytes
        assert_eq!(mem::size_of::<ExplosionPushConstants>(), 48);
        // MarkActivePushConstants: 4 u32s = 16 bytes
        assert_eq!(mem::size_of::<MarkActivePushConstants>(), 16);
        // GridUpdateSparsePushConstants: u32 + f32 + f32 + u32 = 16 bytes
        assert_eq!(mem::size_of::<GridUpdateSparsePushConstants>(), 16);
        // ComputeActivityPushConstants: 4 u32s = 16 bytes
        assert_eq!(mem::size_of::<ComputeActivityPushConstants>(), 16);
        // RenderPushConstants: 4 u32s (16 bytes) + 2 Vec4s (32 bytes) = 48 bytes
        assert_eq!(mem::size_of::<RenderPushConstants>(), 48);
        // ToolbarPushConstants: 4 u32s (16 bytes) + 8 Vec4s (128 bytes) = 144 bytes
        assert_eq!(mem::size_of::<ToolbarPushConstants>(), 144);
    }

    #[test]
    fn create_and_destroy() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");
        sim.destroy(&ctx);
    }

    #[test]
    fn init_and_readback_particles() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let particles = vec![
            Particle::new(Vec3::new(16.0, 16.0, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(17.0, 16.0, 16.0), 1.0, 1, 1),
        ];

        sim.init_particles(&ctx, &particles)
            .expect("Failed to init particles");
        assert_eq!(sim.num_particles(), 2);

        let readback = sim
            .readback_particles(&ctx)
            .expect("Failed to readback particles");
        assert_eq!(readback.len(), 2);
        assert!((readback[0].position() - Vec3::new(16.0, 16.0, 16.0)).length() < 1e-5);
        assert!((readback[1].position() - Vec3::new(17.0, 16.0, 16.0)).length() < 1e-5);

        sim.destroy(&ctx);
    }

    #[test]
    fn step_does_not_crash() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let particles = vec![
            Particle::new(Vec3::new(16.0, 20.0, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(16.0, 20.5, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(16.5, 20.0, 16.0), 1.0, 0, 0),
        ];

        sim.init_particles(&ctx, &particles)
            .expect("Failed to init particles");

        // Execute one simulation step
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd);
        })
        .expect("Simulation step failed");

        let readback = sim
            .readback_particles(&ctx)
            .expect("Failed to readback after step");
        assert_eq!(readback.len(), 3);

        // After one step with gravity, particles should have moved slightly downward
        // (or at least not be NaN/Inf)
        for (i, p) in readback.iter().enumerate() {
            let pos = p.position();
            assert!(
                pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "Particle {i} has non-finite position: {pos:?}"
            );
        }

        sim.destroy(&ctx);
    }

    #[test]
    fn too_many_particles_returns_error() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let too_many = vec![Particle::new(Vec3::ZERO, 1.0, 0, 0); MAX_PARTICLES as usize + 1];
        let result = sim.init_particles(&ctx, &too_many);
        assert!(result.is_err());

        sim.destroy(&ctx);
    }
}
