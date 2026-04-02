//! Headless GPU integration test: verifies the full compute pipeline.
//!
//! Run with: `cargo test -p server --test gpu_integration -- --test-threads=1`
//! GPU tests must be single-threaded (see CLAUDE.md trap #14).

use glam::Vec3;
use gpu_core::VulkanContext;
use server::GpuSimulation;
use shared::{GRID_SIZE, MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle};

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
}

#[test]
fn water_falls_under_gravity() {
    init_tracing();

    // 1. Create headless VulkanContext
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

    // 2. Create GpuSimulation
    let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

    // 3. Create particles: stone floor + water cube
    let mut particles = Vec::new();

    // Stone floor at y=2..4
    for x in 4..28 {
        for z in 4..28 {
            for y in 2..4 {
                particles.push(Particle::new(
                    Vec3::new(x as f32, y as f32, z as f32),
                    1.0,
                    MAT_STONE,
                    PHASE_SOLID,
                ));
            }
        }
    }

    // Water cube at y=15..18 (center)
    for x in 13..19 {
        for z in 13..19 {
            for y in 15..18 {
                particles.push(Particle::new(
                    Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
                    1.0,
                    MAT_WATER,
                    PHASE_LIQUID,
                ));
            }
        }
    }

    let total_count = particles.len();

    // Record initial water center of mass
    let water_count = particles
        .iter()
        .filter(|p| p.material_id() == MAT_WATER)
        .count();
    let initial_water_y: f32 = particles
        .iter()
        .filter(|p| p.material_id() == MAT_WATER)
        .map(|p| p.position().y)
        .sum::<f32>()
        / water_count as f32;

    // 4. Upload particles
    sim.init_particles(&ctx, &particles)
        .expect("Failed to upload particles");

    // 5. Run 50 simulation steps
    for _ in 0..50 {
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd);
        })
        .expect("Simulation step failed");
    }

    // 6. Readback particles
    let result = sim
        .readback_particles(&ctx)
        .expect("Failed to readback particles");

    // 7. Assertions

    // Mass conserved (same particle count)
    assert_eq!(
        result.len(),
        total_count,
        "Particle count changed: expected {}, got {}",
        total_count,
        result.len()
    );

    // No NaN/Inf in any particle
    for (i, p) in result.iter().enumerate() {
        let pos = p.position();
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Particle {} has non-finite position: {:?}",
            i,
            pos
        );
    }

    // Particles stay in bounds (clamped to grid)
    let grid_max = GRID_SIZE as f32;
    for (i, p) in result.iter().enumerate() {
        let pos = p.position();
        assert!(
            pos.x >= 0.0 && pos.x <= grid_max,
            "Particle {} out of X bounds: {}",
            i,
            pos.x
        );
        assert!(
            pos.y >= 0.0 && pos.y <= grid_max,
            "Particle {} out of Y bounds: {}",
            i,
            pos.y
        );
        assert!(
            pos.z >= 0.0 && pos.z <= grid_max,
            "Particle {} out of Z bounds: {}",
            i,
            pos.z
        );
    }

    // Water should have moved down under gravity
    let water_particles: Vec<_> = result
        .iter()
        .filter(|p| p.material_id() == MAT_WATER)
        .collect();

    assert!(
        !water_particles.is_empty(),
        "No water particles found after simulation"
    );

    let final_water_y: f32 = water_particles.iter().map(|p| p.position().y).sum::<f32>()
        / water_particles.len() as f32;

    assert!(
        final_water_y < initial_water_y,
        "Water should fall under gravity: initial_y={:.2}, final_y={:.2}",
        initial_water_y,
        final_water_y
    );

    tracing::info!(
        "Water fell: {:.2} -> {:.2} (delta={:.2})",
        initial_water_y,
        final_water_y,
        initial_water_y - final_water_y
    );

    // Cleanup
    sim.destroy(&ctx);
}
