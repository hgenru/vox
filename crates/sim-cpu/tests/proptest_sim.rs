//! Property-based tests for the CPU simulation.
//!
//! Uses proptest to verify invariants over random particle configurations:
//! - Mass conservation
//! - Energy bounded (no explosion)
//! - No NaN/Inf in positions or velocities
//! - Particles stay within grid bounds

use glam::Vec3;
use proptest::prelude::*;
use shared::{
    constants::DT,
    material::{MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID},
    particle::Particle,
};
use sim_cpu::world::Simulation;

/// Grid size for property tests (small for speed).
const PROP_GS: u32 = 16;

/// Generate a random position within the grid (with margin).
fn arb_position() -> impl Strategy<Value = Vec3> {
    let margin = 4.0 / PROP_GS as f32;
    let lo = margin;
    let hi = 1.0 - margin;
    (lo..hi, lo..hi, lo..hi).prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

/// Generate a random mass (positive, reasonable range).
fn arb_mass() -> impl Strategy<Value = f32> {
    0.1_f32..5.0
}

/// Generate a random particle.
fn arb_particle() -> impl Strategy<Value = Particle> {
    (arb_position(), arb_mass(), prop::bool::ANY).prop_map(|(pos, mass, is_water)| {
        if is_water {
            Particle::new(pos, mass, MAT_WATER, PHASE_LIQUID)
        } else {
            Particle::new(pos, mass, MAT_STONE, PHASE_SOLID)
        }
    })
}

/// Generate a random list of 1-20 particles.
fn arb_particles() -> impl Strategy<Value = Vec<Particle>> {
    prop::collection::vec(arb_particle(), 1..20)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn mass_conserved(particles in arb_particles()) {
        let initial_mass: f32 = particles.iter().map(|p| p.mass()).sum();
        let mut sim = Simulation::with_grid_size(particles, PROP_GS);

        for _ in 0..10 {
            sim.step(DT).unwrap();
        }

        let final_mass: f32 = sim.particles.iter().map(|p| p.mass()).sum();
        prop_assert!(
            (final_mass - initial_mass).abs() < 1e-4,
            "Mass not conserved: initial={}, final={}",
            initial_mass,
            final_mass
        );
    }

    #[test]
    fn no_nan_or_inf(particles in arb_particles()) {
        let mut sim = Simulation::with_grid_size(particles, PROP_GS);

        for _ in 0..10 {
            sim.step(DT).unwrap();
        }

        for (i, p) in sim.particles.iter().enumerate() {
            let pos = p.position();
            let vel = p.velocity();
            prop_assert!(
                pos.is_finite(),
                "Particle {} position is NaN/Inf: {:?}",
                i,
                pos
            );
            prop_assert!(
                vel.is_finite(),
                "Particle {} velocity is NaN/Inf: {:?}",
                i,
                vel
            );
        }
    }

    #[test]
    fn particles_stay_in_bounds(particles in arb_particles()) {
        let mut sim = Simulation::with_grid_size(particles, PROP_GS);

        for _ in 0..10 {
            sim.step(DT).unwrap();
        }

        for (i, p) in sim.particles.iter().enumerate() {
            let pos = p.position();
            prop_assert!(
                pos.x >= 0.0 && pos.x <= 1.0 && pos.y >= 0.0 && pos.y <= 1.0 && pos.z >= 0.0 && pos.z <= 1.0,
                "Particle {} out of bounds: {:?}",
                i,
                pos
            );
        }
    }

    #[test]
    fn energy_bounded(particles in arb_particles()) {
        let mut sim = Simulation::with_grid_size(particles, PROP_GS);

        // Run 20 steps
        for _ in 0..20 {
            sim.step(DT).unwrap();
        }

        let ke = sim.kinetic_energy();
        let total_mass = sim.total_mass();

        // Kinetic energy should be bounded: KE < mass * g * grid_height * safety_factor
        // Allow generous factor for numerical effects
        let max_expected_ke = total_mass * 100.0;

        prop_assert!(
            ke < max_expected_ke,
            "Kinetic energy too high (explosion?): KE={}, max_expected={}",
            ke,
            max_expected_ke
        );
    }

    #[test]
    fn deformation_gradient_finite(particles in arb_particles()) {
        let mut sim = Simulation::with_grid_size(particles, PROP_GS);

        for _ in 0..10 {
            sim.step(DT).unwrap();
        }

        for (i, p) in sim.particles.iter().enumerate() {
            let f = p.deformation_gradient();
            for col in 0..3 {
                for row in 0..3 {
                    prop_assert!(
                        f.col(col)[row].is_finite(),
                        "Particle {} F[{},{}] is NaN/Inf: {}",
                        i, row, col, f.col(col)[row]
                    );
                }
            }
        }
    }
}
