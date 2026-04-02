//! CPU simulation orchestrator.
//!
//! Owns particles, grid, and material table. Provides `step()` which runs
//! the full MPM pipeline: clear grid -> P2G -> grid_update -> G2P.

use glam::Vec3;
use shared::{
    constants::{DT, GRID_CELL_COUNT, GRID_SIZE},
    material::{MATERIAL_COUNT, MaterialParams, default_material_table},
    particle::{GridCell, Particle},
};
use thiserror::Error;

use crate::grid::{clear_grid, g2p, grid_update, p2g};

/// Errors that can occur during simulation.
#[derive(Debug, Error)]
pub enum SimError {
    /// A particle position became NaN or infinite.
    #[error("particle {index} has non-finite position: {position:?}")]
    NonFinitePosition {
        /// Index of the problematic particle.
        index: usize,
        /// The non-finite position.
        position: Vec3,
    },
    /// Material ID is out of range.
    #[error("particle {index} has invalid material_id {material_id}")]
    InvalidMaterial {
        /// Index of the problematic particle.
        index: usize,
        /// The invalid material ID.
        material_id: u32,
    },
}

/// CPU reference MPM simulation.
///
/// Holds particles, a flat grid buffer, and the material parameter table.
/// Call `step()` to advance one timestep.
pub struct Simulation {
    /// All particles in the simulation.
    pub particles: Vec<Particle>,
    /// Flat grid buffer, indexed by `grid_index(ix, iy, iz)`.
    pub grid: Vec<GridCell>,
    /// Material parameter table (indexed by material_id).
    pub materials: [MaterialParams; MATERIAL_COUNT],
    /// Current simulation time.
    pub time: f32,
    /// Total steps executed.
    pub step_count: u64,
}

impl Simulation {
    /// Create a new simulation with the given particles and default materials.
    ///
    /// # Arguments
    /// - `particles`: initial particle list
    pub fn new(particles: Vec<Particle>) -> Self {
        Self {
            particles,
            grid: vec![
                GridCell {
                    velocity_mass: glam::Vec4::ZERO,
                    force_pad: glam::Vec4::ZERO,
                };
                GRID_CELL_COUNT as usize
            ],
            materials: default_material_table(),
            time: 0.0,
            step_count: 0,
        }
    }

    /// Create a new simulation with custom materials.
    ///
    /// # Arguments
    /// - `particles`: initial particle list
    /// - `materials`: material parameter table
    pub fn with_materials(
        particles: Vec<Particle>,
        materials: [MaterialParams; MATERIAL_COUNT],
    ) -> Self {
        Self {
            particles,
            grid: vec![
                GridCell {
                    velocity_mass: glam::Vec4::ZERO,
                    force_pad: glam::Vec4::ZERO,
                };
                GRID_CELL_COUNT as usize
            ],
            materials,
            time: 0.0,
            step_count: 0,
        }
    }

    /// Advance the simulation by one timestep.
    ///
    /// Runs the full MPM pipeline:
    /// 1. Clear grid
    /// 2. Particle-to-Grid transfer (P2G)
    /// 3. Grid update (gravity, boundaries)
    /// 4. Grid-to-Particle transfer (G2P)
    ///
    /// # Arguments
    /// - `dt`: timestep duration (typically `DT` from constants)
    ///
    /// # Returns
    /// `Ok(())` on success, or `Err(SimError)` if simulation state is invalid.
    pub fn step(&mut self, dt: f32) -> Result<(), SimError> {
        // Validate particles before stepping
        self.validate()?;

        // 1. Clear grid (CRITICAL: must be done before P2G, see trap #6)
        clear_grid(&mut self.grid);

        // 2. Particle-to-Grid transfer
        p2g(&self.particles, &mut self.grid, &self.materials, dt);

        // 3. Grid update: momentum → velocity, gravity, boundaries
        grid_update(&mut self.grid, dt);

        // 4. Grid-to-Particle transfer
        g2p(&mut self.particles, &self.grid, dt);

        self.time += dt;
        self.step_count += 1;

        Ok(())
    }

    /// Run multiple simulation steps with the default timestep.
    ///
    /// # Arguments
    /// - `steps`: number of steps to run
    pub fn run(&mut self, steps: u32) -> Result<(), SimError> {
        for _ in 0..steps {
            self.step(DT)?;
        }
        Ok(())
    }

    /// Validate all particles for finite positions and valid material IDs.
    fn validate(&self) -> Result<(), SimError> {
        for (i, p) in self.particles.iter().enumerate() {
            let pos = p.position();
            if !pos.is_finite() {
                return Err(SimError::NonFinitePosition {
                    index: i,
                    position: pos,
                });
            }
            if p.material_id() as usize >= MATERIAL_COUNT {
                return Err(SimError::InvalidMaterial {
                    index: i,
                    material_id: p.material_id(),
                });
            }
        }
        Ok(())
    }

    /// Get the number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Get the grid dimensions.
    pub fn grid_size(&self) -> u32 {
        GRID_SIZE
    }

    /// Compute total kinetic energy of all particles.
    pub fn kinetic_energy(&self) -> f32 {
        self.particles
            .iter()
            .map(|p| 0.5 * p.mass() * p.velocity().length_squared())
            .sum()
    }

    /// Compute total mass of all particles.
    pub fn total_mass(&self) -> f32 {
        self.particles.iter().map(|p| p.mass()).sum()
    }

    /// Compute center of mass position.
    pub fn center_of_mass(&self) -> Vec3 {
        let total_mass = self.total_mass();
        if total_mass < 1e-10 {
            return Vec3::ZERO;
        }
        let weighted_pos: Vec3 = self
            .particles
            .iter()
            .map(|p| p.position() * p.mass())
            .fold(Vec3::ZERO, |a, b| a + b);
        weighted_pos / total_mass
    }
}

/// Create a block of water particles in a cubic region.
///
/// # Arguments
/// - `min`: minimum corner of the block (world space, [0,1]^3)
/// - `max`: maximum corner of the block
/// - `spacing`: distance between particles
/// - `mass_per_particle`: mass of each particle
/// - `material_id`: material ID (e.g., `MAT_WATER`)
/// - `phase`: phase (e.g., `PHASE_LIQUID`)
pub fn create_particle_block(
    min: Vec3,
    max: Vec3,
    spacing: f32,
    mass_per_particle: f32,
    material_id: u32,
    phase: u32,
) -> Vec<Particle> {
    let mut particles = Vec::new();
    let mut x = min.x;
    while x < max.x {
        let mut y = min.y;
        while y < max.y {
            let mut z = min.z;
            while z < max.z {
                particles.push(Particle::new(
                    Vec3::new(x, y, z),
                    mass_per_particle,
                    material_id,
                    phase,
                ));
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }
    particles
}

#[cfg(test)]
mod tests {
    use shared::material::{MAT_WATER, PHASE_LIQUID};

    use super::*;

    #[test]
    fn simulation_step_runs() {
        let particles = vec![Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, 1)];
        let mut sim = Simulation::new(particles);
        sim.step(DT).unwrap();
        assert_eq!(sim.step_count, 1);
        assert!((sim.time - DT).abs() < 1e-8);
    }

    #[test]
    fn simulation_run_multiple_steps() {
        let particles = vec![Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, 1)];
        let mut sim = Simulation::new(particles);
        sim.run(100).unwrap();
        assert_eq!(sim.step_count, 100);
    }

    #[test]
    fn water_cube_falls() {
        let particles = create_particle_block(
            Vec3::new(0.35, 0.5, 0.35),
            Vec3::new(0.65, 0.8, 0.65),
            0.05,
            0.1,
            MAT_WATER,
            PHASE_LIQUID,
        );

        let initial_com_y = {
            let total_mass: f32 = particles.iter().map(|p| p.mass()).sum();
            let weighted_y: f32 = particles.iter().map(|p| p.position().y * p.mass()).sum();
            weighted_y / total_mass
        };

        let mut sim = Simulation::new(particles);
        sim.run(100).unwrap();

        let final_com_y = sim.center_of_mass().y;
        assert!(
            final_com_y < initial_com_y,
            "Water cube should fall: initial COM y={initial_com_y}, final={final_com_y}"
        );
    }

    #[test]
    fn mass_conserved_over_steps() {
        let particles = create_particle_block(
            Vec3::new(0.4, 0.5, 0.4),
            Vec3::new(0.6, 0.7, 0.6),
            0.05,
            0.5,
            MAT_WATER,
            PHASE_LIQUID,
        );

        let initial_mass = particles.iter().map(|p| p.mass()).sum::<f32>();
        let mut sim = Simulation::new(particles);
        sim.run(50).unwrap();

        let final_mass = sim.total_mass();
        assert!(
            (final_mass - initial_mass).abs() < 1e-6,
            "Mass not conserved: initial={initial_mass}, final={final_mass}"
        );
    }

    #[test]
    fn no_nan_after_many_steps() {
        let particles = create_particle_block(
            Vec3::new(0.3, 0.6, 0.3),
            Vec3::new(0.7, 0.9, 0.7),
            0.06,
            0.2,
            MAT_WATER,
            PHASE_LIQUID,
        );

        let mut sim = Simulation::new(particles);
        sim.run(200).unwrap();

        for (i, p) in sim.particles.iter().enumerate() {
            assert!(
                p.position().is_finite(),
                "Particle {i} position is NaN after 200 steps"
            );
            assert!(
                p.velocity().is_finite(),
                "Particle {i} velocity is NaN after 200 steps"
            );
        }
    }

    #[test]
    fn create_particle_block_count() {
        let particles = create_particle_block(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.1, 0.1),
            0.05,
            1.0,
            0,
            0,
        );
        // 0.0, 0.05 = 2 per axis => 2^3 = 8
        assert_eq!(particles.len(), 8);
    }

    #[test]
    fn kinetic_energy_increases_under_gravity() {
        let particles = vec![Particle::new(Vec3::new(0.5, 0.8, 0.5), 1.0, MAT_WATER, 1)];
        let mut sim = Simulation::new(particles);
        let ke_initial = sim.kinetic_energy();
        sim.run(20).unwrap();
        let ke_final = sim.kinetic_energy();

        assert!(
            ke_final > ke_initial,
            "Kinetic energy should increase under gravity: initial={ke_initial}, final={ke_final}"
        );
    }

    #[test]
    fn validate_catches_bad_material() {
        let particles = vec![Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, 99, 0)];
        let mut sim = Simulation::new(particles);
        let result = sim.step(DT);
        assert!(result.is_err());
    }
}
