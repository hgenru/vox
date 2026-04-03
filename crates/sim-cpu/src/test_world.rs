//! Test harness for integration-level behavior tests.
//!
//! [`TestWorld`] wraps the full MPM simulation pipeline (P2G, grid update, G2P,
//! thermal diffusion, phase transitions) behind a convenient builder API.
//! Intended for use in integration tests, not production code.

use glam::Vec3;
use shared::{
    constants::{DT, GRID_CELL_COUNT, GRID_SIZE},
    material::{MATERIAL_COUNT, MaterialParams, default_material_table},
    particle::{GridCell, Particle},
};

use crate::grid::{build_solid_occupancy, clear_grid, g2p, grid_update, p2g};
use crate::thermal::{apply_phase_transitions, diffuse_temperature};

/// A self-contained simulation world for behavior tests.
///
/// Owns particles, a grid buffer, and a material table. Provides helper
/// methods for spawning particles, stepping the simulation, and querying
/// aggregate statistics.
///
/// # Example
/// ```
/// use sim_cpu::test_world::TestWorld;
/// use shared::material::{MAT_WATER, PHASE_LIQUID};
/// use glam::Vec3;
///
/// let mut w = TestWorld::new();
/// w.spawn(Vec3::new(0.5, 0.5, 0.5), MAT_WATER, PHASE_LIQUID, 20.0);
/// w.step();
/// assert_eq!(w.particles().len(), 1);
/// ```
pub struct TestWorld {
    particles: Vec<Particle>,
    materials: [MaterialParams; MATERIAL_COUNT],
    grid: Vec<GridCell>,
}

impl TestWorld {
    /// Create a new empty test world with default materials.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            materials: default_material_table(),
            grid: vec![
                GridCell {
                    velocity_mass: glam::Vec4::ZERO,
                    force_pad: glam::Vec4::ZERO,
                    temp_pad: glam::Vec4::ZERO,
                };
                GRID_CELL_COUNT as usize
            ],
        }
    }

    /// Spawn a single particle at the given position with material, phase, and temperature.
    ///
    /// The particle is created with mass 1.0 and identity deformation gradient.
    /// Returns `&mut Self` for chaining.
    ///
    /// # Arguments
    /// - `pos`: position in world space (each axis in \[0, 1\])
    /// - `material`: material ID (e.g., `MAT_WATER`)
    /// - `phase`: phase (e.g., `PHASE_LIQUID`)
    /// - `temp`: initial temperature in degrees Celsius
    pub fn spawn(&mut self, pos: Vec3, material: u32, phase: u32, temp: f32) -> &mut Self {
        let mut p = Particle::new(pos, 1.0, material, phase);
        p.set_temperature(temp);
        self.particles.push(p);
        self
    }

    /// Spawn a cubic block of particles centered at `center`.
    ///
    /// The block spans `(2 * half_size + 1)^3` particles, each spaced one grid
    /// cell apart. All particles share the same material, phase, and temperature.
    ///
    /// # Arguments
    /// - `center`: center position in world space
    /// - `half_size`: half-extent in grid cells (e.g., 1 gives a 3x3x3 block)
    /// - `material`: material ID
    /// - `phase`: phase
    /// - `temp`: initial temperature
    pub fn spawn_block(
        &mut self,
        center: Vec3,
        half_size: i32,
        material: u32,
        phase: u32,
        temp: f32,
    ) -> &mut Self {
        let dx = 1.0 / GRID_SIZE as f32;
        for dz in -half_size..=half_size {
            for dy in -half_size..=half_size {
                for ddx in -half_size..=half_size {
                    let pos = center
                        + Vec3::new(ddx as f32 * dx, dy as f32 * dx, dz as f32 * dx);
                    self.spawn(pos, material, phase, temp);
                }
            }
        }
        self
    }

    /// Run one full simulation step.
    ///
    /// Executes the complete MPM pipeline in order:
    /// 1. Clear grid
    /// 2. Build solid occupancy
    /// 3. P2G (particle to grid transfer)
    /// 4. Grid update (gravity, boundaries)
    /// 5. G2P (grid to particle transfer)
    /// 6. Thermal diffusion
    /// 7. Phase transitions
    pub fn step(&mut self) {
        // 1. Clear grid (trap #6)
        clear_grid(&mut self.grid);

        // 2. Build solid occupancy for G2P support checks
        let solid_occupancy = build_solid_occupancy(&self.particles);

        // 3. P2G
        p2g(&self.particles, &mut self.grid, &self.materials, DT);

        // 4. Grid update
        grid_update(&mut self.grid, DT);

        // 5. G2P
        g2p(&mut self.particles, &self.grid, &solid_occupancy, DT);

        // 6. Thermal diffusion
        diffuse_temperature(&mut self.particles, &self.materials, DT);

        // 7. Phase transitions
        apply_phase_transitions(&mut self.particles);
    }

    /// Run `n` full simulation steps.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Get a reference to all particles.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Count the number of particles with the given material ID.
    pub fn count_material(&self, mat: u32) -> usize {
        self.particles.iter().filter(|p| p.material_id() == mat).count()
    }

    /// Count the number of particles with the given phase.
    pub fn count_phase(&self, phase: u32) -> usize {
        self.particles.iter().filter(|p| p.phase() == phase).count()
    }

    /// Compute the mean temperature of all particles.
    ///
    /// Returns 0.0 if there are no particles.
    pub fn avg_temperature(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.particles.iter().map(|p| p.temperature()).sum();
        sum / self.particles.len() as f32
    }

    /// Return the maximum temperature across all particles.
    ///
    /// Returns `f32::NEG_INFINITY` if there are no particles.
    pub fn max_temperature(&self) -> f32 {
        self.particles
            .iter()
            .map(|p| p.temperature())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Compute total thermal energy: sum of `T * mass` for all particles.
    pub fn total_thermal_energy(&self) -> f32 {
        self.particles
            .iter()
            .map(|p| p.temperature() * p.mass())
            .sum()
    }
}

impl Default for TestWorld {
    fn default() -> Self {
        Self::new()
    }
}
