//! CPU reference implementation of MPM grid operations: P2G, grid_update, G2P.
//!
//! Uses quadratic B-spline weights for particle-grid transfers with
//! 27 neighbor cells per particle (3x3x3 stencil).

use glam::{Mat3, Vec3};
use shared::{
    constants::{GRAVITY, GRID_SIZE},
    material::MaterialParams,
    particle::{GridCell, Particle},
    physics::constitutive_stress,
};

/// Compute gravity scaled to a given grid size.
///
/// The default `GRAVITY` constant (-196.0) is tuned for `GRID_SIZE=256` (5cm voxels).
/// For a different grid size, gravity must scale proportionally:
/// `gravity = GRAVITY * grid_size / GRID_SIZE`.
#[inline]
fn scaled_gravity(grid_size: u32) -> f32 {
    GRAVITY * grid_size as f32 / GRID_SIZE as f32
}

/// Mass threshold below which a grid cell is considered empty.
const MASS_THRESHOLD: f32 = 1e-8;

/// Quadratic B-spline weight function.
///
/// For distance `d` from particle to grid node (in grid units):
/// - `|d| < 0.5`: `0.75 - d^2`
/// - `0.5 <= |d| < 1.5`: `0.5 * (1.5 - |d|)^2`
/// - `|d| >= 1.5`: `0`
#[inline]
pub fn bspline_weight(d: f32) -> f32 {
    let abs_d = d.abs();
    if abs_d < 0.5 {
        0.75 - abs_d * abs_d
    } else if abs_d < 1.5 {
        let t = 1.5 - abs_d;
        0.5 * t * t
    } else {
        0.0
    }
}

/// Compute the gradient of the quadratic B-spline weight.
///
/// Returns `dw/dx` for distance `d`.
#[inline]
fn bspline_weight_gradient(d: f32) -> f32 {
    let abs_d = d.abs();
    if abs_d < 0.5 {
        -2.0 * d
    } else if abs_d < 1.5 {
        let sign = if d >= 0.0 { 1.0 } else { -1.0 };
        -sign * (1.5 - abs_d)
    } else {
        0.0
    }
}

/// Convert a 3D grid index `(ix, iy, iz)` to a flat index.
///
/// Uses the compile-time `GRID_SIZE` constant. For runtime grid sizes,
/// use [`grid_index_sized`].
///
/// Returns `None` if any index is out of bounds.
#[inline]
pub fn grid_index(ix: i32, iy: i32, iz: i32) -> Option<usize> {
    grid_index_sized(ix, iy, iz, GRID_SIZE)
}

/// Convert a 3D grid index `(ix, iy, iz)` to a flat index for the given `grid_size`.
///
/// Returns `None` if any index is out of bounds.
#[inline]
pub fn grid_index_sized(ix: i32, iy: i32, iz: i32, grid_size: u32) -> Option<usize> {
    let gs = grid_size as i32;
    if ix < 0 || iy < 0 || iz < 0 || ix >= gs || iy >= gs || iz >= gs {
        return None;
    }
    Some((iz * gs * gs + iy * gs + ix) as usize)
}

/// Clear all grid cells to zero (mass, velocity, force).
///
/// This MUST be called before every P2G pass. See CLAUDE.md trap #6.
pub fn clear_grid(grid: &mut [GridCell]) {
    for cell in grid.iter_mut() {
        *cell = GridCell {
            velocity_mass: glam::Vec4::ZERO,
            force_pad: glam::Vec4::ZERO,
            temp_pad: glam::Vec4::ZERO,
        };
    }
}

/// Particle-to-Grid transfer (P2G) using compile-time `GRID_SIZE`.
///
/// Convenience wrapper around [`p2g_sized`].
pub fn p2g(particles: &[Particle], grid: &mut [GridCell], materials: &[MaterialParams], dt: f32) {
    p2g_sized(particles, grid, materials, dt, GRID_SIZE);
}

/// Particle-to-Grid transfer (P2G).
///
/// Scatters mass, momentum, and stress from each particle to its 27 neighboring
/// grid cells using quadratic B-spline weights.
///
/// The stress contribution follows the APIC formulation:
/// `momentum += w * (mass * vel + mass * C * (xi - xp)) + dt * w * stress_force`
///
/// # Arguments
/// - `particles`: slice of all particles
/// - `grid`: mutable grid buffer (must be pre-cleared)
/// - `materials`: material parameter table
/// - `dt`: timestep
/// - `grid_size`: number of cells per axis
pub fn p2g_sized(
    particles: &[Particle],
    grid: &mut [GridCell],
    materials: &[MaterialParams],
    dt: f32,
    grid_size: u32,
) {
    let gs = grid_size as f32;
    let inv_dx = gs; // grid spacing = 1/grid_size in normalized coords
    let _ = inv_dx;

    for particle in particles {
        let pos = particle.position();
        let vel = particle.velocity();
        let mass = particle.mass();
        let temp = particle.temperature();
        let c = particle.affine_momentum();

        // Get material and compute stress
        let mat_id = particle.material_id() as usize;
        let mat = if mat_id < materials.len() {
            &materials[mat_id]
        } else {
            continue;
        };

        let stress = constitutive_stress(particle, mat);

        // Volume estimate: for MPM, particle volume Vp = 1/density * mass
        // In grid space, each cell has volume (dx)^3 = (1/GRID_SIZE)^3
        // We use Vp = (dx)^3 for simplicity (one particle per cell initially)
        let dx = 1.0 / gs;
        let volume = dx * dx * dx;

        // Stress contribution: -dt * Vp * stress (Kirchhoff stress -> force)
        let stress_force = -dt * volume * stress;

        // Base grid cell index (particle position in grid coordinates)
        let grid_pos = pos * gs;
        let base_x = (grid_pos.x - 0.5).floor() as i32;
        let base_y = (grid_pos.y - 0.5).floor() as i32;
        let base_z = (grid_pos.z - 0.5).floor() as i32;

        // Scatter to 3x3x3 neighbor cells
        for dz in 0..3_i32 {
            for dy in 0..3_i32 {
                for dx_i in 0..3_i32 {
                    let ix = base_x + dx_i;
                    let iy = base_y + dy;
                    let iz = base_z + dz;

                    let idx = match grid_index_sized(ix, iy, iz, grid_size) {
                        Some(i) => i,
                        None => continue,
                    };

                    // Grid node position in world space
                    let xi = Vec3::new(ix as f32 / gs, iy as f32 / gs, iz as f32 / gs);

                    // Distances in grid units
                    let dist = grid_pos - Vec3::new(ix as f32, iy as f32, iz as f32);

                    let wx = bspline_weight(dist.x);
                    let wy = bspline_weight(dist.y);
                    let wz = bspline_weight(dist.z);
                    let w = wx * wy * wz;

                    if w < 1e-12 {
                        continue;
                    }

                    // APIC momentum transfer
                    let dpos = xi - pos;
                    let momentum = mass * (vel + c * dpos);

                    // Stress force contribution (using weight gradient)
                    let dwx = bspline_weight_gradient(dist.x);
                    let dwy = bspline_weight_gradient(dist.y);
                    let dwz = bspline_weight_gradient(dist.z);

                    let grad_w = Vec3::new(dwx * wy * wz, wx * dwy * wz, wx * wy * dwz) * gs;

                    let force_contrib = stress_force * grad_w;

                    // Accumulate into grid cell
                    let cell = &mut grid[idx];
                    cell.velocity_mass.x += w * momentum.x + force_contrib.x;
                    cell.velocity_mass.y += w * momentum.y + force_contrib.y;
                    cell.velocity_mass.z += w * momentum.z + force_contrib.z;
                    cell.velocity_mass.w += w * mass;

                    // Scatter weighted temperature (temp * mass * weight).
                    // Normalized by mass in grid_update to get average temperature.
                    cell.temp_pad.x += temp * w * mass;
                }
            }
        }
    }
}

/// Grid update using compile-time `GRID_SIZE`.
///
/// Convenience wrapper around [`grid_update_sized`].
pub fn grid_update(grid: &mut [GridCell], dt: f32) {
    grid_update_sized(grid, dt, GRID_SIZE);
}

/// Grid update: convert momentum to velocity, apply gravity and boundary conditions.
///
/// For each cell with sufficient mass:
/// 1. Divide momentum by mass to get velocity
/// 2. Apply gravity: `vel.y += gravity * dt`
/// 3. Enforce boundary conditions at grid edges (velocity clamping)
///
/// Gravity is automatically scaled to the given `grid_size` relative to `GRID_SIZE=256`.
///
/// # Arguments
/// - `grid`: mutable grid buffer (contains accumulated momentum from P2G)
/// - `dt`: timestep
/// - `grid_size`: number of cells per axis
pub fn grid_update_sized(grid: &mut [GridCell], dt: f32, grid_size: u32) {
    let gs = grid_size as i32;
    let boundary = 3; // boundary cells thickness

    for iz in 0..gs {
        for iy in 0..gs {
            for ix in 0..gs {
                let idx = (iz * gs * gs + iy * gs + ix) as usize;
                let cell = &mut grid[idx];
                let mass = cell.velocity_mass.w;

                if mass < MASS_THRESHOLD {
                    cell.velocity_mass = glam::Vec4::ZERO;
                    cell.temp_pad = glam::Vec4::ZERO;
                    continue;
                }

                // Convert momentum to velocity
                let inv_mass = 1.0 / mass;
                cell.velocity_mass.x *= inv_mass;
                cell.velocity_mass.y *= inv_mass;
                cell.velocity_mass.z *= inv_mass;

                // Apply gravity (scaled for grid_size)
                cell.velocity_mass.y += scaled_gravity(grid_size) * dt;

                // Boundary conditions: clamp velocity at grid edges
                // Sticky boundary (zero velocity at walls)
                if ix < boundary || ix >= gs - boundary {
                    cell.velocity_mass.x = 0.0;
                }
                if iy < boundary {
                    // Floor: only clamp downward velocity
                    if cell.velocity_mass.y < 0.0 {
                        cell.velocity_mass.y = 0.0;
                    }
                }
                if iy >= gs - boundary {
                    // Ceiling: only clamp upward velocity
                    if cell.velocity_mass.y > 0.0 {
                        cell.velocity_mass.y = 0.0;
                    }
                }
                if iz < boundary || iz >= gs - boundary {
                    cell.velocity_mass.z = 0.0;
                }

                // Normalize accumulated temperature by mass.
                // P2G scattered temp * mass * weight; dividing by mass gives
                // the mass-weighted average temperature at this grid cell.
                let cell_temp = cell.temp_pad.x / mass;
                cell.temp_pad = glam::Vec4::new(cell_temp, 0.0, 0.0, 0.0);
            }
        }
    }
}

/// Build a boolean occupancy grid using compile-time `GRID_SIZE`.
///
/// Convenience wrapper around [`build_solid_occupancy_sized`].
pub fn build_solid_occupancy(particles: &[Particle]) -> Vec<bool> {
    build_solid_occupancy_sized(particles, GRID_SIZE)
}

/// Build a boolean occupancy grid of solid particles for support checks.
///
/// Returns a `Vec<bool>` of size `grid_size^3` where each entry is `true`
/// if a solid (phase==0) particle maps to that cell.
///
/// # Arguments
/// - `particles`: slice of all particles
/// - `grid_size`: number of cells per axis
pub fn build_solid_occupancy_sized(particles: &[Particle], grid_size: u32) -> Vec<bool> {
    let gs = grid_size as usize;
    let mut occupancy = vec![false; gs * gs * gs];
    for p in particles {
        if p.phase() != 0 {
            continue;
        }
        let pos = p.position();
        let gp = pos * grid_size as f32;
        let ix = gp.x as i32;
        let iy = gp.y as i32;
        let iz = gp.z as i32;
        if let Some(idx) = grid_index_sized(ix, iy, iz, grid_size) {
            occupancy[idx] = true;
        }
    }
    occupancy
}

/// Check if a solid particle has support using compile-time `GRID_SIZE`.
///
/// Convenience wrapper around [`check_solid_support_sized`].
pub fn check_solid_support(pos: Vec3, solid_occupancy: &[bool]) -> bool {
    check_solid_support_sized(pos, solid_occupancy, GRID_SIZE)
}

/// Check if a solid particle has support from a solid voxel below it.
///
/// A particle is considered supported if:
/// - It is at the grid floor (iy <= 1), or
/// - The cell directly below (iy-1) or 2 cells below (iy-2) in the occupancy
///   grid is occupied by a solid.
///
/// Checking 2 cells below avoids stale-read artifacts where the voxel buffer
/// lags by one frame, causing adjacent falling solids to lose support prematurely.
///
/// # Arguments
/// - `pos`: particle position in world space [0, 1]
/// - `solid_occupancy`: boolean grid from [`build_solid_occupancy`]
/// - `grid_size`: number of cells per axis
pub fn check_solid_support_sized(pos: Vec3, solid_occupancy: &[bool], grid_size: u32) -> bool {
    let gs = grid_size as f32;
    let gp = pos * gs;
    let ix = gp.x as i32;
    let iy = gp.y as i32;
    let iz = gp.z as i32;

    // At bottom of grid = supported by boundary
    if iy <= 1 {
        return true;
    }

    // Check cell directly below
    match grid_index_sized(ix, iy - 1, iz, grid_size) {
        Some(idx) => {
            if solid_occupancy[idx] {
                return true;
            }
        }
        None => return true, // out of bounds = treat as supported (safe default)
    }

    // Check 2 cells below to handle stale voxel buffer
    if iy >= 3 {
        match grid_index_sized(ix, iy - 2, iz, grid_size) {
            Some(idx) => {
                if solid_occupancy[idx] {
                    return true;
                }
            }
            None => return true,
        }
    }

    false
}

/// Grid-to-Particle transfer (G2P) using compile-time `GRID_SIZE`.
///
/// Convenience wrapper around [`g2p_sized`].
pub fn g2p(particles: &mut [Particle], grid: &[GridCell], solid_occupancy: &[bool], dt: f32) {
    g2p_sized(particles, grid, solid_occupancy, dt, GRID_SIZE);
}

/// Grid-to-Particle transfer (G2P).
///
/// Gathers velocity from 27 neighbor grid cells to update each particle:
/// 1. PIC/FLIP blend for velocity (alpha=0.95 PIC, 0.05 FLIP)
/// 2. Update APIC C matrix (affine velocity field)
/// 3. Update deformation gradient: `F_new = (I + dt*C) * F`
/// 4. Advect position: `pos += vel * dt`
///
/// For solid particles (phase==0), checks support from the cell below.
/// Supported solids update only F/C (pinned). Unsupported solids fall normally.
///
/// # Arguments
/// - `particles`: mutable slice of particles
/// - `grid`: grid buffer with updated velocities
/// - `solid_occupancy`: boolean grid from [`build_solid_occupancy`] (pass empty slice to skip support check)
/// - `dt`: timestep
/// - `grid_size`: number of cells per axis
pub fn g2p_sized(
    particles: &mut [Particle],
    grid: &[GridCell],
    solid_occupancy: &[bool],
    dt: f32,
    grid_size: u32,
) {
    let gs = grid_size as f32;
    let pic_blend = 0.7_f32; // PIC/FLIP blend factor

    for particle in particles.iter_mut() {
        let pos = particle.position();
        let old_vel = particle.velocity();

        let grid_pos = pos * gs;
        let base_x = (grid_pos.x - 0.5).floor() as i32;
        let base_y = (grid_pos.y - 0.5).floor() as i32;
        let base_z = (grid_pos.z - 0.5).floor() as i32;

        let mut new_vel = Vec3::ZERO;
        let mut new_c = Mat3::ZERO;
        let mut new_temp = 0.0_f32;

        // Gather from 3x3x3 neighbors
        for dz in 0..3_i32 {
            for dy in 0..3_i32 {
                for dx_i in 0..3_i32 {
                    let ix = base_x + dx_i;
                    let iy = base_y + dy;
                    let iz = base_z + dz;

                    let idx = match grid_index_sized(ix, iy, iz, grid_size) {
                        Some(i) => i,
                        None => continue,
                    };

                    let xi = Vec3::new(ix as f32 / gs, iy as f32 / gs, iz as f32 / gs);

                    let dist = grid_pos - Vec3::new(ix as f32, iy as f32, iz as f32);

                    let wx = bspline_weight(dist.x);
                    let wy = bspline_weight(dist.y);
                    let wz = bspline_weight(dist.z);
                    let w = wx * wy * wz;

                    if w < 1e-12 {
                        continue;
                    }

                    let cell = &grid[idx];
                    let grid_vel = Vec3::new(
                        cell.velocity_mass.x,
                        cell.velocity_mass.y,
                        cell.velocity_mass.z,
                    );

                    // PIC velocity
                    new_vel += w * grid_vel;

                    // APIC C matrix: C = sum(w * v * (xi - xp)^T) * 4/dx^2
                    // The 4/dx^2 factor comes from the B-spline inertia tensor
                    let dpos = xi - pos;
                    let inv_d = 4.0 * gs * gs; // 4 / dx^2
                    new_c += Mat3::from_cols(
                        w * grid_vel * dpos.x * inv_d,
                        w * grid_vel * dpos.y * inv_d,
                        w * grid_vel * dpos.z * inv_d,
                    );

                    // Gather temperature from grid (normalized in grid_update).
                    let grid_temp = cell.temp_pad.x;
                    new_temp += grid_temp * w;
                }
            }
        }

        // Thermal diffusion: blend grid temp with particle temp for gradual transfer.
        // Must match shader g2p.rs thermal_blend value (0.002 = 0.2% per step).
        let thermal_blend = 0.002_f32;
        let old_temp = particle.temperature();
        new_temp = old_temp + thermal_blend * (new_temp - old_temp);

        // Radiative cooling: particles lose heat to environment (Newton's cooling law).
        // Must match shader g2p.rs apply_radiative_cooling().
        // Prevents thermal runaway from repeated explosions.
        let ambient_temp = 20.0_f32;
        let cooling_rate = 0.0005_f32;
        new_temp = new_temp + cooling_rate * (ambient_temp - new_temp);

        // PIC/FLIP blend
        // True FLIP requires old grid velocities which we don't store.
        // Instead, blend PIC (grid velocity) with old particle velocity
        // to reduce excessive damping while maintaining stability.
        let blended_vel = pic_blend * new_vel + (1.0 - pic_blend) * old_vel;

        let mut final_vel = blended_vel;

        let gravity = scaled_gravity(grid_size);

        // Gas buoyancy: counteract most of gravity and add slight upward force
        // so steam rises and persists visually instead of falling immediately.
        if particle.phase() == 2 {
            final_vel.y += (-gravity) * 0.7 * dt;
            final_vel.x *= 0.98;
            final_vel.z *= 0.98;
        }

        // Update deformation gradient: F_new = (I + dt * C) * F
        let f_old = particle.deformation_gradient();
        let f_new = (Mat3::IDENTITY + dt * new_c) * f_old;

        // Solids: check if the particle has support from a solid voxel below.
        // Supported solids update F and C only (pinned, prevents floor drift).
        // Unsupported solids fall normally with minimum fall speed to overcome grid friction.
        if particle.phase() == 0 && !solid_occupancy.is_empty() {
            let supported = check_solid_support_sized(pos, solid_occupancy, grid_size);
            if supported {
                particle.set_affine_momentum(new_c);
                particle.set_deformation_gradient(f_new);
                // Update temperature even for pinned solids (heat conducts through stone)
                particle.set_temperature(new_temp);
                continue;
            }
            // Unsupported solid: bypass PIC blend entirely.
            // Use old particle velocity + gravity directly so blocks accelerate
            // like real falling objects, not crawl one cell per frame.
            final_vel = Vec3::new(
                old_vel.x * 0.9,
                old_vel.y + gravity * dt, // accumulate gravity (scaled)
                old_vel.z * 0.9,
            );
            // Terminal velocity scaled with gravity
            let terminal_vel = -50.0 * grid_size as f32 / GRID_SIZE as f32;
            if final_vel.y < terminal_vel {
                final_vel.y = terminal_vel;
            }
        }

        // Update velocity and temperature
        particle.set_velocity(final_vel);
        particle.set_temperature(new_temp);

        // Update APIC C matrix
        particle.set_affine_momentum(new_c);

        // Update deformation gradient
        particle.set_deformation_gradient(f_new);

        // Advect position
        let new_pos = pos + final_vel * dt;

        // Clamp to grid bounds
        let margin = 2.0 / gs;
        let clamped_pos = Vec3::new(
            new_pos.x.clamp(margin, 1.0 - margin),
            new_pos.y.clamp(margin, 1.0 - margin),
            new_pos.z.clamp(margin, 1.0 - margin),
        );

        particle.set_position(clamped_pos);
    }
}

#[cfg(test)]
mod tests {
    use shared::material::{MAT_WATER, default_material_table};

    use super::*;

    /// Test grid size: small grid for fast unit tests.
    const TEST_GS: u32 = 32;

    fn make_grid() -> Vec<GridCell> {
        vec![
            GridCell {
                velocity_mass: glam::Vec4::ZERO,
                force_pad: glam::Vec4::ZERO,
                temp_pad: glam::Vec4::ZERO,
            };
            (TEST_GS * TEST_GS * TEST_GS) as usize
        ]
    }

    #[test]
    fn bspline_weights_sum_to_one() {
        // For any position within the grid, the quadratic B-spline weights
        // over the 3-node stencil should sum to 1 in each axis.
        // For particle at grid position xp, base = floor(xp - 0.5),
        // distances to nodes base+0, base+1, base+2 are xp-base, xp-base-1, xp-base-2
        for xp in [5.0, 5.1, 5.25, 5.5, 5.75, 5.99] {
            let base = (xp - 0.5_f32).floor() as i32;
            let sum: f32 = (0..3).map(|i| bspline_weight(xp - (base + i) as f32)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "B-spline weights don't sum to 1 for xp={xp}: sum={sum}"
            );
        }
    }

    #[test]
    fn clear_grid_zeroes_all() {
        let mut grid = make_grid();
        grid[0].velocity_mass.x = 42.0;
        grid[100].velocity_mass.w = 7.0;
        clear_grid(&mut grid);

        for cell in &grid {
            assert_eq!(cell.velocity_mass, glam::Vec4::ZERO);
            assert_eq!(cell.force_pad, glam::Vec4::ZERO);
        }
    }

    #[test]
    fn p2g_mass_conservation() {
        let table = default_material_table();
        // Place particles well inside the grid (away from boundaries)
        let particles = vec![
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, 1),
            Particle::new(Vec3::new(0.45, 0.5, 0.5), 2.0, MAT_WATER, 1),
        ];
        let mut grid = make_grid();
        let dt = 0.001;

        p2g_sized(&particles, &mut grid, &table, dt, TEST_GS);

        let total_grid_mass: f32 = grid.iter().map(|c| c.velocity_mass.w).sum();
        let total_particle_mass: f32 = particles.iter().map(|p| p.mass()).sum();

        assert!(
            (total_grid_mass - total_particle_mass).abs() < 1e-4,
            "Mass not conserved: grid={total_grid_mass}, particles={total_particle_mass}"
        );
    }

    #[test]
    fn grid_update_applies_gravity() {
        let mut grid = make_grid();
        let dt = 0.01;

        // Put mass and zero momentum in a cell away from boundaries
        let idx = grid_index_sized(10, 10, 10, TEST_GS).unwrap();
        grid[idx].velocity_mass = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);

        grid_update_sized(&mut grid, dt, TEST_GS);

        let vy = grid[idx].velocity_mass.y;
        let expected = scaled_gravity(TEST_GS) * dt;
        assert!(
            (vy - expected).abs() < 1e-6,
            "Expected vy={expected}, got {vy}",
        );
    }

    #[test]
    fn grid_update_floor_boundary() {
        let mut grid = make_grid();
        let dt = 0.001;

        // Cell on the floor
        let idx = grid_index_sized(10, 1, 10, TEST_GS).unwrap();
        grid[idx].velocity_mass = glam::Vec4::new(0.0, -5.0, 0.0, 1.0); // downward momentum

        grid_update_sized(&mut grid, dt, TEST_GS);

        // Floor should clamp downward velocity
        let vy = grid[idx].velocity_mass.y;
        assert!(
            vy >= 0.0,
            "Floor boundary should prevent downward velocity, got {vy}"
        );
    }

    #[test]
    fn water_falls_downward() {
        let table = default_material_table();
        let initial_y = 0.6;
        let mut particles = vec![Particle::new(
            Vec3::new(0.5, initial_y, 0.5),
            1.0,
            MAT_WATER,
            1,
        )];

        let dt = 0.001;
        let steps = 50;

        for _ in 0..steps {
            let mut grid = make_grid();
            p2g_sized(&particles, &mut grid, &table, dt, TEST_GS);
            grid_update_sized(&mut grid, dt, TEST_GS);
            let occupancy = build_solid_occupancy_sized(&particles, TEST_GS);
            g2p_sized(&mut particles, &grid, &occupancy, dt, TEST_GS);
        }

        let final_y = particles[0].position().y;
        assert!(
            final_y < initial_y,
            "Water should fall: initial_y={initial_y}, final_y={final_y}"
        );
    }

    #[test]
    fn p2g_g2p_no_nan() {
        let table = default_material_table();
        let mut particles = vec![
            Particle::new(Vec3::new(0.3, 0.7, 0.5), 1.0, MAT_WATER, 1),
            Particle::new(Vec3::new(0.5, 0.7, 0.5), 1.0, MAT_WATER, 1),
            Particle::new(Vec3::new(0.7, 0.7, 0.5), 1.0, MAT_WATER, 1),
        ];

        let dt = 0.001;
        for _ in 0..20 {
            let mut grid = make_grid();
            p2g_sized(&particles, &mut grid, &table, dt, TEST_GS);
            grid_update_sized(&mut grid, dt, TEST_GS);
            let occupancy = build_solid_occupancy_sized(&particles, TEST_GS);
            g2p_sized(&mut particles, &grid, &occupancy, dt, TEST_GS);
        }

        for (i, p) in particles.iter().enumerate() {
            let pos = p.position();
            let vel = p.velocity();
            assert!(
                pos.is_finite(),
                "Particle {i} position is not finite: {pos:?}"
            );
            assert!(
                vel.is_finite(),
                "Particle {i} velocity is not finite: {vel:?}"
            );
        }
    }

    #[test]
    fn grid_index_bounds() {
        // Test with default GRID_SIZE
        let gs = shared::constants::GRID_SIZE as i32;
        assert!(grid_index(0, 0, 0).is_some());
        assert!(grid_index(gs - 1, gs - 1, gs - 1).is_some());
        assert!(grid_index(-1, 0, 0).is_none());
        assert!(grid_index(0, gs, 0).is_none());
        assert_eq!(grid_index(0, 0, 0).unwrap(), 0);

        // Test with sized variant
        assert!(grid_index_sized(0, 0, 0, TEST_GS).is_some());
        assert!(grid_index_sized(TEST_GS as i32 - 1, TEST_GS as i32 - 1, TEST_GS as i32 - 1, TEST_GS).is_some());
        assert!(grid_index_sized(TEST_GS as i32, 0, 0, TEST_GS).is_none());
    }

    #[test]
    fn thermal_diffusion_through_grid() {
        // Verify that temperature transfers between nearby particles
        // through the P2G -> grid_update -> G2P cycle.
        let table = default_material_table();
        let dx = 1.0 / TEST_GS as f32;
        let center = 0.5_f32;

        let mut particles = vec![
            // Hot water particle
            {
                let mut p = shared::particle::Particle::new(
                    glam::Vec3::new(center, center, center),
                    1.0,
                    shared::material::MAT_WATER,
                    1,
                );
                p.set_temperature(500.0);
                p
            },
            // Cold water particle 1 grid cell away
            {
                let mut p = shared::particle::Particle::new(
                    glam::Vec3::new(center + dx, center, center),
                    1.0,
                    shared::material::MAT_WATER,
                    1,
                );
                p.set_temperature(0.0);
                p
            },
        ];

        let initial_diff = (particles[0].temperature() - particles[1].temperature()).abs();
        let dt = 0.001;

        for _ in 0..20 {
            let mut grid = make_grid();
            p2g_sized(&particles, &mut grid, &table, dt, TEST_GS);
            grid_update_sized(&mut grid, dt, TEST_GS);
            let occupancy = build_solid_occupancy_sized(&particles, TEST_GS);
            g2p_sized(&mut particles, &grid, &occupancy, dt, TEST_GS);
        }

        let final_diff = (particles[0].temperature() - particles[1].temperature()).abs();

        assert!(
            final_diff < initial_diff,
            "Temperature should diffuse through grid: initial_diff={initial_diff}, final_diff={final_diff}"
        );
        // Both temperatures should be finite
        assert!(particles[0].temperature().is_finite());
        assert!(particles[1].temperature().is_finite());
    }
}
