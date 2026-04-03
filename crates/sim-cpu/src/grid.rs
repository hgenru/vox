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
/// Returns `None` if any index is out of bounds.
#[inline]
pub fn grid_index(ix: i32, iy: i32, iz: i32) -> Option<usize> {
    let gs = GRID_SIZE as i32;
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
        };
    }
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
pub fn p2g(particles: &[Particle], grid: &mut [GridCell], materials: &[MaterialParams], dt: f32) {
    let gs = GRID_SIZE as f32;
    let inv_dx = gs; // grid spacing = 1/GRID_SIZE in normalized coords
    let _ = inv_dx;

    for particle in particles {
        let pos = particle.position();
        let vel = particle.velocity();
        let mass = particle.mass();
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

                    let idx = match grid_index(ix, iy, iz) {
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
                }
            }
        }
    }
}

/// Grid update: convert momentum to velocity, apply gravity and boundary conditions.
///
/// For each cell with sufficient mass:
/// 1. Divide momentum by mass to get velocity
/// 2. Apply gravity: `vel.y += gravity * dt`
/// 3. Enforce boundary conditions at grid edges (velocity clamping)
///
/// # Arguments
/// - `grid`: mutable grid buffer (contains accumulated momentum from P2G)
/// - `dt`: timestep
pub fn grid_update(grid: &mut [GridCell], dt: f32) {
    let gs = GRID_SIZE as i32;
    let boundary = 3; // boundary cells thickness

    for iz in 0..gs {
        for iy in 0..gs {
            for ix in 0..gs {
                let idx = (iz * gs * gs + iy * gs + ix) as usize;
                let cell = &mut grid[idx];
                let mass = cell.velocity_mass.w;

                if mass < MASS_THRESHOLD {
                    cell.velocity_mass = glam::Vec4::ZERO;
                    continue;
                }

                // Convert momentum to velocity
                let inv_mass = 1.0 / mass;
                cell.velocity_mass.x *= inv_mass;
                cell.velocity_mass.y *= inv_mass;
                cell.velocity_mass.z *= inv_mass;

                // Apply gravity
                cell.velocity_mass.y += GRAVITY * dt;

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
            }
        }
    }
}

/// Grid-to-Particle transfer (G2P).
///
/// Gathers velocity from 27 neighbor grid cells to update each particle:
/// 1. PIC/FLIP blend for velocity (alpha=0.95 PIC, 0.05 FLIP)
/// 2. Update APIC C matrix (affine velocity field)
/// 3. Update deformation gradient: `F_new = (I + dt*C) * F`
/// 4. Advect position: `pos += vel * dt`
///
/// # Arguments
/// - `particles`: mutable slice of particles
/// - `grid`: grid buffer with updated velocities
/// - `dt`: timestep
pub fn g2p(particles: &mut [Particle], grid: &[GridCell], dt: f32) {
    let gs = GRID_SIZE as f32;
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

        // Gather from 3x3x3 neighbors
        for dz in 0..3_i32 {
            for dy in 0..3_i32 {
                for dx_i in 0..3_i32 {
                    let ix = base_x + dx_i;
                    let iy = base_y + dy;
                    let iz = base_z + dz;

                    let idx = match grid_index(ix, iy, iz) {
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
                }
            }
        }

        // PIC/FLIP blend
        // True FLIP requires old grid velocities which we don't store.
        // Instead, blend PIC (grid velocity) with old particle velocity
        // to reduce excessive damping while maintaining stability.
        let blended_vel = pic_blend * new_vel + (1.0 - pic_blend) * old_vel;

        let mut final_vel = blended_vel;

        // Gas buoyancy: counteract most of gravity and add slight upward force
        // so steam rises and persists visually instead of falling immediately.
        if particle.phase() == 2 {
            final_vel.y += (-GRAVITY) * 0.7 * dt;
            final_vel.x *= 0.98;
            final_vel.z *= 0.98;
        }

        // Update velocity
        particle.set_velocity(final_vel);

        // Update APIC C matrix
        particle.set_affine_momentum(new_c);

        // Update deformation gradient: F_new = (I + dt * C) * F
        let f_old = particle.deformation_gradient();
        let f_new = (Mat3::IDENTITY + dt * new_c) * f_old;
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
    use shared::{
        constants::GRID_CELL_COUNT,
        material::{MAT_WATER, default_material_table},
    };

    use super::*;

    fn make_grid() -> Vec<GridCell> {
        vec![
            GridCell {
                velocity_mass: glam::Vec4::ZERO,
                force_pad: glam::Vec4::ZERO,
            };
            GRID_CELL_COUNT as usize
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

        p2g(&particles, &mut grid, &table, dt);

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
        let idx = grid_index(16, 16, 16).unwrap();
        grid[idx].velocity_mass = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);

        grid_update(&mut grid, dt);

        let vy = grid[idx].velocity_mass.y;
        assert!(
            (vy - GRAVITY * dt).abs() < 1e-6,
            "Expected vy={}, got {vy}",
            GRAVITY * dt
        );
    }

    #[test]
    fn grid_update_floor_boundary() {
        let mut grid = make_grid();
        let dt = 0.001;

        // Cell on the floor
        let idx = grid_index(16, 1, 16).unwrap();
        grid[idx].velocity_mass = glam::Vec4::new(0.0, -5.0, 0.0, 1.0); // downward momentum

        grid_update(&mut grid, dt);

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
            p2g(&particles, &mut grid, &table, dt);
            grid_update(&mut grid, dt);
            g2p(&mut particles, &grid, dt);
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
            p2g(&particles, &mut grid, &table, dt);
            grid_update(&mut grid, dt);
            g2p(&mut particles, &grid, dt);
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
        assert!(grid_index(0, 0, 0).is_some());
        assert!(grid_index(31, 31, 31).is_some());
        assert!(grid_index(-1, 0, 0).is_none());
        assert!(grid_index(0, 32, 0).is_none());
        assert_eq!(grid_index(0, 0, 0).unwrap(), 0);
    }
}
