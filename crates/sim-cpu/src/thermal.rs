//! Temperature diffusion and phase transition processing.
//!
//! Heat diffusion via grid: `T += conductivity * (T_avg_neighbors - T) * dt`
//! Phase transitions are checked after temperature update.

use glam::Vec3;
use shared::{
    constants::GRID_SIZE,
    material::MaterialParams,
    particle::Particle,
    phase::{apply_phase_transition, check_phase_transition},
};

use crate::grid::{bspline_weight, grid_index};

/// Diffuse temperature on the grid using a simple averaging scheme.
///
/// For each particle, gather temperature from neighbors and apply diffusion:
/// `T_new = T + conductivity * (T_avg_neighbors - T) * dt`
///
/// This is a simplified particle-level diffusion using the grid as a medium.
///
/// # Arguments
/// - `particles`: mutable particle slice
/// - `materials`: material parameter table
/// - `dt`: timestep
pub fn diffuse_temperature(particles: &mut [Particle], materials: &[MaterialParams], dt: f32) {
    let gs = GRID_SIZE as f32;

    // First, accumulate temperature onto grid (weighted average)
    let grid_count = (GRID_SIZE * GRID_SIZE * GRID_SIZE) as usize;
    let mut grid_temp = vec![0.0_f32; grid_count];
    let mut grid_mass = vec![0.0_f32; grid_count];

    for particle in particles.iter() {
        let pos = particle.position();
        let temp = particle.temperature();
        let mass = particle.mass();
        let grid_pos = pos * gs;
        let base_x = (grid_pos.x - 0.5).floor() as i32;
        let base_y = (grid_pos.y - 0.5).floor() as i32;
        let base_z = (grid_pos.z - 0.5).floor() as i32;

        for dz in 0..3_i32 {
            for dy in 0..3_i32 {
                for dx in 0..3_i32 {
                    let ix = base_x + dx;
                    let iy = base_y + dy;
                    let iz = base_z + dz;
                    let idx = match grid_index(ix, iy, iz) {
                        Some(i) => i,
                        None => continue,
                    };
                    let dist = grid_pos - Vec3::new(ix as f32, iy as f32, iz as f32);
                    let w =
                        bspline_weight(dist.x) * bspline_weight(dist.y) * bspline_weight(dist.z);

                    grid_temp[idx] += w * mass * temp;
                    grid_mass[idx] += w * mass;
                }
            }
        }
    }

    // Convert to average temperature
    for i in 0..grid_count {
        if grid_mass[i] > 1e-10 {
            grid_temp[i] /= grid_mass[i];
        }
    }

    // Gather diffused temperature back to particles
    for particle in particles.iter_mut() {
        let mat_id = particle.material_id() as usize;
        let conductivity = if mat_id < materials.len() {
            materials[mat_id].thermal.z
        } else {
            0.0
        };

        if conductivity < 1e-10 {
            continue;
        }

        let pos = particle.position();
        let grid_pos = pos * gs;
        let base_x = (grid_pos.x - 0.5).floor() as i32;
        let base_y = (grid_pos.y - 0.5).floor() as i32;
        let base_z = (grid_pos.z - 0.5).floor() as i32;

        let mut t_avg = 0.0_f32;
        let mut w_total = 0.0_f32;

        for dz in 0..3_i32 {
            for dy in 0..3_i32 {
                for dx in 0..3_i32 {
                    let ix = base_x + dx;
                    let iy = base_y + dy;
                    let iz = base_z + dz;
                    let idx = match grid_index(ix, iy, iz) {
                        Some(i) => i,
                        None => continue,
                    };
                    let dist = grid_pos - Vec3::new(ix as f32, iy as f32, iz as f32);
                    let w =
                        bspline_weight(dist.x) * bspline_weight(dist.y) * bspline_weight(dist.z);

                    t_avg += w * grid_temp[idx];
                    w_total += w;
                }
            }
        }

        if w_total > 1e-10 {
            t_avg /= w_total;
        }

        let current_temp = particle.temperature();
        let new_temp = current_temp + conductivity * (t_avg - current_temp) * dt;
        particle.set_temperature(new_temp);
    }
}

/// Check and apply phase transitions for all particles.
///
/// After temperature update, each particle is checked against phase
/// transition rules. On transition: F = Identity, damage = 0.
///
/// Gunpowder explosion additionally sets outward velocity and boosts
/// temperature to 3000C, matching the shader implementation in g2p.rs.
///
/// # Arguments
/// - `particles`: mutable particle slice
pub fn apply_phase_transitions(particles: &mut [Particle]) {
    for particle in particles.iter_mut() {
        let was_gunpowder_solid = particle.material_id() == shared::material::MAT_GUNPOWDER
            && particle.phase() == shared::material::PHASE_SOLID;
        let transition = check_phase_transition(particle);
        apply_phase_transition(particle, transition);

        // Gunpowder explosion: apply outward velocity and boost temperature.
        // Must match shader g2p.rs apply_gunpowder_explosion().
        if was_gunpowder_solid && particle.phase() == shared::material::PHASE_GAS {
            apply_gunpowder_explosion_cpu(particle);
        }
    }
}

/// Apply gunpowder explosion velocity and temperature boost (CPU side).
///
/// Uses particle position as a seed for pseudo-random direction.
/// Matches the shader implementation in g2p.rs `apply_gunpowder_explosion()`.
fn apply_gunpowder_explosion_cpu(particle: &mut Particle) {
    let pos = particle.position();
    let px = pos.x;
    let py = pos.y;
    let pz = pos.z;

    // Hash each axis with different seeds (matches shader hash_f32 logic)
    let hx = hash_f32_cpu(px * 73.17 + pz * 37.91);
    let hy = hash_f32_cpu(py * 127.31 + px * 53.47);
    let hz = hash_f32_cpu(pz * 91.53 + py * 67.13);

    let explosion_speed = 150.0_f32;
    let gs = shared::constants::GRID_SIZE as f32;

    // Convert grid-space velocity to world-space (divide by grid size)
    let vel = Vec3::new(
        hx * explosion_speed / gs,
        hy.abs() * explosion_speed / gs + 80.0 / gs,
        hz * explosion_speed / gs,
    );
    particle.set_velocity(vel);
    particle.set_temperature(3000.0);
}

/// CPU-side pseudo-random hash matching shader hash_f32.
///
/// Reinterprets float bits and applies xorshift mixing.
fn hash_f32_cpu(x: f32) -> f32 {
    let mut bits = x.to_bits();
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x45d9f3b);
    bits ^= bits >> 16;
    (bits & 0xFFFF) as f32 / 32768.0 - 1.0
}

#[cfg(test)]
mod tests {
    use shared::material::{
        MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, default_material_table,
    };

    use super::*;

    #[test]
    fn temperature_diffusion_equalizes() {
        let table = default_material_table();
        // Place particles close enough to share grid cells.
        // GRID_SIZE=256, so 1 grid cell = 1/256 ~ 0.0039.
        // B-spline stencil covers +-1.5 cells, so particles within ~3 cells
        // (~0.012 world units) will overlap and exchange heat.
        let dx = 1.0 / shared::constants::GRID_SIZE as f32; // one grid cell
        let mut particles = vec![
            // Hot particle
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_STONE, PHASE_SOLID);
                p.set_temperature(1000.0);
                p
            },
            // Cold particle 1 grid cell away (within B-spline overlap)
            {
                let mut p =
                    Particle::new(Vec3::new(0.5 + dx, 0.5, 0.5), 1.0, MAT_STONE, PHASE_SOLID);
                p.set_temperature(100.0);
                p
            },
        ];

        let initial_diff = (particles[0].temperature() - particles[1].temperature()).abs();

        for _ in 0..100 {
            diffuse_temperature(&mut particles, &table, 0.01);
        }

        let final_diff = (particles[0].temperature() - particles[1].temperature()).abs();

        assert!(
            final_diff < initial_diff,
            "Temperature should equalize: initial_diff={initial_diff}, final_diff={final_diff}"
        );
    }

    #[test]
    fn phase_transition_stone_to_lava() {
        let mut particles = vec![{
            let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
            p.set_temperature(1600.0);
            p
        }];

        apply_phase_transitions(&mut particles);

        assert_eq!(particles[0].material_id(), MAT_LAVA);
        assert_eq!(particles[0].phase(), PHASE_LIQUID);
    }

    #[test]
    fn phase_transition_water_to_steam() {
        let mut particles = vec![{
            let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_LIQUID);
            p.set_temperature(110.0);
            p
        }];

        apply_phase_transitions(&mut particles);

        assert_eq!(particles[0].material_id(), MAT_WATER);
        assert_eq!(particles[0].phase(), shared::material::PHASE_GAS);
    }

    #[test]
    fn lava_cools_to_stone() {
        let table = default_material_table();
        let dx = 1.0 / shared::constants::GRID_SIZE as f32;
        let mut particles = vec![
            // Hot lava
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_LAVA, PHASE_LIQUID);
                p.set_temperature(1600.0);
                p
            },
            // Cold stone 1 grid cell away (acts as heat sink)
            {
                let mut p = Particle::new(
                    Vec3::new(0.5 + dx, 0.5, 0.5),
                    10.0,
                    MAT_STONE,
                    PHASE_SOLID,
                );
                p.set_temperature(20.0);
                p
            },
        ];

        // Run many diffusion + transition steps
        for _ in 0..500 {
            diffuse_temperature(&mut particles, &table, 0.01);
            apply_phase_transitions(&mut particles);
        }

        // The lava should have cooled and solidified
        assert_eq!(
            particles[0].material_id(),
            MAT_STONE,
            "Lava should have solidified to stone"
        );
        assert_eq!(particles[0].phase(), PHASE_SOLID);
    }
}
