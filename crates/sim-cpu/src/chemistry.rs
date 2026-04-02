//! Chemical reaction checking and application.
//!
//! Detects particle contacts via grid overlap and applies reaction rules.
//! Water + Lava -> Stone + Steam is the primary reaction.

use shared::{
    constants::GRID_SIZE,
    particle::Particle,
    reaction::{default_reaction_table, find_reaction, ReactionRule},
};

use crate::grid::grid_index;

/// Check and apply chemical reactions between particles that share grid cells.
///
/// Algorithm:
/// 1. Build a grid mapping from cell index to list of particle indices
/// 2. For each cell, check all pairs of particles for matching reactions
/// 3. Apply reactions: transform materials, reset F, set temperatures
///
/// Each particle can only react once per call to prevent chain reactions
/// in a single step.
///
/// # Arguments
/// - `particles`: mutable slice of particles
pub fn check_reactions(particles: &mut [Particle]) {
    let rules = default_reaction_table();
    if rules.is_empty() || particles.is_empty() {
        return;
    }

    let gs = GRID_SIZE as f32;
    let grid_count = (GRID_SIZE * GRID_SIZE * GRID_SIZE) as usize;

    // Build grid -> particle index mapping
    // For each grid cell, store the indices of particles that have weight > 0 there
    let mut cell_particles: Vec<Vec<usize>> = vec![Vec::new(); grid_count];

    for (pi, particle) in particles.iter().enumerate() {
        let pos = particle.position();
        let grid_pos = pos * gs;
        let base_x = (grid_pos.x - 0.5).floor() as i32;
        let base_y = (grid_pos.y - 0.5).floor() as i32;
        let base_z = (grid_pos.z - 0.5).floor() as i32;

        // Only add to the nearest cell (center of stencil) for efficiency
        let cx = (grid_pos.x).round() as i32;
        let cy = (grid_pos.y).round() as i32;
        let cz = (grid_pos.z).round() as i32;
        let _ = (base_x, base_y, base_z); // suppress unused

        if let Some(idx) = grid_index(cx, cy, cz) {
            cell_particles[idx].push(pi);
        }
    }

    // Track which particles have already reacted
    let mut reacted = vec![false; particles.len()];

    // Check each cell for reaction pairs
    for cell_parts in &cell_particles {
        if cell_parts.len() < 2 {
            continue;
        }

        for i in 0..cell_parts.len() {
            for j in (i + 1)..cell_parts.len() {
                let pi_a = cell_parts[i];
                let pi_b = cell_parts[j];

                if reacted[pi_a] || reacted[pi_b] {
                    continue;
                }

                let mat_a = particles[pi_a].material_id();
                let mat_b = particles[pi_b].material_id();

                if let Some((rule, swapped)) = find_reaction(mat_a, mat_b, rules) {
                    // Check temperature requirement
                    let temp_a = particles[pi_a].temperature();
                    let temp_b = particles[pi_b].temperature();
                    let min_temp = temp_a.min(temp_b);

                    if min_temp < rule.min_temperature {
                        continue;
                    }

                    // Apply reaction
                    if swapped {
                        apply_reaction_to_pair(particles, pi_b, pi_a, rule);
                    } else {
                        apply_reaction_to_pair(particles, pi_a, pi_b, rule);
                    }

                    reacted[pi_a] = true;
                    reacted[pi_b] = true;
                }
            }
        }
    }
}

/// Apply a reaction rule to a specific pair of particles.
///
/// `pa_idx` corresponds to reactant_a, `pb_idx` to reactant_b.
fn apply_reaction_to_pair(
    particles: &mut [Particle],
    pa_idx: usize,
    pb_idx: usize,
    rule: &ReactionRule,
) {
    // Transform particle A
    particles[pa_idx].ids.x = rule.product_a_material;
    particles[pa_idx].set_phase(rule.product_a_phase);
    particles[pa_idx].reset_deformation_gradient();
    particles[pa_idx].set_damage(0.0);
    if !rule.product_a_temp.is_nan() {
        particles[pa_idx].set_temperature(rule.product_a_temp);
    }

    // Transform particle B
    particles[pb_idx].ids.x = rule.product_b_material;
    particles[pb_idx].set_phase(rule.product_b_phase);
    particles[pb_idx].reset_deformation_gradient();
    particles[pb_idx].set_damage(0.0);
    if !rule.product_b_temp.is_nan() {
        particles[pb_idx].set_temperature(rule.product_b_temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Vec3};
    use shared::material::{
        MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_GAS, PHASE_LIQUID, PHASE_SOLID,
    };

    #[test]
    fn water_lava_reaction_produces_stone_and_steam() {
        let mut particles = vec![
            // Water particle
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID);
                p.set_temperature(50.0);
                p
            },
            // Lava particle at same position
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_LAVA, PHASE_LIQUID);
                p.set_temperature(1600.0);
                p
            },
        ];

        check_reactions(&mut particles);

        // Water should become stone
        assert_eq!(particles[0].material_id(), MAT_STONE);
        assert_eq!(particles[0].phase(), PHASE_SOLID);

        // Lava should become steam (water gas)
        assert_eq!(particles[1].material_id(), MAT_WATER);
        assert_eq!(particles[1].phase(), PHASE_GAS);
    }

    #[test]
    fn reaction_resets_deformation_gradient() {
        let mut particles = vec![
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID);
                p.set_deformation_gradient(Mat3::from_diagonal(Vec3::new(1.5, 0.8, 1.2)));
                p
            },
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_LAVA, PHASE_LIQUID);
                p.set_deformation_gradient(Mat3::from_diagonal(Vec3::new(0.9, 1.1, 0.95)));
                p
            },
        ];

        check_reactions(&mut particles);

        assert_eq!(particles[0].deformation_gradient(), Mat3::IDENTITY);
        assert_eq!(particles[1].deformation_gradient(), Mat3::IDENTITY);
    }

    #[test]
    fn no_reaction_same_material() {
        let mut particles = vec![
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID),
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID),
        ];

        check_reactions(&mut particles);

        // Both should remain water
        assert_eq!(particles[0].material_id(), MAT_WATER);
        assert_eq!(particles[1].material_id(), MAT_WATER);
    }

    #[test]
    fn reaction_mass_conserved() {
        let mut particles = vec![
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 2.0, MAT_WATER, PHASE_LIQUID);
                p.set_temperature(50.0);
                p
            },
            {
                let mut p = Particle::new(Vec3::new(0.5, 0.5, 0.5), 3.0, MAT_LAVA, PHASE_LIQUID);
                p.set_temperature(1600.0);
                p
            },
        ];

        let total_mass_before: f32 = particles.iter().map(|p| p.mass()).sum();
        check_reactions(&mut particles);
        let total_mass_after: f32 = particles.iter().map(|p| p.mass()).sum();

        assert!(
            (total_mass_before - total_mass_after).abs() < 1e-8,
            "Mass not conserved: before={total_mass_before}, after={total_mass_after}"
        );
    }

    #[test]
    fn particles_far_apart_dont_react() {
        let mut particles = vec![
            Particle::new(Vec3::new(0.2, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID),
            Particle::new(Vec3::new(0.8, 0.5, 0.5), 1.0, MAT_LAVA, PHASE_LIQUID),
        ];

        check_reactions(&mut particles);

        // Should NOT react (different grid cells)
        assert_eq!(particles[0].material_id(), MAT_WATER);
        assert_eq!(particles[1].material_id(), MAT_LAVA);
    }

    #[test]
    fn each_particle_reacts_only_once() {
        // Three particles in the same cell: 2 water + 1 lava
        let mut particles = vec![
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID),
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_LAVA, PHASE_LIQUID),
            Particle::new(Vec3::new(0.5, 0.5, 0.5), 1.0, MAT_WATER, PHASE_LIQUID),
        ];

        check_reactions(&mut particles);

        // Only one water+lava pair should react
        let water_count = particles
            .iter()
            .filter(|p| p.material_id() == MAT_WATER && p.phase() == PHASE_LIQUID)
            .count();
        // One water should remain unreacted
        assert_eq!(
            water_count, 1,
            "One water should remain unreacted"
        );
    }
}
