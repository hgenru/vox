//! Phase transition rules for MPM particles.
//!
//! Uses data-driven phase transition table from `reaction::PhaseTransitionRule`.
//! No more hardcoded match blocks — adding a new material only requires
//! adding a rule to `default_phase_transition_table()`.
//!
//! On transition: reset F = Identity, damage = 0, update phase.

use crate::{
    particle::Particle,
    reaction::PhaseTransitionRule,
};

/// Result of checking a phase transition for a particle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseTransition {
    /// No transition occurs.
    None,
    /// Transition to a new phase.
    Transition {
        /// New material ID after transition.
        new_material_id: u32,
        /// New phase after transition.
        new_phase: u32,
        /// Behavior flags (bit 0 = reset_F, bit 1 = explosion).
        flags: u32,
    },
}

/// Check if a particle should undergo a phase transition based on temperature.
///
/// Iterates over the provided reaction rules and returns the first matching
/// transition. Rules are matched by (material_id, phase) and temperature range.
///
/// # Arguments
/// - `particle`: the particle to check
/// - `rules`: phase transition rule table (slice of valid entries)
pub fn check_phase_transition(
    particle: &Particle,
    rules: &[PhaseTransitionRule],
) -> PhaseTransition {
    let mat_id = particle.material_id();
    let phase = particle.phase();
    let temp = particle.temperature();

    for rule in rules {
        if rule.materials.x == mat_id && rule.materials.y == phase {
            let temp_min = rule.conditions.x;
            let temp_max = rule.conditions.y;
            if temp > temp_min && temp < temp_max {
                return PhaseTransition::Transition {
                    new_material_id: rule.materials.z,
                    new_phase: rule.materials.w,
                    flags: rule.conditions.z as u32,
                };
            }
        }
    }
    PhaseTransition::None
}

/// Apply a phase transition to a particle.
///
/// Resets deformation gradient to identity and damage to 0 if FLAG_RESET_F is set.
/// See CLAUDE.md trap #8: F must be reset on phase transition.
pub fn apply_phase_transition(particle: &mut Particle, transition: PhaseTransition) {
    if let PhaseTransition::Transition {
        new_material_id,
        new_phase,
        flags,
    } = transition
    {
        particle.ids.x = new_material_id;
        particle.set_phase(new_phase);
        if flags & crate::reaction::FLAG_RESET_F != 0 {
            particle.reset_deformation_gradient();
        }
        particle.set_damage(0.0);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::material::{
        MAT_GUNPOWDER, MAT_ICE, MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_GAS, PHASE_LIQUID,
        PHASE_SOLID,
    };
    use crate::reaction::{default_phase_transition_table, FLAG_EXPLOSION, FLAG_RESET_F};

    /// Helper: get the valid rules slice from the default table.
    fn rules() -> &'static [PhaseTransitionRule] {
        static RULES: std::sync::LazyLock<([PhaseTransitionRule; 16], usize)> =
            std::sync::LazyLock::new(default_phase_transition_table);
        &RULES.0[..RULES.1]
    }

    #[test]
    fn stone_melts_to_lava() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1600.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_LAVA,
                new_phase: PHASE_LIQUID,
                flags: FLAG_RESET_F,
            }
        );
    }

    #[test]
    fn stone_stays_solid_below_melting() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1000.0);
        assert_eq!(check_phase_transition(&p, rules()), PhaseTransition::None);
    }

    #[test]
    fn water_boils_to_steam() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_LIQUID);
        p.set_temperature(101.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_WATER,
                new_phase: PHASE_GAS,
                flags: FLAG_RESET_F,
            }
        );
    }

    #[test]
    fn water_freezes_to_ice() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_LIQUID);
        p.set_temperature(-5.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_ICE,
                new_phase: PHASE_SOLID,
                flags: FLAG_RESET_F,
            }
        );
    }

    #[test]
    fn ice_melts_to_water() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_ICE, PHASE_SOLID);
        p.set_temperature(5.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_WATER,
                new_phase: PHASE_LIQUID,
                flags: FLAG_RESET_F,
            }
        );
    }

    #[test]
    fn lava_solidifies_to_stone() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_LAVA, PHASE_LIQUID);
        p.set_temperature(1400.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_STONE,
                new_phase: PHASE_SOLID,
                flags: FLAG_RESET_F,
            }
        );
    }

    #[test]
    fn gunpowder_explodes_to_gas() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_GUNPOWDER, PHASE_SOLID);
        p.set_temperature(250.0);
        let tr = check_phase_transition(&p, rules());
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_GUNPOWDER,
                new_phase: PHASE_GAS,
                flags: FLAG_RESET_F | FLAG_EXPLOSION,
            }
        );
    }

    #[test]
    fn gunpowder_stays_solid_below_ignition() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_GUNPOWDER, PHASE_SOLID);
        p.set_temperature(100.0);
        assert_eq!(check_phase_transition(&p, rules()), PhaseTransition::None);
    }

    #[test]
    fn apply_transition_resets_f_and_damage() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1600.0);
        use glam::Mat3;
        p.set_deformation_gradient(Mat3::from_diagonal(Vec3::new(1.5, 0.8, 1.2)));
        p.set_damage(0.7);

        let tr = check_phase_transition(&p, rules());
        apply_phase_transition(&mut p, tr);

        assert_eq!(p.material_id(), MAT_LAVA);
        assert_eq!(p.phase(), PHASE_LIQUID);
        assert_eq!(p.deformation_gradient(), Mat3::IDENTITY);
        assert!((p.damage() - 0.0).abs() < 1e-8);
    }
}
