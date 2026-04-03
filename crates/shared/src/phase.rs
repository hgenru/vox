//! Phase transition rules for MPM particles.
//!
//! Defines temperature-based phase transitions:
//! - Stone T > 1500 -> Lava (liquid)
//! - Water T > 100 -> Steam (gas)
//! - Water T < 0 -> Ice (solid)
//! - Lava T < 1500 -> Stone (solid)
//! - Gunpowder T > 200 -> Gas (explosion; GPU shader also boosts T to 3000)
//!
//! On transition: reset F = Identity, damage = 0, update phase.

use crate::{
    material::{MAT_GUNPOWDER, MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_GAS, PHASE_LIQUID, PHASE_SOLID},
    particle::Particle,
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
    },
}

/// Check if a particle should undergo a phase transition based on temperature.
///
/// Returns the transition result (either no change or new material/phase).
///
/// # Rules
/// - Stone (material=0, phase=solid) + T > 1500 -> Lava (material=2, phase=liquid)
/// - Water (material=1, phase=liquid) + T > 100 -> Steam (material=1, phase=gas)
/// - Water (material=1, phase=liquid) + T < 0 -> Ice (material=1, phase=solid)
/// - Lava (material=2, phase=liquid) + T < 1500 -> Stone (material=0, phase=solid)
/// - Ice (material=1, phase=solid) + T > 0 -> Water (material=1, phase=liquid)
pub fn check_phase_transition(particle: &Particle) -> PhaseTransition {
    let mat_id = particle.material_id();
    let phase = particle.phase();
    let temp = particle.temperature();

    match (mat_id, phase) {
        // Stone solid -> Lava liquid when T > 1500
        (MAT_STONE, PHASE_SOLID) if temp > 1500.0 => PhaseTransition::Transition {
            new_material_id: MAT_LAVA,
            new_phase: PHASE_LIQUID,
        },
        // Water liquid -> Steam gas when T > 100
        (MAT_WATER, PHASE_LIQUID) if temp > 100.0 => PhaseTransition::Transition {
            new_material_id: MAT_WATER,
            new_phase: PHASE_GAS,
        },
        // Water liquid -> Ice solid when T < 0
        (MAT_WATER, PHASE_LIQUID) if temp < 0.0 => PhaseTransition::Transition {
            new_material_id: MAT_WATER,
            new_phase: PHASE_SOLID,
        },
        // Ice solid -> Water liquid when T > 0
        (MAT_WATER, PHASE_SOLID) if temp > 0.0 => PhaseTransition::Transition {
            new_material_id: MAT_WATER,
            new_phase: PHASE_LIQUID,
        },
        // Lava liquid -> Stone solid when T < 1500
        (MAT_LAVA, PHASE_LIQUID) if temp < 1500.0 => PhaseTransition::Transition {
            new_material_id: MAT_STONE,
            new_phase: PHASE_SOLID,
        },
        // Gunpowder solid -> Gas when T > 200 (explosion)
        (MAT_GUNPOWDER, PHASE_SOLID) if temp > 200.0 => PhaseTransition::Transition {
            new_material_id: MAT_GUNPOWDER,
            new_phase: PHASE_GAS,
        },
        _ => PhaseTransition::None,
    }
}

/// Apply a phase transition to a particle.
///
/// Resets deformation gradient to identity and damage to 0.
/// See CLAUDE.md trap #8: F must be reset on phase transition.
pub fn apply_phase_transition(particle: &mut Particle, transition: PhaseTransition) {
    if let PhaseTransition::Transition {
        new_material_id,
        new_phase,
    } = transition
    {
        particle.ids.x = new_material_id;
        particle.set_phase(new_phase);
        particle.reset_deformation_gradient();
        particle.set_damage(0.0);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn stone_melts_to_lava() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1600.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_LAVA,
                new_phase: PHASE_LIQUID,
            }
        );
    }

    #[test]
    fn stone_stays_solid_below_melting() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1000.0);
        assert_eq!(check_phase_transition(&p), PhaseTransition::None);
    }

    #[test]
    fn water_boils_to_steam() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_LIQUID);
        p.set_temperature(101.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_WATER,
                new_phase: PHASE_GAS,
            }
        );
    }

    #[test]
    fn water_freezes_to_ice() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_LIQUID);
        p.set_temperature(-5.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_WATER,
                new_phase: PHASE_SOLID,
            }
        );
    }

    #[test]
    fn ice_melts_to_water() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, PHASE_SOLID);
        p.set_temperature(5.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_WATER,
                new_phase: PHASE_LIQUID,
            }
        );
    }

    #[test]
    fn lava_solidifies_to_stone() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_LAVA, PHASE_LIQUID);
        p.set_temperature(1400.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_STONE,
                new_phase: PHASE_SOLID,
            }
        );
    }

    #[test]
    fn gunpowder_explodes_to_gas() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_GUNPOWDER, PHASE_SOLID);
        p.set_temperature(250.0);
        let tr = check_phase_transition(&p);
        assert_eq!(
            tr,
            PhaseTransition::Transition {
                new_material_id: MAT_GUNPOWDER,
                new_phase: PHASE_GAS,
            }
        );
    }

    #[test]
    fn gunpowder_stays_solid_below_ignition() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_GUNPOWDER, PHASE_SOLID);
        p.set_temperature(100.0);
        assert_eq!(check_phase_transition(&p), PhaseTransition::None);
    }

    #[test]
    fn apply_transition_resets_f_and_damage() {
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, PHASE_SOLID);
        p.set_temperature(1600.0);
        // Give it some deformation and damage
        use glam::Mat3;
        p.set_deformation_gradient(Mat3::from_diagonal(Vec3::new(1.5, 0.8, 1.2)));
        p.set_damage(0.7);

        let tr = check_phase_transition(&p);
        apply_phase_transition(&mut p, tr);

        assert_eq!(p.material_id(), MAT_LAVA);
        assert_eq!(p.phase(), PHASE_LIQUID);
        assert_eq!(p.deformation_gradient(), Mat3::IDENTITY);
        assert!((p.damage() - 0.0).abs() < 1e-8);
    }
}
