//! Data-driven phase transition and chemical reaction rules.
//!
//! Phase transitions (temperature-based) and chemical reactions (contact-based)
//! are encoded as GPU-friendly `PhaseTransitionRule` and `ReactionRule` structs.
//! This replaces hardcoded match blocks and enables adding new materials
//! without editing code in multiple places.

use bytemuck::{Pod, Zeroable};
use glam::{UVec4, Vec4};

use crate::material::{
    MAT_GUNPOWDER, MAT_ICE, MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_GAS, PHASE_LIQUID, PHASE_SOLID,
};

/// Maximum number of phase transition rules in the table.
pub const MAX_PHASE_RULES: usize = 16;

/// Maximum number of slow (progress-based) reaction rules.
pub const MAX_SLOW_RULES: usize = 16;

/// How fast a reaction progresses. Used for tick budget scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReactionSpeed {
    /// Instant reaction, checked every tick (water+lava -> stone+steam).
    Instant = 0,
    /// Slow reaction, accumulates progress over ticks (stone -> diamond).
    Slow = 1,
    /// Chunk-level reaction using aggregate state (background processes).
    ChunkLevel = 2,
}

// SAFETY: ReactionSpeed is #[repr(u32)] with all bit patterns 0..=2 valid,
// and zero (Instant) is a valid variant.
unsafe impl Zeroable for ReactionSpeed {}
unsafe impl Pod for ReactionSpeed {}

/// A reaction that accumulates progress over time.
/// When progress reaches `ticks_needed`, the reaction completes.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SlowReactionRule {
    /// Source material ID.
    pub from_material: u32,
    /// Target material ID.
    pub to_material: u32,
    /// Temperature threshold (must be above this).
    pub min_temperature: f32,
    /// Number of ticks needed for completion.
    pub ticks_needed: u32,
}

/// Build the default slow reaction table.
///
/// Returns a fixed-size array and the count of valid entries.
/// Entries beyond `count` are zeroed. Currently empty -- slow reactions
/// will be populated when the content/materials system is expanded.
pub fn default_slow_reaction_table() -> ([SlowReactionRule; MAX_SLOW_RULES], usize) {
    let table = [SlowReactionRule::zeroed(); MAX_SLOW_RULES];
    (table, 0)
}

/// Flags for phase transition behavior.
///
/// - Bit 0 (1): Reset deformation gradient F to identity.
/// - Bit 1 (2): Apply gunpowder-style explosion (outward velocity + temp boost).
pub const FLAG_RESET_F: u32 = 1;
/// Flag to apply explosion behavior on transition.
pub const FLAG_EXPLOSION: u32 = 2;

/// A single phase transition rule. 32 bytes, 16-byte aligned.
///
/// GPU-friendly struct that encodes: "if a particle has material X in phase Y,
/// and its temperature satisfies `temp_min < T < temp_max`, transition to
/// material Z in phase W."
///
/// - `materials`: x=from_material, y=from_phase, z=to_material, w=to_phase
/// - `conditions`: x=temp_min, y=temp_max, z=flags (as f32), w=pad
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct PhaseTransitionRule {
    /// x = from_material_id, y = from_phase, z = to_material_id, w = to_phase.
    pub materials: UVec4,
    /// x = temp_min (trigger if T > this), y = temp_max (trigger if T < this),
    /// z = flags (bit 0 = reset_F, bit 1 = explosion), w = pad.
    pub conditions: Vec4,
}

/// Build the default phase transition table.
///
/// Returns a fixed-size array and the count of valid entries.
/// Entries beyond `count` are zeroed.
///
/// # Rules
/// - Stone (solid) + T > 1500 -> Lava (liquid)
/// - Water (liquid) + T > 100 -> Steam (gas)
/// - Water (liquid) + T < 0 -> Ice (solid)
/// - Ice (solid) + T > 0 -> Water (liquid)
/// - Lava (liquid) + T < 1500 -> Stone (solid)
/// - Gunpowder (solid) + T > 150 -> Gas (explosion)
pub fn default_phase_transition_table() -> ([PhaseTransitionRule; MAX_PHASE_RULES], usize) {
    let mut table = [PhaseTransitionRule::zeroed(); MAX_PHASE_RULES];

    // Stone (solid) + T > 1500 -> Lava (liquid)
    table[0] = PhaseTransitionRule {
        materials: UVec4::new(MAT_STONE, PHASE_SOLID, MAT_LAVA, PHASE_LIQUID),
        conditions: Vec4::new(1500.0, f32::MAX, FLAG_RESET_F as f32, 0.0),
    };
    // Water (liquid) + T > 100 -> Steam (gas)
    table[1] = PhaseTransitionRule {
        materials: UVec4::new(MAT_WATER, PHASE_LIQUID, MAT_WATER, PHASE_GAS),
        conditions: Vec4::new(100.0, f32::MAX, FLAG_RESET_F as f32, 0.0),
    };
    // Water (liquid) + T < 0 -> Ice (solid)
    table[2] = PhaseTransitionRule {
        materials: UVec4::new(MAT_WATER, PHASE_LIQUID, MAT_ICE, PHASE_SOLID),
        conditions: Vec4::new(f32::MIN, 0.0, FLAG_RESET_F as f32, 0.0),
    };
    // Ice (solid) + T > 0 -> Water (liquid)
    table[3] = PhaseTransitionRule {
        materials: UVec4::new(MAT_ICE, PHASE_SOLID, MAT_WATER, PHASE_LIQUID),
        conditions: Vec4::new(0.0, f32::MAX, FLAG_RESET_F as f32, 0.0),
    };
    // Lava (liquid) + T < 1500 -> Stone (solid)
    table[4] = PhaseTransitionRule {
        materials: UVec4::new(MAT_LAVA, PHASE_LIQUID, MAT_STONE, PHASE_SOLID),
        conditions: Vec4::new(f32::MIN, 1500.0, FLAG_RESET_F as f32, 0.0),
    };
    // Gunpowder (solid) + T > 150 -> Gas (explosion)
    table[5] = PhaseTransitionRule {
        materials: UVec4::new(MAT_GUNPOWDER, PHASE_SOLID, MAT_GUNPOWDER, PHASE_GAS),
        conditions: Vec4::new(
            150.0,
            f32::MAX,
            (FLAG_RESET_F | FLAG_EXPLOSION) as f32,
            0.0,
        ),
    };

    (table, 6)
}

/// A chemical reaction rule (contact-based, between two particles).
///
/// When `reactant_a` contacts `reactant_b`, they transform into
/// `product_a` and `product_b` respectively.
#[derive(Debug, Clone, Copy)]
pub struct ReactionRule {
    /// Material ID of the first reactant.
    pub reactant_a_material: u32,
    /// Material ID of the second reactant.
    pub reactant_b_material: u32,
    /// Material ID that reactant A becomes.
    pub product_a_material: u32,
    /// Phase that reactant A becomes.
    pub product_a_phase: u32,
    /// Material ID that reactant B becomes.
    pub product_b_material: u32,
    /// Phase that reactant B becomes.
    pub product_b_phase: u32,
    /// Minimum temperature required for reaction (f32::MIN for instant).
    pub min_temperature: f32,
    /// Temperature to set on product A (NAN = keep current).
    pub product_a_temp: f32,
    /// Temperature to set on product B (NAN = keep current).
    pub product_b_temp: f32,
}

/// Build the default chemical reaction table.
///
/// Currently includes:
/// - Water + Lava -> Stone + Steam (instant contact reaction)
pub fn default_reaction_table() -> &'static [ReactionRule] {
    &[
        // Water + Lava -> Stone + Steam
        ReactionRule {
            reactant_a_material: MAT_WATER,
            reactant_b_material: MAT_LAVA,
            product_a_material: MAT_STONE,
            product_a_phase: PHASE_SOLID,
            product_b_material: MAT_WATER,
            product_b_phase: PHASE_GAS,
            min_temperature: f32::MIN,
            product_a_temp: 300.0,
            product_b_temp: 100.0,
        },
    ]
}

/// Check if two material IDs match a reaction rule (in either order).
///
/// Returns the matching rule and whether the order is swapped.
/// If swapped, particle A is actually reactant_b and vice versa.
pub fn find_reaction(
    mat_a: u32,
    mat_b: u32,
    rules: &[ReactionRule],
) -> Option<(&ReactionRule, bool)> {
    for rule in rules {
        if mat_a == rule.reactant_a_material && mat_b == rule.reactant_b_material {
            return Some((rule, false));
        }
        if mat_a == rule.reactant_b_material && mat_b == rule.reactant_a_material {
            return Some((rule, true));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{MAT_LAVA, MAT_STONE, MAT_WATER};

    #[test]
    fn phase_transition_rule_layout() {
        assert_eq!(size_of::<PhaseTransitionRule>(), 32);
        assert_eq!(align_of::<PhaseTransitionRule>(), 16);
    }

    #[test]
    fn default_table_has_six_rules() {
        let (table, count) = default_phase_transition_table();
        assert_eq!(count, 6);
        // Remaining entries should be zeroed
        for i in count..MAX_PHASE_RULES {
            assert_eq!(table[i].materials, UVec4::ZERO);
        }
    }

    #[test]
    fn default_table_bytemuck_cast() {
        let (table, _) = default_phase_transition_table();
        let bytes: &[u8] = bytemuck::cast_slice(&table);
        assert_eq!(bytes.len(), MAX_PHASE_RULES * 32);
    }

    #[test]
    fn stone_melting_rule() {
        let (table, _) = default_phase_transition_table();
        assert_eq!(table[0].materials.x, MAT_STONE);
        assert_eq!(table[0].materials.y, PHASE_SOLID);
        assert_eq!(table[0].materials.z, MAT_LAVA);
        assert_eq!(table[0].materials.w, PHASE_LIQUID);
        assert_eq!(table[0].conditions.x, 1500.0);
    }

    #[test]
    fn gunpowder_has_explosion_flag() {
        let (table, _) = default_phase_transition_table();
        let gp_flags = table[5].conditions.z as u32;
        assert_ne!(gp_flags & FLAG_EXPLOSION, 0);
        assert_ne!(gp_flags & FLAG_RESET_F, 0);
    }

    #[test]
    fn find_water_lava_reaction() {
        let rules = default_reaction_table();
        let result = find_reaction(MAT_WATER, MAT_LAVA, rules);
        assert!(result.is_some());
        let (rule, swapped) = result.unwrap();
        assert!(!swapped);
        assert_eq!(rule.product_a_material, MAT_STONE);
    }

    #[test]
    fn find_lava_water_reaction_swapped() {
        let rules = default_reaction_table();
        let result = find_reaction(MAT_LAVA, MAT_WATER, rules);
        assert!(result.is_some());
        let (_rule, swapped) = result.unwrap();
        assert!(swapped);
    }

    #[test]
    fn no_reaction_stone_stone() {
        let rules = default_reaction_table();
        let result = find_reaction(MAT_STONE, MAT_STONE, rules);
        assert!(result.is_none());
    }

    #[test]
    fn no_reaction_water_water() {
        let rules = default_reaction_table();
        let result = find_reaction(MAT_WATER, MAT_WATER, rules);
        assert!(result.is_none());
    }

    #[test]
    fn reaction_speed_repr() {
        assert_eq!(size_of::<ReactionSpeed>(), 4);
        assert_eq!(ReactionSpeed::Instant as u32, 0);
        assert_eq!(ReactionSpeed::Slow as u32, 1);
        assert_eq!(ReactionSpeed::ChunkLevel as u32, 2);
    }

    #[test]
    fn reaction_speed_bytemuck_zeroed() {
        let speed: ReactionSpeed = bytemuck::Zeroable::zeroed();
        assert_eq!(speed, ReactionSpeed::Instant);
    }

    #[test]
    fn reaction_speed_bytemuck_cast() {
        let speed = ReactionSpeed::Slow;
        let bytes: &[u8] = bytemuck::bytes_of(&speed);
        assert_eq!(bytes.len(), 4);
        // Little-endian: value 1
        assert_eq!(bytes[0], 1);
    }

    #[test]
    fn slow_reaction_rule_layout() {
        assert_eq!(size_of::<SlowReactionRule>(), 16);
        assert_eq!(align_of::<SlowReactionRule>(), 4);
    }

    #[test]
    fn slow_reaction_rule_bytemuck_cast() {
        let rules = [SlowReactionRule {
            from_material: 1,
            to_material: 2,
            min_temperature: 500.0,
            ticks_needed: 100,
        }];
        let bytes: &[u8] = bytemuck::cast_slice(&rules);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn default_slow_table_empty() {
        let (table, count) = default_slow_reaction_table();
        assert_eq!(count, 0);
        for rule in &table {
            assert_eq!(rule.from_material, 0);
            assert_eq!(rule.to_material, 0);
            assert_eq!(rule.ticks_needed, 0);
        }
    }

    #[test]
    fn default_slow_table_bytemuck_cast() {
        let (table, _) = default_slow_reaction_table();
        let bytes: &[u8] = bytemuck::cast_slice(&table);
        assert_eq!(bytes.len(), MAX_SLOW_RULES * 16);
    }
}
