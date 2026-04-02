//! Chemical reaction rules for particle interactions.
//!
//! Reactions are checked when particles of different materials are in contact
//! (overlapping grid cells). Each rule specifies two reactant materials and
//! what they transform into.

/// A chemical reaction rule.
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

/// Build the default reaction table.
///
/// Currently includes:
/// - Water + Lava -> Stone + Steam (instant contact reaction)
pub fn default_reaction_table() -> &'static [ReactionRule] {
    &[
        // Water + Lava -> Stone + Steam
        ReactionRule {
            reactant_a_material: crate::material::MAT_WATER,
            reactant_b_material: crate::material::MAT_LAVA,
            product_a_material: crate::material::MAT_STONE,
            product_a_phase: crate::material::PHASE_SOLID,
            product_b_material: crate::material::MAT_WATER,
            product_b_phase: crate::material::PHASE_GAS,
            min_temperature: f32::MIN,
            product_a_temp: 300.0,  // cooled stone
            product_b_temp: 100.0,  // steam at boiling point
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
}
