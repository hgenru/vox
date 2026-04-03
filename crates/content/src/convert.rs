//! Conversion from RON-deserialized types to GPU-friendly `shared` structs.

use std::collections::HashSet;

use glam::Vec4;
use shared::{
    material::{MATERIAL_COUNT, MaterialParams},
    reaction::{
        FLAG_EXPLOSION, FLAG_RESET_F, MAX_PHASE_RULES, PhaseTransitionRule, ReactionRule,
    },
};

use crate::{
    ContentError,
    types::{MaterialDatabaseDef, MaterialDef, PhaseTransitionDef, ReactionDef},
};

/// Validated material database ready for GPU upload.
///
/// Created via [`MaterialDatabase::from_def`] which validates all
/// cross-references and constraints. Use the accessor methods to obtain
/// GPU-friendly arrays for upload to storage buffers.
#[derive(Debug)]
pub struct MaterialDatabase {
    materials: Vec<MaterialDef>,
    phase_transitions: Vec<PhaseTransitionDef>,
    reactions: Vec<ReactionDef>,
}

impl MaterialDatabase {
    /// Validate and construct a [`MaterialDatabase`] from a parsed RON definition.
    ///
    /// # Errors
    ///
    /// Returns [`ContentError`] if:
    /// - Any material IDs are duplicated
    /// - Phase transitions reference unknown material IDs
    /// - Reactions reference unknown material IDs
    /// - Too many phase transition rules for the GPU buffer
    pub fn from_def(def: MaterialDatabaseDef) -> std::result::Result<Self, ContentError> {
        // Check for duplicate material ids
        let mut seen_ids = HashSet::new();
        for mat in &def.materials {
            if !seen_ids.insert(mat.id) {
                return Err(ContentError::DuplicateMaterialId(mat.id));
            }
        }

        // Validate phase transition material references
        for pt in &def.phase_transitions {
            if !seen_ids.contains(&pt.from_material) {
                return Err(ContentError::UnknownTransitionMaterial(pt.from_material));
            }
            if !seen_ids.contains(&pt.to_material) {
                return Err(ContentError::UnknownTransitionMaterial(pt.to_material));
            }
        }

        // Validate reaction material references
        for r in &def.reactions {
            if !seen_ids.contains(&r.reactant_a) {
                return Err(ContentError::UnknownReactionMaterial(r.reactant_a));
            }
            if !seen_ids.contains(&r.reactant_b) {
                return Err(ContentError::UnknownReactionMaterial(r.reactant_b));
            }
            if !seen_ids.contains(&r.product_a_material) {
                return Err(ContentError::UnknownReactionMaterial(r.product_a_material));
            }
            if !seen_ids.contains(&r.product_b_material) {
                return Err(ContentError::UnknownReactionMaterial(r.product_b_material));
            }
        }

        if def.phase_transitions.len() > MAX_PHASE_RULES {
            return Err(ContentError::TooManyPhaseRules {
                count: def.phase_transitions.len(),
                max: MAX_PHASE_RULES,
            });
        }

        tracing::info!(
            "Loaded {} materials, {} phase transitions, {} reactions",
            def.materials.len(),
            def.phase_transitions.len(),
            def.reactions.len(),
        );

        Ok(Self {
            materials: def.materials,
            phase_transitions: def.phase_transitions,
            reactions: def.reactions,
        })
    }

    /// Convert materials to the GPU-friendly array indexed by material ID.
    ///
    /// Returns a fixed-size array of [`MaterialParams`] where each material
    /// is placed at its `id` index. Unused slots are zeroed.
    pub fn material_params(&self) -> [MaterialParams; MATERIAL_COUNT] {
        let mut table = [MaterialParams {
            elastic: Vec4::ZERO,
            thermal: Vec4::ZERO,
            visual: Vec4::ZERO,
            color: Vec4::ZERO,
        }; MATERIAL_COUNT];

        for mat in &self.materials {
            let idx = mat.id as usize;
            if idx < MATERIAL_COUNT {
                table[idx] = MaterialParams {
                    elastic: Vec4::new(
                        mat.elastic.youngs_modulus,
                        mat.elastic.poissons_ratio,
                        mat.elastic.yield_stress,
                        mat.elastic.viscosity,
                    ),
                    thermal: Vec4::new(
                        mat.thermal.melting_point,
                        mat.thermal.boiling_point,
                        mat.thermal.heat_conductivity,
                        mat.thermal.specific_heat,
                    ),
                    visual: Vec4::new(
                        mat.visual.density,
                        mat.visual.emissive_temp,
                        mat.visual.opacity,
                        0.0,
                    ),
                    color: Vec4::new(mat.visual.color[0], mat.visual.color[1], mat.visual.color[2], 0.0),
                };
            }
        }

        table
    }

    /// Convert phase transitions to the GPU-friendly fixed-size array.
    ///
    /// Returns the array and the count of valid entries. Entries beyond `count`
    /// are zeroed.
    pub fn phase_transition_rules(&self) -> ([PhaseTransitionRule; MAX_PHASE_RULES], usize) {
        let mut table = [PhaseTransitionRule {
            materials: glam::UVec4::ZERO,
            conditions: Vec4::ZERO,
        }; MAX_PHASE_RULES];

        for (i, pt) in self.phase_transitions.iter().enumerate() {
            if i >= MAX_PHASE_RULES {
                break;
            }
            let mut flags: u32 = 0;
            if pt.reset_deformation {
                flags |= FLAG_RESET_F;
            }
            if pt.explosion {
                flags |= FLAG_EXPLOSION;
            }
            table[i] = PhaseTransitionRule {
                materials: glam::UVec4::new(
                    pt.from_material,
                    pt.from_phase.to_u32(),
                    pt.to_material,
                    pt.to_phase.to_u32(),
                ),
                conditions: Vec4::new(pt.temp_min, pt.temp_max, flags as f32, 0.0),
            };
        }

        (table, self.phase_transitions.len().min(MAX_PHASE_RULES))
    }

    /// Convert reactions to shared [`ReactionRule`] structs.
    ///
    /// Returns a `Vec` of reaction rules ready for use by the simulation.
    pub fn reaction_rules(&self) -> Vec<ReactionRule> {
        self.reactions
            .iter()
            .map(|r| ReactionRule {
                reactant_a_material: r.reactant_a,
                reactant_b_material: r.reactant_b,
                product_a_material: r.product_a_material,
                product_a_phase: r.product_a_phase.to_u32(),
                product_b_material: r.product_b_material,
                product_b_phase: r.product_b_phase.to_u32(),
                min_temperature: r.min_temperature,
                product_a_temp: r.product_a_temp,
                product_b_temp: r.product_b_temp,
            })
            .collect()
    }

    /// Get the list of material definitions.
    pub fn materials(&self) -> &[MaterialDef] {
        &self.materials
    }

    /// Get the list of phase transition definitions.
    pub fn phase_transitions(&self) -> &[PhaseTransitionDef] {
        &self.phase_transitions
    }

    /// Get the list of reaction definitions.
    pub fn reactions(&self) -> &[ReactionDef] {
        &self.reactions
    }
}

#[cfg(test)]
mod tests {
    use shared::material;
    use shared::reaction;

    use crate::parse_material_database;

    /// Load the canonical RON file and compare with hardcoded defaults.
    #[test]
    fn ron_matches_hardcoded_materials() {
        let ron_str = include_str!("../../../assets/materials.ron");
        let db = parse_material_database(ron_str).expect("Failed to parse materials.ron");
        let loaded = db.material_params();
        let hardcoded = material::default_material_table();

        for i in 0..material::MATERIAL_COUNT {
            let l = &loaded[i];
            let h = &hardcoded[i];
            assert!(
                vecs_approx_eq(l.elastic, h.elastic),
                "Material {i} elastic mismatch: loaded={:?} hardcoded={:?}",
                l.elastic,
                h.elastic,
            );
            assert!(
                vecs_approx_eq(l.thermal, h.thermal),
                "Material {i} thermal mismatch: loaded={:?} hardcoded={:?}",
                l.thermal,
                h.thermal,
            );
            assert!(
                vecs_approx_eq(l.visual, h.visual),
                "Material {i} visual mismatch: loaded={:?} hardcoded={:?}",
                l.visual,
                h.visual,
            );
            assert!(
                vecs_approx_eq(l.color, h.color),
                "Material {i} color mismatch: loaded={:?} hardcoded={:?}",
                l.color,
                h.color,
            );
        }
    }

    #[test]
    fn ron_matches_hardcoded_phase_transitions() {
        let ron_str = include_str!("../../../assets/materials.ron");
        let db = parse_material_database(ron_str).expect("Failed to parse materials.ron");
        let (loaded, loaded_count) = db.phase_transition_rules();
        let (hardcoded, hardcoded_count) = reaction::default_phase_transition_table();

        assert_eq!(loaded_count, hardcoded_count, "Phase rule count mismatch");
        for i in 0..loaded_count {
            assert_eq!(
                loaded[i].materials, hardcoded[i].materials,
                "Phase rule {i} materials mismatch"
            );
            assert!(
                vecs_approx_eq(loaded[i].conditions, hardcoded[i].conditions),
                "Phase rule {i} conditions mismatch: loaded={:?} hardcoded={:?}",
                loaded[i].conditions,
                hardcoded[i].conditions,
            );
        }
    }

    #[test]
    fn ron_matches_hardcoded_reactions() {
        let ron_str = include_str!("../../../assets/materials.ron");
        let db = parse_material_database(ron_str).expect("Failed to parse materials.ron");
        let loaded = db.reaction_rules();
        let hardcoded = reaction::default_reaction_table();

        assert_eq!(loaded.len(), hardcoded.len(), "Reaction count mismatch");
        for (i, (l, h)) in loaded.iter().zip(hardcoded.iter()).enumerate() {
            assert_eq!(l.reactant_a_material, h.reactant_a_material, "Reaction {i} reactant_a");
            assert_eq!(l.reactant_b_material, h.reactant_b_material, "Reaction {i} reactant_b");
            assert_eq!(l.product_a_material, h.product_a_material, "Reaction {i} product_a_mat");
            assert_eq!(l.product_a_phase, h.product_a_phase, "Reaction {i} product_a_phase");
            assert_eq!(l.product_b_material, h.product_b_material, "Reaction {i} product_b_mat");
            assert_eq!(l.product_b_phase, h.product_b_phase, "Reaction {i} product_b_phase");
            assert!(
                floats_approx_eq(l.min_temperature, h.min_temperature),
                "Reaction {i} min_temp mismatch",
            );
            assert!(
                floats_approx_eq(l.product_a_temp, h.product_a_temp),
                "Reaction {i} product_a_temp mismatch",
            );
            assert!(
                floats_approx_eq(l.product_b_temp, h.product_b_temp),
                "Reaction {i} product_b_temp mismatch",
            );
        }
    }

    #[test]
    fn duplicate_material_id_rejected() {
        let ron = r#"(
            materials: [
                (name: "A", id: 0, default_phase: Solid, elastic: (youngs_modulus: 0.0, poissons_ratio: 0.0, yield_stress: 0.0, viscosity: 0.0), thermal: (melting_point: Inf, boiling_point: Inf, heat_conductivity: 0.0, specific_heat: 0.0), visual: (density: 0.0, emissive_temp: Inf, opacity: 1.0, color: [0.0, 0.0, 0.0])),
                (name: "B", id: 0, default_phase: Solid, elastic: (youngs_modulus: 0.0, poissons_ratio: 0.0, yield_stress: 0.0, viscosity: 0.0), thermal: (melting_point: Inf, boiling_point: Inf, heat_conductivity: 0.0, specific_heat: 0.0), visual: (density: 0.0, emissive_temp: Inf, opacity: 1.0, color: [0.0, 0.0, 0.0])),
            ],
            phase_transitions: [],
            reactions: [],
        )"#;
        let result = parse_material_database(ron);
        assert!(result.is_err());
    }

    #[test]
    fn unknown_transition_material_rejected() {
        let ron = r#"(
            materials: [
                (name: "A", id: 0, default_phase: Solid, elastic: (youngs_modulus: 0.0, poissons_ratio: 0.0, yield_stress: 0.0, viscosity: 0.0), thermal: (melting_point: Inf, boiling_point: Inf, heat_conductivity: 0.0, specific_heat: 0.0), visual: (density: 0.0, emissive_temp: Inf, opacity: 1.0, color: [0.0, 0.0, 0.0])),
            ],
            phase_transitions: [
                (from_material: 0, from_phase: Solid, to_material: 99, to_phase: Liquid, temp_min: 100.0, temp_max: Inf, reset_deformation: true),
            ],
            reactions: [],
        )"#;
        let result = parse_material_database(ron);
        assert!(result.is_err());
    }

    /// Compare two Vec4s with tolerance, treating f32::MAX == f32::MAX as equal.
    fn vecs_approx_eq(a: glam::Vec4, b: glam::Vec4) -> bool {
        floats_approx_eq(a.x, b.x)
            && floats_approx_eq(a.y, b.y)
            && floats_approx_eq(a.z, b.z)
            && floats_approx_eq(a.w, b.w)
    }

    fn floats_approx_eq(a: f32, b: f32) -> bool {
        if a == b {
            return true; // handles infinity, MAX, MIN
        }
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < 1e-3
    }
}
