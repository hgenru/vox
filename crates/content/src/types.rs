//! RON-deserializable type definitions for material content.
//!
//! These types mirror the GPU-friendly structs in `shared` but use
//! human-readable field names and nested grouping suitable for RON files.

use serde::Deserialize;

/// Top-level RON structure containing all material definitions.
#[derive(Debug, Deserialize)]
pub struct MaterialDatabaseDef {
    /// All material definitions, indexed by their `id` field.
    pub materials: Vec<MaterialDef>,
    /// Phase transition rules (temperature-driven).
    pub phase_transitions: Vec<PhaseTransitionDef>,
    /// Chemical reaction rules (contact-driven).
    pub reactions: Vec<ReactionDef>,
}

/// A single material definition in the RON file.
#[derive(Debug, Deserialize)]
pub struct MaterialDef {
    /// Human-readable name (e.g. "Stone", "Water").
    pub name: String,
    /// Numeric material ID. Must match the constants in `shared::material`.
    pub id: u32,
    /// Default phase for freshly-spawned particles of this material.
    pub default_phase: PhaseDef,
    /// Elastic / mechanical properties.
    pub elastic: ElasticDef,
    /// Thermal properties.
    pub thermal: ThermalDef,
    /// Visual / rendering properties.
    pub visual: VisualDef,
}

/// Phase identifier used in RON files.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum PhaseDef {
    /// Solid phase (0).
    Solid,
    /// Liquid phase (1).
    Liquid,
    /// Gas phase (2).
    Gas,
}

impl PhaseDef {
    /// Convert to the `u32` constant used by `shared`.
    pub fn to_u32(self) -> u32 {
        match self {
            PhaseDef::Solid => shared::material::PHASE_SOLID,
            PhaseDef::Liquid => shared::material::PHASE_LIQUID,
            PhaseDef::Gas => shared::material::PHASE_GAS,
        }
    }
}

/// Elastic / mechanical properties of a material.
#[derive(Debug, Deserialize)]
pub struct ElasticDef {
    /// Young's modulus (stiffness). 0 for liquids/gas.
    pub youngs_modulus: f32,
    /// Poisson's ratio.
    pub poissons_ratio: f32,
    /// Yield stress (fracture threshold). 0 for liquids.
    pub yield_stress: f32,
    /// Dynamic viscosity. 0 for solids.
    pub viscosity: f32,
}

/// Thermal properties of a material.
#[derive(Debug, Deserialize)]
pub struct ThermalDef {
    /// Temperature at which material melts. Use `3.4028235e38` if it doesn't melt.
    pub melting_point: FloatOrInf,
    /// Temperature at which material boils. Use `3.4028235e38` if it doesn't boil.
    pub boiling_point: FloatOrInf,
    /// Heat conductivity coefficient.
    pub heat_conductivity: f32,
    /// Specific heat capacity.
    pub specific_heat: f32,
}

/// Visual / rendering properties of a material.
#[derive(Debug, Deserialize)]
pub struct VisualDef {
    /// Density in kg/m^3.
    pub density: f32,
    /// Temperature above which the material glows. Use `3.4028235e38` if it never glows.
    pub emissive_temp: FloatOrInf,
    /// Opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
    /// RGB color as [r, g, b] in 0..1 range.
    pub color: [f32; 3],
}

/// Wrapper for float values that may represent `f32::MAX` or `f32::MIN`.
///
/// In RON files, use numeric values directly. For sentinel values, use:
/// - `3.4028235e38` for `f32::MAX` (material doesn't melt/boil/glow)
/// - `-3.4028235e38` for `f32::MIN` (always triggers if below max)
pub type FloatOrInf = f32;

/// A phase transition rule definition in the RON file.
#[derive(Debug, Deserialize)]
pub struct PhaseTransitionDef {
    /// Material ID of the source particle.
    pub from_material: u32,
    /// Phase the source particle must be in.
    pub from_phase: PhaseDef,
    /// Material ID after transition.
    pub to_material: u32,
    /// Phase after transition.
    pub to_phase: PhaseDef,
    /// Minimum temperature to trigger. Use `-3.4028235e38` for "always if below max".
    pub temp_min: FloatOrInf,
    /// Maximum temperature to trigger. Use `3.4028235e38` for "always if above min".
    pub temp_max: FloatOrInf,
    /// Whether to reset deformation gradient F to identity.
    #[serde(default)]
    pub reset_deformation: bool,
    /// Whether to apply explosion behavior.
    #[serde(default)]
    pub explosion: bool,
}

/// A chemical reaction rule definition in the RON file.
#[derive(Debug, Deserialize)]
pub struct ReactionDef {
    /// Material ID of the first reactant.
    pub reactant_a: u32,
    /// Material ID of the second reactant.
    pub reactant_b: u32,
    /// Material ID that reactant A becomes.
    pub product_a_material: u32,
    /// Phase that reactant A becomes.
    pub product_a_phase: PhaseDef,
    /// Material ID that reactant B becomes.
    pub product_b_material: u32,
    /// Phase that reactant B becomes.
    pub product_b_phase: PhaseDef,
    /// Minimum temperature to trigger. Use `-3.4028235e38` for instant.
    pub min_temperature: FloatOrInf,
    /// Temperature to assign to product A.
    pub product_a_temp: FloatOrNan,
    /// Temperature to assign to product B.
    pub product_b_temp: FloatOrNan,
}

/// Wrapper for float values that may represent `f32::NAN`.
///
/// In RON files, use `nan` for "keep current temperature".
pub type FloatOrNan = f32;
