//! Material parameters and material table.
//!
//! Each material has elastic, thermal, visual, and color properties
//! packed into Vec4 fields for GPU compatibility.

use bytemuck::{Pod, Zeroable};
use glam::Vec4;

/// Material parameters for GPU uniform buffer. 64 bytes per material.
///
/// - `elastic`: x=youngs_modulus, y=poissons_ratio, z=yield_stress, w=viscosity
/// - `thermal`: x=melting_point, y=boiling_point, z=heat_conductivity, w=specific_heat
/// - `visual`: x=density, y=emissive_temp, z=opacity, w=pad
/// - `color`: xyz=rgb (0..1), w=pad
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct MaterialParams {
    pub elastic: Vec4,
    pub thermal: Vec4,
    pub visual: Vec4,
    pub color: Vec4,
}

/// Material ID constants.
pub const MAT_STONE: u32 = 0;
pub const MAT_WATER: u32 = 1;
pub const MAT_LAVA: u32 = 2;

/// Phase constants.
pub const PHASE_SOLID: u32 = 0;
pub const PHASE_LIQUID: u32 = 1;
pub const PHASE_GAS: u32 = 2;

/// Number of materials in the table.
pub const MATERIAL_COUNT: usize = 3;

/// Build the default material parameter table.
///
/// Returns an array of `MaterialParams` indexed by material ID.
/// Stone=0, Water=1, Lava=2.
pub fn default_material_table() -> [MaterialParams; MATERIAL_COUNT] {
    [
        // Stone (solid)
        MaterialParams {
            elastic: Vec4::new(
                1e6, // youngs_modulus
                0.3, // poissons_ratio
                1e4, // yield_stress
                0.0, // viscosity (solid — not used)
            ),
            thermal: Vec4::new(
                1500.0,   // melting_point → becomes lava
                f32::MAX, // boiling_point (stone doesn't boil)
                1.0,      // heat_conductivity
                800.0,    // specific_heat
            ),
            visual: Vec4::new(
                2700.0,   // density (kg/m³)
                f32::MAX, // emissive_temp (stone doesn't glow)
                1.0,      // opacity
                0.0,      // pad
            ),
            color: Vec4::new(0.5, 0.5, 0.5, 0.0), // gray
        },
        // Water (liquid)
        MaterialParams {
            elastic: Vec4::new(
                0.0,  // youngs_modulus (liquid — not used)
                0.0,  // poissons_ratio
                0.0,  // yield_stress
                1e-3, // viscosity
            ),
            thermal: Vec4::new(
                0.0,    // melting_point (water freezes at 0)
                100.0,  // boiling_point → steam
                0.6,    // heat_conductivity
                4186.0, // specific_heat
            ),
            visual: Vec4::new(
                1000.0,   // density
                f32::MAX, // emissive_temp
                0.3,      // opacity (translucent)
                0.0,
            ),
            color: Vec4::new(0.2, 0.4, 0.9, 0.0), // blue
        },
        // Lava (liquid, hot)
        MaterialParams {
            elastic: Vec4::new(
                0.0,   // youngs_modulus
                0.0,   // poissons_ratio
                0.0,   // yield_stress
                100.0, // viscosity (very viscous)
            ),
            thermal: Vec4::new(
                1500.0,   // melting_point (below this → stone)
                f32::MAX, // boiling_point
                1.5,      // heat_conductivity
                1000.0,   // specific_heat
            ),
            visual: Vec4::new(
                2500.0, // density
                800.0,  // emissive_temp (glows above this)
                1.0,    // opacity
                0.0,
            ),
            color: Vec4::new(1.0, 0.3, 0.0, 0.0), // orange-red
        },
    ]
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn material_params_struct_layout() {
        assert_eq!(size_of::<MaterialParams>(), 64);
        assert_eq!(align_of::<MaterialParams>(), 16);

        assert_eq!(offset_of!(MaterialParams, elastic), 0);
        assert_eq!(offset_of!(MaterialParams, thermal), 16);
        assert_eq!(offset_of!(MaterialParams, visual), 32);
        assert_eq!(offset_of!(MaterialParams, color), 48);
    }

    #[test]
    fn material_table_uniform_buffer_size() {
        let table = default_material_table();
        let total_bytes = size_of::<MaterialParams>() * table.len();
        // 3 materials × 64 bytes = 192 bytes (fits in uniform buffer)
        assert_eq!(total_bytes, 192);
    }

    #[test]
    fn material_table_bytemuck_cast() {
        let table = default_material_table();
        let bytes: &[u8] = bytemuck::cast_slice(&table);
        assert_eq!(bytes.len(), 192);
    }

    #[test]
    fn stone_is_material_zero() {
        let table = default_material_table();
        assert_eq!(table[MAT_STONE as usize].color.x, 0.5); // gray
    }

    #[test]
    fn water_is_material_one() {
        let table = default_material_table();
        assert!(table[MAT_WATER as usize].elastic.w > 0.0); // has viscosity
        assert!(table[MAT_WATER as usize].visual.z < 1.0); // translucent
    }

    #[test]
    fn lava_is_material_two() {
        let table = default_material_table();
        assert!(table[MAT_LAVA as usize].elastic.w > 1.0); // very viscous
        assert!(table[MAT_LAVA as usize].visual.y < 1000.0); // emissive
    }
}
