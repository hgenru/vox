//! Material properties for the CA substrate.
//!
//! Each material type has a [`MaterialPropertiesCA`] entry in a GPU buffer,
//! indexed by material_id (0-1023).

use bytemuck::{Pod, Zeroable};

/// Material properties for the CA substrate. 64 bytes, 16-byte aligned.
///
/// All temperatures are in game units (0-255). Phase transitions and
/// combustion are driven by comparing voxel temperature against these
/// thresholds.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialPropertiesCA {
    /// Phase: 0=solid, 1=powder, 2=liquid, 3=gas.
    pub phase: u32,
    /// Density (arbitrary units, higher = heavier).
    pub density: u32,
    /// Viscosity (0=free-flowing, higher=thicker).
    pub viscosity: u32,
    /// Thermal conductivity (0=insulator, higher=conducts faster).
    pub conductivity: u32,
    /// Temperature above which this material melts.
    pub melt_temp: u32,
    /// Temperature above which this material boils/evaporates.
    pub boil_temp: u32,
    /// Temperature below which this material freezes.
    pub freeze_temp: u32,
    /// Material ID to become when melting.
    pub melt_into: u32,
    /// Material ID to become when boiling.
    pub boil_into: u32,
    /// Material ID to become when freezing.
    pub freeze_into: u32,
    /// Flammability (0=fireproof, higher=catches fire easier).
    pub flammability: u32,
    /// Temperature at which this material ignites.
    pub ignition_temp: u32,
    /// Material ID to become when burned.
    pub burn_into: u32,
    /// Heat released when burning (added to temperature).
    pub burn_heat: u32,
    /// Structural strength (for bond/support calculations).
    pub strength: u32,
    /// Maximum load before structural failure.
    pub max_load: u32,
}

// Compile-time size check
const _: () = assert!(core::mem::size_of::<MaterialPropertiesCA>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_size() {
        assert_eq!(size_of::<MaterialPropertiesCA>(), 64);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let mat: MaterialPropertiesCA = MaterialPropertiesCA::zeroed();
        assert_eq!(mat.phase, 0);
        assert_eq!(mat.density, 0);
        assert_eq!(mat.viscosity, 0);
        assert_eq!(mat.conductivity, 0);
        assert_eq!(mat.melt_temp, 0);
        assert_eq!(mat.boil_temp, 0);
        assert_eq!(mat.freeze_temp, 0);
        assert_eq!(mat.melt_into, 0);
        assert_eq!(mat.boil_into, 0);
        assert_eq!(mat.freeze_into, 0);
        assert_eq!(mat.flammability, 0);
        assert_eq!(mat.ignition_temp, 0);
        assert_eq!(mat.burn_into, 0);
        assert_eq!(mat.burn_heat, 0);
        assert_eq!(mat.strength, 0);
        assert_eq!(mat.max_load, 0);
    }
}
