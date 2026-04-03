//! World configuration parameters.
//!
//! [`WorldConfig`] is a GPU-uploadable uniform buffer containing all
//! physics constants that can vary per-world (gravity, wind, timestep, etc.).

use bytemuck::{Pod, Zeroable};
use glam::Vec4;

/// World configuration parameters. Uploaded to GPU as uniform buffer.
///
/// All physics constants that could vary per-world go here.
/// Uses [`Vec4`] everywhere to satisfy GPU alignment requirements (trap #1).
///
/// # Layout
/// - `gravity`: Gravity vector (xyz) + padding (w). Default: `(0, -196, 0, 0)`.
/// - `wind`: Wind force vector (xyz) + padding (w). Default: `(0, 0, 0, 0)`.
/// - `params`: Packed parameters:
///   - `x` = ambient temperature (Celsius). Default: `20.0`.
///   - `y` = air density (kg/m^3). Default: `1.225`.
///   - `z` = simulation timestep (seconds). Default: `0.001`.
///   - `w` = grid size as f32. Default: `256.0`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct WorldConfig {
    /// Gravity vector (xyz) + padding (w). Default: `(0, -196, 0, 0)` for 5cm voxels.
    pub gravity: Vec4,
    /// Wind force vector (xyz) + padding (w). Default: `(0, 0, 0, 0)`.
    pub wind: Vec4,
    /// Packed parameters: x=ambient_temperature, y=air_density, z=dt, w=grid_size as f32.
    pub params: Vec4,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vec4::new(0.0, -196.0, 0.0, 0.0),
            wind: Vec4::ZERO,
            params: Vec4::new(20.0, 1.225, 0.001, 256.0),
        }
    }
}

impl WorldConfig {
    /// Returns the gravity vector as (x, y, z).
    pub fn gravity_xyz(&self) -> (f32, f32, f32) {
        (self.gravity.x, self.gravity.y, self.gravity.z)
    }

    /// Returns the wind vector as (x, y, z).
    pub fn wind_xyz(&self) -> (f32, f32, f32) {
        (self.wind.x, self.wind.y, self.wind.z)
    }

    /// Returns the ambient temperature in Celsius.
    pub fn ambient_temperature(&self) -> f32 {
        self.params.x
    }

    /// Returns the air density in kg/m^3.
    pub fn air_density(&self) -> f32 {
        self.params.y
    }

    /// Returns the simulation timestep in seconds.
    pub fn dt(&self) -> f32 {
        self.params.z
    }

    /// Returns the grid size (cells per axis) as f32.
    pub fn grid_size(&self) -> f32 {
        self.params.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn world_config_size_and_alignment() {
        // 3 Vec4s = 3 * 16 = 48 bytes
        assert_eq!(mem::size_of::<WorldConfig>(), 48);
        assert_eq!(mem::align_of::<WorldConfig>(), 16);
    }

    #[test]
    fn world_config_default_values() {
        let cfg = WorldConfig::default();

        // Gravity: -196 on Y axis
        assert_eq!(cfg.gravity.x, 0.0);
        assert_eq!(cfg.gravity.y, -196.0);
        assert_eq!(cfg.gravity.z, 0.0);
        assert_eq!(cfg.gravity.w, 0.0);

        // Wind: zero
        assert_eq!(cfg.wind, Vec4::ZERO);

        // Params
        assert_eq!(cfg.ambient_temperature(), 20.0);
        assert!((cfg.air_density() - 1.225).abs() < 1e-6);
        assert_eq!(cfg.dt(), 0.001);
        assert_eq!(cfg.grid_size(), 256.0);
    }

    #[test]
    fn world_config_accessors() {
        let cfg = WorldConfig {
            gravity: Vec4::new(1.0, -9.81, 0.5, 0.0),
            wind: Vec4::new(5.0, 0.0, -2.0, 0.0),
            params: Vec4::new(25.0, 1.0, 0.002, 128.0),
        };

        assert_eq!(cfg.gravity_xyz(), (1.0, -9.81, 0.5));
        assert_eq!(cfg.wind_xyz(), (5.0, 0.0, -2.0));
        assert_eq!(cfg.ambient_temperature(), 25.0);
        assert_eq!(cfg.air_density(), 1.0);
        assert_eq!(cfg.dt(), 0.002);
        assert_eq!(cfg.grid_size(), 128.0);
    }

    #[test]
    fn world_config_is_pod() {
        // This compiles only if WorldConfig: Pod + Zeroable
        let zeroed: WorldConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.gravity, Vec4::ZERO);
        let cfg = WorldConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&cfg);
        assert_eq!(bytes.len(), 48);
    }
}
