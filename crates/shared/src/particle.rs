//! GPU-shared particle and grid cell types.
//!
//! **CRITICAL:** All fields are `Vec4`/`UVec4` — never `Vec3`.
//! See CLAUDE.md trap #1 for why.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, UVec4, Vec3, Vec4};

/// A single MPM particle. 144 bytes, 16-byte aligned.
///
/// Fields pack multiple values into Vec4 components:
/// - `pos_mass`: xyz = position, w = mass
/// - `vel_temp`: xyz = velocity, w = temperature
/// - `F_col0/1/2`: deformation gradient columns (w = unused)
/// - `C_col0/1/2`: APIC affine momentum columns (w = unused)
/// - `ids`: x = material_id, y = phase, z = object_id, w = padding
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Particle {
    pub pos_mass: Vec4,
    pub vel_temp: Vec4,
    pub f_col0: Vec4,
    pub f_col1: Vec4,
    pub f_col2: Vec4,
    pub c_col0: Vec4,
    pub c_col1: Vec4,
    pub c_col2: Vec4,
    pub ids: UVec4,
}

impl Particle {
    /// Create a new particle at the given position with material and phase.
    pub fn new(position: Vec3, mass: f32, material_id: u32, phase: u32) -> Self {
        Self {
            pos_mass: Vec4::new(position.x, position.y, position.z, mass),
            vel_temp: Vec4::ZERO,
            f_col0: Vec4::new(1.0, 0.0, 0.0, 0.0),
            f_col1: Vec4::new(0.0, 1.0, 0.0, 0.0),
            f_col2: Vec4::new(0.0, 0.0, 1.0, 0.0),
            c_col0: Vec4::ZERO,
            c_col1: Vec4::ZERO,
            c_col2: Vec4::ZERO,
            ids: UVec4::new(material_id, phase, 0, 0),
        }
    }

    /// Extract position as Vec3.
    #[inline]
    pub fn position(&self) -> Vec3 {
        self.pos_mass.truncate()
    }

    /// Extract velocity as Vec3.
    #[inline]
    pub fn velocity(&self) -> Vec3 {
        self.vel_temp.truncate()
    }

    /// Get mass.
    #[inline]
    pub fn mass(&self) -> f32 {
        self.pos_mass.w
    }

    /// Get temperature.
    #[inline]
    pub fn temperature(&self) -> f32 {
        self.vel_temp.w
    }

    /// Get material ID.
    #[inline]
    pub fn material_id(&self) -> u32 {
        self.ids.x
    }

    /// Get phase (0=solid, 1=liquid, 2=gas).
    #[inline]
    pub fn phase(&self) -> u32 {
        self.ids.y
    }

    /// Get damage (0.0 = undamaged, 1.0 = fully damaged).
    /// Stored in `c_col2.w` (padding field).
    #[inline]
    pub fn damage(&self) -> f32 {
        self.c_col2.w
    }

    /// Set damage value.
    #[inline]
    pub fn set_damage(&mut self, damage: f32) {
        self.c_col2.w = damage;
    }

    /// Set temperature.
    #[inline]
    pub fn set_temperature(&mut self, temp: f32) {
        self.vel_temp.w = temp;
    }

    /// Set phase.
    #[inline]
    pub fn set_phase(&mut self, phase: u32) {
        self.ids.y = phase;
    }

    /// Set velocity.
    #[inline]
    pub fn set_velocity(&mut self, vel: Vec3) {
        self.vel_temp.x = vel.x;
        self.vel_temp.y = vel.y;
        self.vel_temp.z = vel.z;
    }

    /// Set position.
    #[inline]
    pub fn set_position(&mut self, pos: Vec3) {
        self.pos_mass.x = pos.x;
        self.pos_mass.y = pos.y;
        self.pos_mass.z = pos.z;
    }

    /// Extract APIC affine momentum matrix C as Mat3.
    #[inline]
    pub fn affine_momentum(&self) -> Mat3 {
        Mat3::from_cols(
            self.c_col0.truncate(),
            self.c_col1.truncate(),
            self.c_col2.truncate(),
        )
    }

    /// Set APIC affine momentum matrix C from Mat3.
    /// Preserves `c_col2.w` (damage).
    #[inline]
    pub fn set_affine_momentum(&mut self, c: Mat3) {
        self.c_col0 = c.col(0).extend(0.0);
        self.c_col1 = c.col(1).extend(0.0);
        let damage = self.c_col2.w;
        self.c_col2 = c.col(2).extend(damage);
    }

    /// Extract deformation gradient F as Mat3.
    #[inline]
    pub fn deformation_gradient(&self) -> Mat3 {
        Mat3::from_cols(
            self.f_col0.truncate(),
            self.f_col1.truncate(),
            self.f_col2.truncate(),
        )
    }

    /// Set deformation gradient F from Mat3.
    #[inline]
    pub fn set_deformation_gradient(&mut self, f: Mat3) {
        self.f_col0 = f.col(0).extend(0.0);
        self.f_col1 = f.col(1).extend(0.0);
        self.f_col2 = f.col(2).extend(0.0);
    }

    /// Reset deformation gradient to identity (used on phase transitions).
    #[inline]
    pub fn reset_deformation_gradient(&mut self) {
        self.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        self.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        self.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
}

/// A single grid cell for MPM transfer. 48 bytes, 16-byte aligned.
///
/// - `velocity_mass`: xyz = velocity (or momentum during P2G), w = mass
/// - `force_pad`: xyz = force, w = solid flag
/// - `temp_pad`: x = accumulated temperature (weighted by mass during P2G,
///   normalized after grid_update), yzw = padding (reserved for future use)
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GridCell {
    pub velocity_mass: Vec4,
    pub force_pad: Vec4,
    pub temp_pad: Vec4,
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn particle_struct_layout() {
        assert_eq!(size_of::<Particle>(), 144);
        assert_eq!(align_of::<Particle>(), 16);

        assert_eq!(offset_of!(Particle, pos_mass), 0);
        assert_eq!(offset_of!(Particle, vel_temp), 16);
        assert_eq!(offset_of!(Particle, f_col0), 32);
        assert_eq!(offset_of!(Particle, f_col1), 48);
        assert_eq!(offset_of!(Particle, f_col2), 64);
        assert_eq!(offset_of!(Particle, c_col0), 80);
        assert_eq!(offset_of!(Particle, c_col1), 96);
        assert_eq!(offset_of!(Particle, c_col2), 112);
        assert_eq!(offset_of!(Particle, ids), 128);
    }

    #[test]
    fn grid_cell_struct_layout() {
        assert_eq!(size_of::<GridCell>(), 48);
        assert_eq!(align_of::<GridCell>(), 16);

        assert_eq!(offset_of!(GridCell, velocity_mass), 0);
        assert_eq!(offset_of!(GridCell, force_pad), 16);
        assert_eq!(offset_of!(GridCell, temp_pad), 32);
    }

    #[test]
    fn particle_new_identity_f() {
        let p = Particle::new(Vec3::new(1.0, 2.0, 3.0), 0.5, 0, 1);
        assert_eq!(p.position(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.mass(), 0.5);
        assert_eq!(p.material_id(), 0);
        assert_eq!(p.phase(), 1);
        assert_eq!(p.deformation_gradient(), Mat3::IDENTITY);
    }

    #[test]
    fn particle_bytemuck_cast() {
        let p = Particle::new(Vec3::ZERO, 1.0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 144);
    }
}
