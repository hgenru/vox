//! Explosion force field compute shader.
//!
//! Applies a radial impulse to all particles within a given radius of an
//! explosion center. Uses softened inverse-square falloff so particles near
//! the center receive a large but finite push, while distant ones get a
//! gentle nudge.
//!
//! Optionally accumulates damage on solid-phase particles (ids.z).

use crate::types::Particle;
use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Push constants for the explosion shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ExplosionPushConstants {
    /// Explosion center (xyz), w unused.
    pub center: Vec4,
    /// x = radius, y = strength, z = dt, w = num_particles as f32.
    pub params: Vec4,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Apply radial impulse to a single particle from an explosion.
///
/// The velocity impulse uses a linear falloff: `strength * (1 - dist/radius)`.
/// This gives a direct velocity change (not force * dt), ensuring the effect
/// is visible regardless of the simulation timestep.
/// Solid-phase particles also accumulate damage in `ids.z` (clamped to 255).
pub fn apply_explosion(particle: &mut Particle, center: Vec4, radius: f32, strength: f32, _dt: f32) {
    let pos = particle.pos_mass.truncate();
    let center_pos = center.truncate();
    let dir = pos - center_pos;
    let dist = dir.length();

    if dist >= radius || dist < 0.01 {
        return;
    }

    // Linear falloff: full strength at center, zero at radius edge.
    // This is a direct velocity impulse, NOT force * dt.
    let falloff = 1.0 - dist / radius;
    let impulse = dir * (strength * falloff / dist);

    particle.vel_temp = Vec4::new(
        particle.vel_temp.x + impulse.x,
        particle.vel_temp.y + impulse.y,
        particle.vel_temp.z + impulse.z,
        particle.vel_temp.w,
    );

    // Also push position directly (solids skip velocity update in G2P)
    particle.pos_mass = Vec4::new(
        particle.pos_mass.x + impulse.x * 0.01,
        particle.pos_mass.y + impulse.y * 0.01,
        particle.pos_mass.z + impulse.z * 0.01,
        particle.pos_mass.w,
    );

    // Solid particles near blast center: convert to liquid (debris)
    // so they start moving via normal physics
    if particle.ids.y == 0 {
        let damage_add = (strength * falloff * 0.5) as u32;
        let new_damage = particle.ids.z + damage_add;
        particle.ids.z = if new_damage > 255 { 255 } else { new_damage };

        // High damage → become liquid debris (phase 1)
        // Also heat above melting point so phase transition doesn't
        // immediately revert it back to solid
        if particle.ids.z > 100 {
            particle.ids.y = 1; // solid → liquid
            particle.vel_temp = Vec4::new(
                particle.vel_temp.x,
                particle.vel_temp.y,
                particle.vel_temp.z,
                2000.0, // above stone melting point (1500)
            );
            // Reset F = Identity (trap #8)
            particle.f_col0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
            particle.f_col1 = Vec4::new(0.0, 1.0, 0.0, 0.0);
            particle.f_col2 = Vec4::new(0.0, 0.0, 1.0, 0.0);
        }
    }
}

/// Compute shader entry point: explosion force field.
///
/// Applies a radial impulse to particles within range of the explosion center.
/// Descriptor set 0, binding 0: storage buffer of `Particle` (read-write).
/// Push constants: `ExplosionPushConstants`.
/// Dispatch with `(ceil(num_particles / 64), 1, 1)` workgroups.
#[spirv(compute(threads(64)))]
pub fn explosion(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ExplosionPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] particles: &mut [Particle],
) {
    let idx = id.x as usize;
    if id.x >= push.num_particles {
        return;
    }
    let radius = push.params.x;
    let strength = push.params.y;
    let dt = push.params.z;
    apply_explosion(&mut particles[idx], push.center, radius, strength, dt);
}
