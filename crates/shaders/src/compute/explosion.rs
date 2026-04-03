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
/// The force uses a softened inverse-square law: `strength / (dist^2 + 1.0)`.
/// Solid-phase particles also accumulate damage in `ids.z` (clamped to 255).
pub fn apply_explosion(particle: &mut Particle, center: Vec4, radius: f32, strength: f32, dt: f32) {
    let pos = particle.pos_mass.truncate();
    let center_pos = center.truncate();
    let dir = pos - center_pos;
    let dist = dir.length();

    if dist >= radius || dist < 0.01 {
        return;
    }

    let force = strength / (dist * dist + 1.0);
    let impulse = dir * (force / dist) * dt; // normalize(dir) * force * dt

    particle.vel_temp = Vec4::new(
        particle.vel_temp.x + impulse.x,
        particle.vel_temp.y + impulse.y,
        particle.vel_temp.z + impulse.z,
        particle.vel_temp.w,
    );

    // Accumulate damage on solid-phase particles
    if particle.ids.y == 0 {
        let damage_add = (force * 0.001) as u32;
        let new_damage = particle.ids.z + damage_add;
        particle.ids.z = if new_damage > 255 { 255 } else { new_damage };
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
