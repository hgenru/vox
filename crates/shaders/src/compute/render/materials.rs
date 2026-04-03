//! Material color and emissive functions for the render shader.

use crate::types::{MaterialParams, MAT_WOOD};
use spirv_std::glam::Vec3;

use super::math::value_noise;
use super::sky::blackbody_color;

/// Get procedurally textured color for a material at a given voxel position.
///
/// Reads the base color from the MaterialParams buffer, then applies procedural
/// noise on top for per-voxel visual variation. Returns (r, g, b) as floats in [0, 1].
pub(crate) fn textured_material_color(
    material_id: u32,
    vx: i32,
    vy: i32,
    vz: i32,
    materials: &[MaterialParams],
) -> Vec3 {
    // Read base color from MaterialParams buffer (safe index with fallback)
    let mat_count = materials.len() as u32;
    let safe_id = if material_id < mat_count {
        material_id
    } else {
        0
    };
    let base = materials[safe_id as usize].color.truncate();

    // Apply procedural noise for per-voxel variation: base_color * (0.85 + noise * 0.15)
    let noise = value_noise(vx as f32 * 1.0, vy as f32 * 1.0, vz as f32 * 1.0);
    let factor = 0.85 + noise * 0.15;

    Vec3::new(base.x * factor, base.y * factor, base.z * factor)
}

/// Apply emissive glow for lava (material_id=2).
///
/// Uses temperature-based blackbody color for physically plausible glow.
fn emissive_lava(lit_color: Vec3, temperature: f32) -> Vec3 {
    let emissive = blackbody_color(temperature);
    Vec3::new(
        lit_color.x + emissive.x,
        lit_color.y + emissive.y,
        lit_color.z + emissive.z,
    )
}

/// Apply emissive glow for burning wood (material_id=3).
///
/// Uses temperature-based blackbody with reduced intensity.
fn emissive_wood(lit_color: Vec3, temperature: f32) -> Vec3 {
    let emissive = blackbody_color(temperature);
    // Wood burns less brightly than lava — scale down
    let scale = 0.6;
    Vec3::new(
        lit_color.x + emissive.x * scale,
        lit_color.y + emissive.y * scale,
        lit_color.z + emissive.z * scale,
    )
}

/// Apply emissive glow for burning gunpowder gas (material_id=6, phase=2).
///
/// Bright white-orange flash using temperature.
fn emissive_gunpowder(lit_color: Vec3, temperature: f32) -> Vec3 {
    let emissive = blackbody_color(temperature);
    // Gunpowder explosion is very bright
    let scale = 1.5;
    Vec3::new(
        lit_color.x + emissive.x * scale,
        lit_color.y + emissive.y * scale,
        lit_color.z + emissive.z * scale,
    )
}

/// Apply emissive glow to a lit surface color based on material and temperature.
///
/// Split into per-material helpers to avoid >3 if/else branches (trap #15).
pub(crate) fn apply_emissive(lit_color: Vec3, material_id: u32, phase: u32, temperature: f32) -> Vec3 {
    if material_id == 2 {
        return emissive_lava(lit_color, temperature);
    }
    if material_id == MAT_WOOD {
        return emissive_wood(lit_color, temperature);
    }
    if material_id == 6 && phase == 2 {
        return emissive_gunpowder(lit_color, temperature);
    }
    lit_color
}
