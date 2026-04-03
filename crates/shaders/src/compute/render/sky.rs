//! Sky and blackbody color functions for the render shader.

use spirv_std::glam::Vec3;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Compute blackbody-like emissive color from temperature.
///
/// Returns an RGB color that approximates blackbody radiation:
/// - Below 500C: no glow
/// - 500-800C: faint dark red
/// - 800-1200C: bright red-orange
/// - 1200-2000C: orange to yellow
/// - 2000C+: yellow-white
pub(crate) fn blackbody_color(temperature: f32) -> Vec3 {
    if temperature < 500.0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    // Normalize temperature to [0, 1] over the 500-3000 range
    let t = ((temperature - 500.0) / 2500.0).clamp(0.0, 1.0);

    // Red ramps up quickly, green follows, blue last
    let r = (t * 3.0).clamp(0.0, 1.0);
    let g = ((t - 0.2) * 2.0).clamp(0.0, 1.0);
    let b = ((t - 0.6) * 2.5).clamp(0.0, 1.0);

    // Scale intensity — brighter at higher temps
    let intensity = 0.3 + t * 1.2;
    Vec3::new(r * intensity, g * intensity, b * intensity)
}

/// Compute sky color from ray direction for background.
///
/// Returns a gradient from dark navy (horizon) to slightly lighter blue (zenith).
pub(crate) fn compute_sky_color(ray_dir: Vec3) -> Vec3 {
    let t = (ray_dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
    Vec3::new(
        0.05 + (0.1 - 0.05) * t,
        0.05 + (0.1 - 0.05) * t,
        0.12 + (0.2 - 0.12) * t,
    )
}
