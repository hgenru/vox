//! MagicaVoxel `.vox` model loader.
//!
//! Converts `.vox` voxel models into engine [`Particle`]s, mapping palette
//! colours to materials via simple heuristic rules or a user-supplied palette.

use glam::Vec3;
use shared::{
    MAT_GUNPOWDER, MAT_ICE, MAT_LAVA, MAT_STONE, MAT_WATER, MAT_WOOD, PHASE_LIQUID, PHASE_SOLID,
    Particle,
};

use crate::ContentError;

/// RGBA colour extracted from a MagicaVoxel palette entry.
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
    /// Alpha channel (0-255).
    pub a: u8,
}

/// A range of colours described by independent per-channel bounds.
#[derive(Debug, Clone)]
pub struct ColorRange {
    /// Minimum red value (inclusive).
    pub r_min: u8,
    /// Maximum red value (inclusive).
    pub r_max: u8,
    /// Minimum green value (inclusive).
    pub g_min: u8,
    /// Maximum green value (inclusive).
    pub g_max: u8,
    /// Minimum blue value (inclusive).
    pub b_min: u8,
    /// Maximum blue value (inclusive).
    pub b_max: u8,
}

impl ColorRange {
    /// Check whether `color` falls within this range on all channels.
    pub fn contains(&self, c: &Color) -> bool {
        c.r >= self.r_min
            && c.r <= self.r_max
            && c.g >= self.g_min
            && c.g <= self.g_max
            && c.b >= self.b_min
            && c.b <= self.b_max
    }
}

/// Mapping from a palette colour to a game material.
#[derive(Debug, Clone)]
pub struct PaletteMapping {
    /// Engine material id (e.g. `MAT_STONE`).
    pub material_id: u32,
    /// Phase constant (e.g. `PHASE_SOLID`).
    pub phase: u32,
    /// Default temperature in degrees Celsius.
    pub temperature: f32,
}

/// Classify a single colour into a [`PaletteMapping`] using simple heuristics.
///
/// Rules (evaluated in order):
/// - Dark (r<50, g<50, b<50) -> gunpowder
/// - Blue dominant (b > r+30 && b > g+30) -> water
/// - Red/orange (r>150, g<100, b<80) -> lava (T=2000)
/// - Brown (r>100, g>50, b<80, r>g) -> wood
/// - White/light (r>200, g>200, b>200) -> ice (T=-50)
/// - Gray (r~=g~=b, r>100) -> stone
/// - Default -> stone
pub fn classify_color(c: &Color) -> PaletteMapping {
    // Dark -> gunpowder
    if c.r < 50 && c.g < 50 && c.b < 50 {
        return PaletteMapping {
            material_id: MAT_GUNPOWDER,
            phase: PHASE_SOLID,
            temperature: 20.0,
        };
    }
    // Blue dominant -> water
    if c.b > c.r.saturating_add(30) && c.b > c.g.saturating_add(30) {
        return PaletteMapping {
            material_id: MAT_WATER,
            phase: PHASE_LIQUID,
            temperature: 20.0,
        };
    }
    // Red/orange -> lava
    if c.r > 150 && c.g < 100 && c.b < 80 {
        return PaletteMapping {
            material_id: MAT_LAVA,
            phase: PHASE_LIQUID,
            temperature: 2000.0,
        };
    }
    // Brown -> wood
    if c.r > 100 && c.g > 50 && c.b < 80 && c.r > c.g {
        return PaletteMapping {
            material_id: MAT_WOOD,
            phase: PHASE_SOLID,
            temperature: 20.0,
        };
    }
    // White/light -> ice
    if c.r > 200 && c.g > 200 && c.b > 200 {
        return PaletteMapping {
            material_id: MAT_ICE,
            phase: PHASE_SOLID,
            temperature: -50.0,
        };
    }
    // Gray -> stone
    let rg_diff = (c.r as i16 - c.g as i16).unsigned_abs();
    let rb_diff = (c.r as i16 - c.b as i16).unsigned_abs();
    if rg_diff < 30 && rb_diff < 30 && c.r > 100 {
        return PaletteMapping {
            material_id: MAT_STONE,
            phase: PHASE_SOLID,
            temperature: 20.0,
        };
    }
    // Default -> stone
    PaletteMapping {
        material_id: MAT_STONE,
        phase: PHASE_SOLID,
        temperature: 20.0,
    }
}

/// Build the default palette mapping table using [`classify_color`] heuristics.
///
/// Returns a vector of `(ColorRange, PaletteMapping)` rules evaluated in order.
/// The first matching range wins.
pub fn default_palette_mapping() -> Vec<(ColorRange, PaletteMapping)> {
    vec![
        // Dark -> gunpowder
        (
            ColorRange { r_min: 0, r_max: 49, g_min: 0, g_max: 49, b_min: 0, b_max: 49 },
            PaletteMapping { material_id: MAT_GUNPOWDER, phase: PHASE_SOLID, temperature: 20.0 },
        ),
        // Blue -> water
        (
            ColorRange { r_min: 0, r_max: 100, g_min: 0, g_max: 100, b_min: 130, b_max: 255 },
            PaletteMapping { material_id: MAT_WATER, phase: PHASE_LIQUID, temperature: 20.0 },
        ),
        // Red/orange -> lava
        (
            ColorRange { r_min: 151, r_max: 255, g_min: 0, g_max: 99, b_min: 0, b_max: 79 },
            PaletteMapping { material_id: MAT_LAVA, phase: PHASE_LIQUID, temperature: 2000.0 },
        ),
        // Brown -> wood
        (
            ColorRange { r_min: 101, r_max: 200, g_min: 51, g_max: 150, b_min: 0, b_max: 79 },
            PaletteMapping { material_id: MAT_WOOD, phase: PHASE_SOLID, temperature: 20.0 },
        ),
        // White/light -> ice
        (
            ColorRange { r_min: 201, r_max: 255, g_min: 201, g_max: 255, b_min: 201, b_max: 255 },
            PaletteMapping { material_id: MAT_ICE, phase: PHASE_SOLID, temperature: -50.0 },
        ),
        // Gray -> stone (fallback-ish; 100..200 with similar channels)
        (
            ColorRange { r_min: 80, r_max: 200, g_min: 80, g_max: 200, b_min: 80, b_max: 200 },
            PaletteMapping { material_id: MAT_STONE, phase: PHASE_SOLID, temperature: 20.0 },
        ),
    ]
}

/// Find the first matching palette entry for a colour, or fall back to
/// [`classify_color`] if no range matches.
fn find_palette_match(color: Color, palette: &[(ColorRange, PaletteMapping)]) -> PaletteMapping {
    for (range, mapping) in palette {
        if range.contains(&color) {
            return mapping.clone();
        }
    }
    classify_color(&color)
}

/// Load a `.vox` (MagicaVoxel) file and convert each voxel to a [`Particle`].
///
/// The first model in the file is used. Each voxel becomes one particle placed
/// at `offset + voxel_position + 0.5` (centre of the voxel cell).
///
/// MagicaVoxel uses Z-up coordinates while the engine uses Y-up, so the axis
/// mapping is: MV.x -> X, MV.z -> Y, MV.y -> Z.
///
/// # Errors
///
/// Returns [`ContentError::VoxLoad`] if the file cannot be read or parsed.
pub fn load_vox_model(
    path: &str,
    offset: Vec3,
    palette: &[(ColorRange, PaletteMapping)],
) -> Result<Vec<Particle>, ContentError> {
    let data =
        dot_vox::load(path).map_err(|e| ContentError::VoxLoad(e.to_string()))?;
    let model = data
        .models
        .first()
        .ok_or_else(|| ContentError::VoxLoad("no models in .vox file".to_string()))?;

    let mut particles = Vec::with_capacity(model.voxels.len());

    for voxel in &model.voxels {
        // dot_vox palette is 256 entries; voxel.i is 1-based index into palette
        // (index 0 is unused/empty). The `dot_vox` crate already adjusts this so
        // `data.palette[voxel.i as usize]` gives the correct colour.
        let raw = data.palette[voxel.i as usize];
        let color = Color { r: raw.r, g: raw.g, b: raw.b, a: raw.a };
        let mapping = find_palette_match(color, palette);

        // Axis swap: MagicaVoxel Z-up -> engine Y-up
        let pos = Vec3::new(
            voxel.x as f32 + offset.x + 0.5,
            voxel.z as f32 + offset.y + 0.5,
            voxel.y as f32 + offset.z + 0.5,
        );

        let mut p = Particle::new(pos, 1.0, mapping.material_id, mapping.phase);
        if (mapping.temperature - 20.0).abs() > f32::EPSILON {
            p.set_temperature(mapping.temperature);
        }
        particles.push(p);
    }

    tracing::info!(
        "Loaded .vox model '{}': {} voxels -> {} particles",
        path,
        model.voxels.len(),
        particles.len()
    );

    Ok(particles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_dark_as_gunpowder() {
        let c = Color { r: 10, g: 10, b: 10, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_GUNPOWDER);
        assert_eq!(m.phase, PHASE_SOLID);
    }

    #[test]
    fn classify_blue_as_water() {
        let c = Color { r: 30, g: 50, b: 200, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_WATER);
        assert_eq!(m.phase, PHASE_LIQUID);
    }

    #[test]
    fn classify_red_as_lava() {
        let c = Color { r: 220, g: 50, b: 20, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_LAVA);
        assert_eq!(m.phase, PHASE_LIQUID);
        assert!((m.temperature - 2000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn classify_brown_as_wood() {
        let c = Color { r: 140, g: 80, b: 30, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_WOOD);
        assert_eq!(m.phase, PHASE_SOLID);
    }

    #[test]
    fn classify_white_as_ice() {
        let c = Color { r: 240, g: 240, b: 240, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_ICE);
        assert_eq!(m.phase, PHASE_SOLID);
        assert!((m.temperature - (-50.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn classify_gray_as_stone() {
        let c = Color { r: 150, g: 150, b: 150, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_STONE);
        assert_eq!(m.phase, PHASE_SOLID);
    }

    #[test]
    fn classify_unknown_defaults_to_stone() {
        // Bright green - doesn't match any specific rule well
        let c = Color { r: 50, g: 200, b: 50, a: 255 };
        let m = classify_color(&c);
        assert_eq!(m.material_id, MAT_STONE);
    }

    #[test]
    fn color_range_contains() {
        let range = ColorRange { r_min: 100, r_max: 200, g_min: 50, g_max: 150, b_min: 0, b_max: 80 };
        assert!(range.contains(&Color { r: 150, g: 100, b: 40, a: 255 }));
        assert!(!range.contains(&Color { r: 50, g: 100, b: 40, a: 255 })); // r too low
        assert!(!range.contains(&Color { r: 150, g: 200, b: 40, a: 255 })); // g too high
    }

    #[test]
    fn palette_match_uses_first_hit() {
        let palette = default_palette_mapping();
        // Dark colour should match gunpowder (first rule)
        let m = find_palette_match(Color { r: 20, g: 20, b: 20, a: 255 }, &palette);
        assert_eq!(m.material_id, MAT_GUNPOWDER);
    }

    #[test]
    fn palette_match_falls_back_to_classify() {
        // Empty palette -> falls back to classify_color
        let m = find_palette_match(Color { r: 220, g: 50, b: 20, a: 255 }, &[]);
        assert_eq!(m.material_id, MAT_LAVA);
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let palette = default_palette_mapping();
        let result = load_vox_model("nonexistent.vox", Vec3::ZERO, &palette);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ContentError::VoxLoad(_)));
    }
}
