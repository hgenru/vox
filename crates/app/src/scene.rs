//! Scene generation for the island demo.
//!
//! Contains terrain heightmap generation, particle spawning helpers,
//! material palette definitions, and all scene-related constants.

use glam::Vec3;
use shared::{
    GRID_SIZE, MAT_ASH, MAT_GUNPOWDER, MAT_ICE, MAT_LAVA, MAT_STONE, MAT_WATER, MAT_WOOD,
    PHASE_LIQUID, PHASE_SOLID, Particle,
};

/// Water level scales with grid: ~12% of grid height.
pub(crate) const WATER_LEVEL_FRAC: f32 = 0.12;
pub(crate) const MARGIN: u32 = 2;
pub(crate) const SHELL_THICKNESS: i32 = 3;
pub(crate) const FLOOR_THICKNESS: i32 = 2;

/// A toolbar slot binding a display name, material id, phase, default
/// temperature, and UI colour together.
pub(crate) struct MaterialSlot {
    pub name: &'static str,
    pub mat_id: u32,
    pub phase: u32,
    pub temperature: f32,
    pub color: glam::Vec4,
}

/// Build the full material palette available on the toolbar.
pub(crate) fn material_palette() -> Vec<MaterialSlot> {
    vec![
        MaterialSlot {
            name: "Stone",
            mat_id: MAT_STONE,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
        },
        MaterialSlot {
            name: "Water",
            mat_id: MAT_WATER,
            phase: PHASE_LIQUID,
            temperature: 0.0,
            color: glam::Vec4::new(0.2, 0.4, 0.9, 1.0),
        },
        MaterialSlot {
            name: "Lava",
            mat_id: MAT_LAVA,
            phase: PHASE_LIQUID,
            temperature: 2000.0,
            color: glam::Vec4::new(1.0, 0.3, 0.0, 1.0),
        },
        MaterialSlot {
            name: "Wood",
            mat_id: MAT_WOOD,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.55, 0.35, 0.15, 1.0),
        },
        MaterialSlot {
            name: "Ash",
            mat_id: MAT_ASH,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.4, 0.4, 0.4, 1.0),
        },
        MaterialSlot {
            name: "Ice",
            mat_id: MAT_ICE,
            phase: PHASE_SOLID,
            temperature: -50.0,
            color: glam::Vec4::new(0.7, 0.85, 0.95, 1.0),
        },
        MaterialSlot {
            name: "Gunpowder",
            mat_id: MAT_GUNPOWDER,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.25, 0.2, 0.15, 1.0),
        },
    ]
}

/// Compute terrain height at a given (x, z) world position.
///
/// Uses multi-octave sine-based noise combined with island falloff to produce
/// natural-looking terrain with rolling hills, a central ridge/plateau, valleys,
/// and gentle beach slopes. All dimensions scale with GRID_SIZE.
pub(crate) fn island_height(x: f32, z: f32) -> f32 {
    let gs = GRID_SIZE as f32;
    let cx = gs * 0.5;
    let cz = gs * 0.5;
    let dx = x - cx;
    let dz = z - cz;
    let dist = (dx * dx + dz * dz).sqrt();

    let base = MARGIN as f32;
    let island_radius = gs * 0.38;

    // Island falloff: smooth drop to ocean at edges
    if dist > island_radius {
        return base;
    }
    let edge_t = dist / island_radius;
    let falloff = (1.0 - edge_t * edge_t).max(0.0);

    // Multi-octave terrain noise (no external deps, just sin/cos)
    let n1 = (x * 0.05).sin() * (z * 0.07).cos() * gs * 0.06; // big rolling hills
    let n2 = (x * 0.13 + 1.7).sin() * (z * 0.11 + 2.3).cos() * gs * 0.03; // medium features
    let n3 = (x * 0.31 + 0.5).cos() * (z * 0.29 + 1.1).sin() * gs * 0.015; // small details

    // Ridge/plateau at center instead of cone peak
    let center_dist = dist / (gs * 0.12);
    let ridge = if center_dist < 1.0 {
        (1.0 - center_dist * center_dist) * gs * 0.22 // plateau (~56 cells at 256)
    } else {
        0.0
    };

    let terrain = n1 + n2 + n3 + ridge;
    base + falloff * (gs * 0.12 + terrain.max(0.0))
}

/// Spawn 8 particles (2x2x2 sub-grid) within a single cell.
pub(crate) fn spawn_cell(
    particles: &mut Vec<Particle>,
    x: f32, y: f32, z: f32,
    mass_per_particle: f32, mat: u32, phase: u32,
) {
    for dx in 0..2u32 {
        for dy in 0..2u32 {
            for dz in 0..2u32 {
                let pos = Vec3::new(
                    x + 0.25 + dx as f32 * 0.5,
                    y + 0.25 + dy as f32 * 0.5,
                    z + 0.25 + dz as f32 * 0.5,
                );
                particles.push(Particle::new(pos, mass_per_particle, mat, phase));
            }
        }
    }
}

/// Create the initial set of particles for the island demo scene.
///
/// All ranges scale with GRID_SIZE so the scene works at any resolution.
pub(crate) fn create_island_particles() -> Vec<Particle> {
    let mut particles = Vec::new();
    let gs = GRID_SIZE;
    let margin = MARGIN;
    let upper = gs - margin;
    let water_level = (gs as f32 * WATER_LEVEL_FRAC) as i32;

    // 1a. Thin floor at the bottom
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            for y in (margin as i32)..(margin as i32 + FLOOR_THICKNESS).min(upper as i32) {
                particles.push(Particle::new(
                    Vec3::new(fx, y as f32 + 0.5, fz),
                    1.0, MAT_STONE, PHASE_SOLID,
                ));
            }
        }
    }

    // 1b. Terrain shell: only top SHELL_THICKNESS layers
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let height = island_height(fx, fz);
            let max_y = (height.ceil() as i32).min(upper as i32);
            let min_y = (max_y - SHELL_THICKNESS).max(margin as i32 + FLOOR_THICKNESS);
            for y in min_y..max_y {
                particles.push(Particle::new(
                    Vec3::new(fx, y as f32 + 0.5, fz),
                    1.0, MAT_STONE, PHASE_SOLID,
                ));
            }
        }
    }

    // 2. Ocean water -- shoreline band, 1 layer deep, 8 PPC for proper MPM flow
    let cx = gs as f32 * 0.5;
    let cz = gs as f32 * 0.5;
    let island_radius = gs as f32 * 0.375;
    let water_inner = island_radius * 0.6;  // inner edge of water band
    let water_outer = island_radius * 1.3;  // outer edge of water band
    for x in margin..(gs - margin) {
        for z in margin..(gs - margin) {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let dx = fx - cx;
            let dz = fz - cz;
            let dist = (dx * dx + dz * dz).sqrt();
            // Only generate water in a band around the island
            if dist < water_inner || dist > water_outer {
                continue;
            }
            let terrain_h = island_height(fx, fz);
            if terrain_h < water_level as f32 {
                // Only spawn the top 1 layer of water (8 PPC for proper MPM fluid behavior)
                let water_top = water_level;
                let water_bottom = (water_top - 1).max(terrain_h.ceil() as i32);
                for y in water_bottom..water_top {
                    spawn_cell(&mut particles, fx - 0.5, y as f32, fz - 0.5, 0.125, MAT_WATER, PHASE_LIQUID);
                }
            }
        }
    }

    // 3. Lava pool at the volcano summit -- 8 PPC, max 2 layers for proper MPM flow
    let lava_min = (gs as f32 * 0.44) as u32;
    let lava_max = (gs as f32 * 0.56) as u32;
    let max_lava_layers: i32 = 2;
    for x in lava_min..lava_max {
        for z in lava_min..lava_max {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let terrain_h = island_height(fx, fz);
            let lava_top = (terrain_h + gs as f32 * 0.03).min((gs - margin) as f32) as i32;
            let lava_bottom = (lava_top - max_lava_layers).max(terrain_h.ceil() as i32);
            for y in lava_bottom..lava_top {
                spawn_cell(&mut particles, fx - 0.5, y as f32, fz - 0.5, 0.125, MAT_LAVA, PHASE_LIQUID);
            }
            // Set temperature on newly spawned lava particles
            let lava_count = ((lava_top - lava_bottom).max(0) as usize) * 8;
            let start = particles.len() - lava_count;
            for p in &mut particles[start..] {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, 2000.0);
            }
        }
    }

    tracing::info!(
        "Island scene: {} stone + {} water + {} lava = {} total",
        particles.iter().filter(|p| p.material_id() == MAT_STONE).count(),
        particles.iter().filter(|p| p.material_id() == MAT_WATER).count(),
        particles.iter().filter(|p| p.material_id() == MAT_LAVA).count(),
        particles.len(),
    );
    particles
}

/// Generate the CPU-side heightmap for player ground collision.
pub(crate) fn generate_heightmap() -> Vec<f32> {
    let size = GRID_SIZE;
    let mut heightmap = vec![0.0f32; (size * size) as usize];
    for z in 0..size {
        for x in 0..size {
            heightmap[(z * size + x) as usize] = island_height(x as f32 + 0.5, z as f32 + 0.5);
        }
    }
    heightmap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn island_particles_under_limit() {
        let particles = create_island_particles();
        assert!(
            particles.len() < shared::MAX_PARTICLES as usize,
            "Scene has {} particles, max is {}",
            particles.len(),
            shared::MAX_PARTICLES
        );
    }

    #[test]
    fn heightmap_values_valid() {
        let hm = generate_heightmap();
        assert_eq!(hm.len(), (GRID_SIZE * GRID_SIZE) as usize);
        for h in &hm {
            assert!(h.is_finite(), "NaN in heightmap");
            assert!(*h >= 0.0 && *h < GRID_SIZE as f32, "Height {} out of bounds", h);
        }
    }
}
