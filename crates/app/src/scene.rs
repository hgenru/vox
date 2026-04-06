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

/// Build the material palette for CA simulation (--sim2 mode).
///
/// Maps to CA material IDs: 1=stone, 2=sand, 3=water, 4=lava, 6=ice.
pub(crate) fn ca_material_palette() -> Vec<MaterialSlot> {
    vec![
        MaterialSlot {
            name: "Stone",
            mat_id: 1,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
        },
        MaterialSlot {
            name: "Water",
            mat_id: 3,
            phase: PHASE_LIQUID,
            temperature: 0.0,
            color: glam::Vec4::new(0.2, 0.4, 0.9, 1.0),
        },
        MaterialSlot {
            name: "Lava",
            mat_id: 4,
            phase: PHASE_LIQUID,
            temperature: 2000.0,
            color: glam::Vec4::new(1.0, 0.3, 0.0, 1.0),
        },
        MaterialSlot {
            name: "Sand",
            mat_id: 2,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.76, 0.70, 0.50, 1.0),
        },
        MaterialSlot {
            name: "Ice",
            mat_id: 6,
            phase: PHASE_SOLID,
            temperature: -50.0,
            color: glam::Vec4::new(0.7, 0.85, 0.95, 1.0),
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
    generate_heightmap_with(island_height)
}

/// Generate the CPU-side heightmap for the mountain range scene.
pub(crate) fn generate_mountain_heightmap() -> Vec<f32> {
    generate_heightmap_with(mountain_height)
}

/// Generate a heightmap using the given height function.
fn generate_heightmap_with(height_fn: fn(f32, f32) -> f32) -> Vec<f32> {
    let size = GRID_SIZE;
    let mut heightmap = vec![0.0f32; (size * size) as usize];
    for z in 0..size {
        for x in 0..size {
            heightmap[(z * size + x) as usize] = height_fn(x as f32 + 0.5, z as f32 + 0.5);
        }
    }
    heightmap
}

/// Compute a Gaussian peak contribution at (x, z) centered at (px, pz).
fn gaussian_peak(x: f32, z: f32, px: f32, pz: f32, radius: f32, height: f32) -> f32 {
    let dx = x - px;
    let dz = z - pz;
    let dist_sq = dx * dx + dz * dz;
    height * (-dist_sq / (2.0 * radius * radius)).exp()
}

/// Compute terrain height for the mountain range scene at (x, z).
///
/// Features multiple peaks of varied heights, rolling hills, and valleys.
/// Coordinates scale with GRID_SIZE.
pub(crate) fn mountain_height(x: f32, z: f32) -> f32 {
    let gs = GRID_SIZE as f32;
    let cx = gs * 0.5;
    let cz = gs * 0.5;

    let base = MARGIN as f32;

    // Boundary falloff — keep particles away from grid edges
    let edge_margin = gs * 0.04;
    let fx = if x < edge_margin {
        x / edge_margin
    } else if x > gs - edge_margin {
        (gs - x) / edge_margin
    } else {
        1.0
    };
    let fz = if z < edge_margin {
        z / edge_margin
    } else if z > gs - edge_margin {
        (gs - z) / edge_margin
    } else {
        1.0
    };
    let edge_falloff = (fx * fz).max(0.0).min(1.0);

    // Multiple mountain peaks at different positions — wider and taller
    let peak1 = gaussian_peak(x, z, cx - 25.0, cz - 15.0, 50.0, gs * 0.60);
    let peak2 = gaussian_peak(x, z, cx + 40.0, cz + 30.0, 45.0, gs * 0.52);
    let peak3 = gaussian_peak(x, z, cx, cz + 50.0, 55.0, gs * 0.72);
    let peak4 = gaussian_peak(x, z, cx - 45.0, cz + 35.0, 40.0, gs * 0.45);
    let peak5 = gaussian_peak(x, z, cx + 15.0, cz - 40.0, 50.0, gs * 0.55);
    let peak6 = gaussian_peak(x, z, cx + 55.0, cz - 15.0, 35.0, gs * 0.40);
    let peak7 = gaussian_peak(x, z, cx - 15.0, cz - 50.0, 40.0, gs * 0.42);

    // Multi-octave rolling hills for base terrain — higher base
    let n1 = (x * 0.04).sin() * (z * 0.05).cos() * gs * 0.05;
    let n2 = (x * 0.09 + 1.7).sin() * (z * 0.08 + 2.3).cos() * gs * 0.03;
    let n3 = (x * 0.21 + 0.5).cos() * (z * 0.19 + 1.1).sin() * gs * 0.015;
    let n4 = (x * 0.37 + 3.1).sin() * (z * 0.41 + 0.7).cos() * gs * 0.008;
    let hills = n1 + n2 + n3 + n4;

    let peaks = peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7;
    let terrain = hills + peaks;

    // Clamp height to leave room at top of grid
    let max_height = gs * 0.88;
    let h = base + (gs * 0.06 + terrain.max(0.0)) * edge_falloff;
    h.min(max_height)
}

/// Create the initial set of particles for the mountain range scene.
///
/// Features multiple mountain peaks, valleys with water pools, a lava river,
/// and a cave tunnel through one mountain. With 4M max particles, the terrain
/// shell is thicker and water pools are deeper for a denser world.
pub(crate) fn create_mountain_particles() -> Vec<Particle> {
    let mut particles = Vec::new();
    let gs = GRID_SIZE;
    let margin = MARGIN;
    let upper = gs - margin;

    // 1. Thin floor
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

    // 2. Terrain shell — thicker shell (16 layers) for dense mountains
    let shell = 16_i32;
    let gs_f = gs as f32;
    let cx = gs_f * 0.5;
    let cz = gs_f * 0.5;
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let height = mountain_height(fx, fz);
            let max_y = (height.ceil() as i32).min(upper as i32);
            let min_y = (max_y - shell).max(margin as i32 + FLOOR_THICKNESS);

            // Cave tunnel through peak3 (the tallest mountain at cx, cz+55)
            let cave_cx = cx;
            let cave_cz = cz + 55.0;
            let cave_dx = fx - cave_cx;
            let cave_dz = fz - cave_cz;
            // Tunnel runs along X axis, cylindrical hole
            let cave_radius = 6.0;
            let cave_y_center = MARGIN as f32 + gs_f * 0.15;
            let is_in_cave = cave_dz.abs() < cave_radius
                && cave_dx.abs() < gs_f * 0.15;
            let cave_dist_sq = cave_dz * cave_dz;

            for y in min_y..max_y {
                let fy = y as f32 + 0.5;

                // Carve cave tunnel
                if is_in_cave {
                    let dy = fy - cave_y_center;
                    if cave_dist_sq + dy * dy < cave_radius * cave_radius {
                        continue; // skip this voxel — it's inside the cave
                    }
                }

                particles.push(Particle::new(
                    Vec3::new(fx, fy, fz),
                    1.0, MAT_STONE, PHASE_SOLID,
                ));
            }
        }
    }

    // 3. Water pools in valleys — deeper (3 layers, 8 PPC) for denser world
    let water_level = (gs_f * 0.18) as i32; // valleys fill up to ~18% of grid height
    let valley_regions: [(f32, f32, f32); 4] = [
        (cx + 10.0, cz + 10.0, 45.0),  // valley between peaks 1,2,3 (bigger)
        (cx - 40.0, cz - 30.0, 30.0),  // pool near peak1 (bigger)
        (cx + 50.0, cz - 20.0, 25.0),  // pool near peak5 (bigger)
        (cx - 10.0, cz - 45.0, 20.0),  // new pool near peak7
    ];
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let terrain_h = mountain_height(fx, fz);

            // Only place water if in a valley region
            let in_valley = valley_regions.iter().any(|(vx, vz, vr)| {
                let dx = fx - vx;
                let dz = fz - vz;
                dx * dx + dz * dz < vr * vr
            });
            if !in_valley { continue; }

            if terrain_h < water_level as f32 {
                let top = water_level.min(upper as i32);
                let bottom = (top - 3).max(terrain_h.ceil() as i32); // 3 layers deep
                for y in bottom..top {
                    spawn_cell(&mut particles, fx - 0.5, y as f32, fz - 0.5,
                        0.125, MAT_WATER, PHASE_LIQUID);
                }
            }
        }
    }

    // 4. Lava river in a valley between peak4 and peak3 — wider
    let lava_segments: [(f32, f32, f32, f32); 4] = [
        (cx - 50.0, cz + 40.0, cx - 35.0, cz + 45.0),
        (cx - 35.0, cz + 45.0, cx - 20.0, cz + 50.0),
        (cx - 20.0, cz + 50.0, cx - 5.0, cz + 53.0),
        (cx - 5.0, cz + 53.0, cx + 10.0, cz + 55.0),
    ];
    let river_width = 6.0_f32; // wider river
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;

            // Check distance to any river segment
            let near_river = lava_segments.iter().any(|&(x1, z1, x2, z2)| {
                point_segment_dist_sq(fx, fz, x1, z1, x2, z2) < river_width * river_width
            });
            if !near_river { continue; }

            let terrain_h = mountain_height(fx, fz);
            // Lava sits on top of terrain, 1 layer
            let lava_y = terrain_h.ceil() as i32;
            if lava_y >= upper as i32 { continue; }

            spawn_cell(&mut particles, fx - 0.5, lava_y as f32, fz - 0.5,
                0.125, MAT_LAVA, PHASE_LIQUID);
            // Set temperature on the 8 lava particles just spawned
            let start = particles.len() - 8;
            for p in &mut particles[start..] {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, 2000.0);
            }
        }
    }

    // Log material counts
    let stone_count = particles.iter().filter(|p| p.material_id() == MAT_STONE).count();
    let water_count = particles.iter().filter(|p| p.material_id() == MAT_WATER).count();
    let lava_count = particles.iter().filter(|p| p.material_id() == MAT_LAVA).count();
    tracing::info!(
        "Mountain scene: {} stone + {} water + {} lava = {} total",
        stone_count, water_count, lava_count, particles.len(),
    );

    assert!(
        particles.len() < shared::MAX_PARTICLES as usize,
        "Mountain scene has {} particles, max is {}",
        particles.len(),
        shared::MAX_PARTICLES
    );

    particles
}

/// Squared distance from point (px, pz) to line segment (x1,z1)-(x2,z2).
fn point_segment_dist_sq(px: f32, pz: f32, x1: f32, z1: f32, x2: f32, z2: f32) -> f32 {
    let dx = x2 - x1;
    let dz = z2 - z1;
    let len_sq = dx * dx + dz * dz;
    if len_sq < 1e-6 {
        let ex = px - x1;
        let ez = pz - z1;
        return ex * ex + ez * ez;
    }
    let t = ((px - x1) * dx + (pz - z1) * dz) / len_sq;
    let t = t.max(0.0).min(1.0);
    let proj_x = x1 + t * dx;
    let proj_z = z1 + t * dz;
    let ex = px - proj_x;
    let ez = pz - proj_z;
    ex * ex + ez * ez
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

    #[test]
    fn mountain_particles_under_limit() {
        let particles = create_mountain_particles();
        assert!(
            particles.len() < shared::MAX_PARTICLES as usize,
            "Mountain scene has {} particles, max is {}",
            particles.len(),
            shared::MAX_PARTICLES
        );
        // Dense scene should have >1M particles with thicker shells and deeper pools
        assert!(
            particles.len() > 1_000_000,
            "Mountain scene should have >1M particles, got {}",
            particles.len()
        );
    }

    #[test]
    fn mountain_heightmap_values_valid() {
        let hm = generate_mountain_heightmap();
        assert_eq!(hm.len(), (GRID_SIZE * GRID_SIZE) as usize);
        for h in &hm {
            assert!(h.is_finite(), "NaN in mountain heightmap");
            assert!(*h >= 0.0 && *h < GRID_SIZE as f32, "Height {} out of bounds", h);
        }
    }

    #[test]
    fn mountain_has_varied_terrain() {
        let hm = generate_mountain_heightmap();
        let min_h = hm.iter().cloned().fold(f32::MAX, f32::min);
        let max_h = hm.iter().cloned().fold(f32::MIN, f32::max);
        let range = max_h - min_h;
        // Should have significant height variation (at least 30% of grid)
        assert!(
            range > GRID_SIZE as f32 * 0.3,
            "Height range {:.1} is too small (min={:.1}, max={:.1})",
            range, min_h, max_h
        );
    }
}
