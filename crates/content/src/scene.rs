//! RON-deserializable scene description types and particle spawning.
//!
//! A [`SceneDef`] describes a scene as a list of geometric primitives
//! (boxes, spheres, layers) filled with particles. Each primitive specifies
//! a material, phase, temperature, and particle density.
//!
//! Scene files live in `assets/scenes/*.ron` and are loaded at startup
//! via the `--scene` CLI flag.

use glam::Vec3;
use serde::Deserialize;
use shared::{GRID_SIZE, Particle};

/// Top-level scene definition loaded from a RON file.
#[derive(Deserialize, Debug)]
pub struct SceneDef {
    /// Human-readable scene name (for logging).
    pub name: String,
    /// Objects to spawn as particles.
    pub objects: Vec<SceneObject>,
    /// Optional camera placement hint. The app may override this.
    pub camera: Option<CameraDef>,
}

/// Camera placement hint within a scene file.
#[derive(Deserialize, Debug, Clone)]
pub struct CameraDef {
    /// Camera eye (position) in grid units.
    pub eye: (f32, f32, f32),
    /// Camera look-at target in grid units.
    pub target: (f32, f32, f32),
}

/// A geometric primitive that gets filled with particles during spawning.
#[derive(Deserialize, Debug)]
pub enum SceneObject {
    /// Axis-aligned box filled with particles. Coordinates in grid units.
    Box {
        /// Material ID (see `shared::material` constants).
        material: u32,
        /// Phase: 0 = solid, 1 = liquid, 2 = gas.
        phase: u32,
        /// Minimum corner (inclusive).
        min: (f32, f32, f32),
        /// Maximum corner (exclusive).
        max: (f32, f32, f32),
        /// Initial temperature in degrees. Default: 20.0.
        #[serde(default = "default_temp")]
        temperature: f32,
        /// Particles per cell. 1 for solids, 8 for liquids/gas (2x2x2 sub-grid). Default: 1.
        #[serde(default = "default_ppc")]
        ppc: u32,
    },
    /// Hollow box (outer minus inner). Good for walls and containers.
    HollowBox {
        /// Material ID.
        material: u32,
        /// Phase.
        phase: u32,
        /// Outer box minimum corner.
        outer_min: (f32, f32, f32),
        /// Outer box maximum corner.
        outer_max: (f32, f32, f32),
        /// Inner box minimum corner (hollow region).
        inner_min: (f32, f32, f32),
        /// Inner box maximum corner (hollow region).
        inner_max: (f32, f32, f32),
        /// Initial temperature. Default: 20.0.
        #[serde(default = "default_temp")]
        temperature: f32,
        /// Particles per cell. Default: 1.
        #[serde(default = "default_ppc")]
        ppc: u32,
    },
    /// Sphere filled with particles.
    Sphere {
        /// Material ID.
        material: u32,
        /// Phase.
        phase: u32,
        /// Center in grid units.
        center: (f32, f32, f32),
        /// Radius in grid units.
        radius: f32,
        /// Initial temperature. Default: 20.0.
        #[serde(default = "default_temp")]
        temperature: f32,
        /// Particles per cell. Default: 1.
        #[serde(default = "default_ppc")]
        ppc: u32,
    },
    /// Full-width horizontal layer spanning the entire grid in X and Z.
    Layer {
        /// Material ID.
        material: u32,
        /// Phase.
        phase: u32,
        /// Minimum Y coordinate.
        y_min: f32,
        /// Maximum Y coordinate.
        y_max: f32,
        /// Initial temperature. Default: 20.0.
        #[serde(default = "default_temp")]
        temperature: f32,
        /// Particles per cell. Default: 1.
        #[serde(default = "default_ppc")]
        ppc: u32,
    },
}

fn default_temp() -> f32 {
    20.0
}
fn default_ppc() -> u32 {
    1
}

impl SceneDef {
    /// Spawn all particles described by this scene definition.
    ///
    /// Iterates over every [`SceneObject`] and generates particles within
    /// the grid bounds (`GRID_SIZE`). Particles are created with the specified
    /// material, phase, and temperature. When `ppc > 1`, a 2x2x2 sub-grid
    /// pattern is used (8 particles per cell with mass = 1.0/8).
    ///
    /// Returns the full list of spawned particles.
    pub fn spawn_particles(&self) -> Vec<Particle> {
        let mut particles = Vec::new();
        for obj in &self.objects {
            match obj {
                SceneObject::Box {
                    material, phase, min, max, temperature, ppc,
                } => {
                    spawn_box(
                        &mut particles,
                        *material, *phase, *min, *max, *temperature, *ppc,
                    );
                }
                SceneObject::HollowBox {
                    material, phase,
                    outer_min, outer_max, inner_min, inner_max,
                    temperature, ppc,
                } => {
                    spawn_hollow_box(
                        &mut particles,
                        *material, *phase,
                        *outer_min, *outer_max, *inner_min, *inner_max,
                        *temperature, *ppc,
                    );
                }
                SceneObject::Sphere {
                    material, phase, center, radius, temperature, ppc,
                } => {
                    spawn_sphere(
                        &mut particles,
                        *material, *phase, *center, *radius, *temperature, *ppc,
                    );
                }
                SceneObject::Layer {
                    material, phase, y_min, y_max, temperature, ppc,
                } => {
                    spawn_layer(
                        &mut particles,
                        *material, *phase, *y_min, *y_max, *temperature, *ppc,
                    );
                }
            }
        }
        tracing::info!(
            "Scene '{}': spawned {} particles from {} objects",
            self.name,
            particles.len(),
            self.objects.len(),
        );
        particles
    }
}

/// Clamp a coordinate range to valid grid bounds [2, GRID_SIZE-2).
fn clamp_range(lo: f32, hi: f32) -> (i32, i32) {
    let margin = 2;
    let grid_max = GRID_SIZE as i32 - margin;
    let lo = (lo.floor() as i32).max(margin);
    let hi = (hi.ceil() as i32).min(grid_max);
    (lo, hi)
}

/// Spawn a single particle or a 2x2x2 sub-grid cell at integer grid position (x, y, z).
fn spawn_at_cell(
    particles: &mut Vec<Particle>,
    x: i32, y: i32, z: i32,
    material: u32, phase: u32, temperature: f32, ppc: u32,
) {
    if ppc <= 1 {
        // Single particle at cell center
        let mut p = Particle::new(
            Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
            1.0,
            material,
            phase,
        );
        if temperature != 20.0 {
            p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, temperature);
        }
        particles.push(p);
    } else {
        // 2x2x2 sub-grid (8 particles per cell)
        let mass = 1.0 / 8.0;
        for dx in 0..2u32 {
            for dy in 0..2u32 {
                for dz in 0..2u32 {
                    let pos = Vec3::new(
                        x as f32 + 0.25 + dx as f32 * 0.5,
                        y as f32 + 0.25 + dy as f32 * 0.5,
                        z as f32 + 0.25 + dz as f32 * 0.5,
                    );
                    let mut p = Particle::new(pos, mass, material, phase);
                    if temperature != 20.0 {
                        p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, temperature);
                    }
                    particles.push(p);
                }
            }
        }
    }
}

/// Fill an axis-aligned box region with particles.
fn spawn_box(
    particles: &mut Vec<Particle>,
    material: u32, phase: u32,
    min: (f32, f32, f32), max: (f32, f32, f32),
    temperature: f32, ppc: u32,
) {
    let (x0, x1) = clamp_range(min.0, max.0);
    let (y0, y1) = clamp_range(min.1, max.1);
    let (z0, z1) = clamp_range(min.2, max.2);
    for x in x0..x1 {
        for y in y0..y1 {
            for z in z0..z1 {
                spawn_at_cell(particles, x, y, z, material, phase, temperature, ppc);
            }
        }
    }
}

/// Fill a hollow box (outer minus inner) with particles.
fn spawn_hollow_box(
    particles: &mut Vec<Particle>,
    material: u32, phase: u32,
    outer_min: (f32, f32, f32), outer_max: (f32, f32, f32),
    inner_min: (f32, f32, f32), inner_max: (f32, f32, f32),
    temperature: f32, ppc: u32,
) {
    let (ox0, ox1) = clamp_range(outer_min.0, outer_max.0);
    let (oy0, oy1) = clamp_range(outer_min.1, outer_max.1);
    let (oz0, oz1) = clamp_range(outer_min.2, outer_max.2);
    let ix0 = inner_min.0.floor() as i32;
    let ix1 = inner_max.0.ceil() as i32;
    let iy0 = inner_min.1.floor() as i32;
    let iy1 = inner_max.1.ceil() as i32;
    let iz0 = inner_min.2.floor() as i32;
    let iz1 = inner_max.2.ceil() as i32;

    for x in ox0..ox1 {
        for y in oy0..oy1 {
            for z in oz0..oz1 {
                // Skip if inside the inner box
                if x >= ix0 && x < ix1 && y >= iy0 && y < iy1 && z >= iz0 && z < iz1 {
                    continue;
                }
                spawn_at_cell(particles, x, y, z, material, phase, temperature, ppc);
            }
        }
    }
}

/// Fill a sphere with particles.
fn spawn_sphere(
    particles: &mut Vec<Particle>,
    material: u32, phase: u32,
    center: (f32, f32, f32), radius: f32,
    temperature: f32, ppc: u32,
) {
    let r2 = radius * radius;
    let (x0, x1) = clamp_range(center.0 - radius, center.0 + radius);
    let (y0, y1) = clamp_range(center.1 - radius, center.1 + radius);
    let (z0, z1) = clamp_range(center.2 - radius, center.2 + radius);

    for x in x0..x1 {
        for y in y0..y1 {
            for z in z0..z1 {
                let cx = x as f32 + 0.5 - center.0;
                let cy = y as f32 + 0.5 - center.1;
                let cz = z as f32 + 0.5 - center.2;
                if cx * cx + cy * cy + cz * cz <= r2 {
                    spawn_at_cell(particles, x, y, z, material, phase, temperature, ppc);
                }
            }
        }
    }
}

/// Fill a full-width horizontal layer (spans entire grid in X and Z).
fn spawn_layer(
    particles: &mut Vec<Particle>,
    material: u32, phase: u32,
    y_min: f32, y_max: f32,
    temperature: f32, ppc: u32,
) {
    let margin = 2;
    let grid_max = GRID_SIZE as i32 - margin;
    let (y0, y1) = clamp_range(y_min, y_max);

    for x in margin..grid_max {
        for y in y0..y1 {
            for z in margin..grid_max {
                spawn_at_cell(particles, x, y, z, material, phase, temperature, ppc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_demo_scene() {
        let ron_str = include_str!("../../../assets/scenes/lava_basin.ron");
        let scene: SceneDef = ron::from_str(ron_str).expect("Failed to parse lava_basin.ron");
        assert_eq!(scene.name, "Lava Basin");
        assert!(!scene.objects.is_empty());
        let particles = scene.spawn_particles();
        assert!(particles.len() > 0, "Scene should produce particles");
        assert!(
            (particles.len() as u32) < shared::MAX_PARTICLES,
            "Scene has {} particles, exceeds MAX_PARTICLES {}",
            particles.len(),
            shared::MAX_PARTICLES,
        );
    }

    #[test]
    fn parse_stress_test_scene() {
        let ron_str = include_str!("../../../assets/scenes/stress_test.ron");
        let scene: SceneDef = ron::from_str(ron_str).expect("Failed to parse stress_test.ron");
        assert!(!scene.name.is_empty());
        let particles = scene.spawn_particles();
        assert!(particles.len() > 0);
    }

    #[test]
    fn box_spawns_correct_particles() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Box {
                material: 0,
                phase: 0,
                min: (10.0, 10.0, 10.0),
                max: (12.0, 12.0, 12.0),
                temperature: 20.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        // 2x2x2 = 8 cells, 1 particle each
        assert_eq!(particles.len(), 8);
        for p in &particles {
            assert_eq!(p.material_id(), 0);
            assert_eq!(p.phase(), 0);
        }
    }

    #[test]
    fn box_ppc8_spawns_8x() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Box {
                material: 1,
                phase: 1,
                min: (10.0, 10.0, 10.0),
                max: (11.0, 11.0, 11.0),
                temperature: 20.0,
                ppc: 8,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        // 1 cell x 8 particles
        assert_eq!(particles.len(), 8);
        for p in &particles {
            assert!((p.pos_mass.w - 0.125).abs() < 1e-6, "mass should be 1/8");
        }
    }

    #[test]
    fn hollow_box_has_hollow_interior() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::HollowBox {
                material: 0,
                phase: 0,
                outer_min: (10.0, 10.0, 10.0),
                outer_max: (20.0, 20.0, 20.0),
                inner_min: (12.0, 12.0, 12.0),
                inner_max: (18.0, 18.0, 18.0),
                temperature: 20.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        let outer_count = 10 * 10 * 10; // 1000
        let inner_count = 6 * 6 * 6; // 216
        assert_eq!(particles.len(), outer_count - inner_count);
    }

    #[test]
    fn sphere_spawns_within_radius() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Sphere {
                material: 0,
                phase: 0,
                center: (32.0, 32.0, 32.0),
                radius: 5.0,
                temperature: 20.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        assert!(particles.len() > 0);
        for p in &particles {
            let pos = p.position();
            let dx = pos.x - 32.0;
            let dy = pos.y - 32.0;
            let dz = pos.z - 32.0;
            // Cell center is at +0.5, so allow slightly more than radius
            assert!(
                dx * dx + dy * dy + dz * dz <= (5.5 * 5.5),
                "Particle at {:?} outside sphere",
                pos,
            );
        }
    }

    #[test]
    fn temperature_is_applied() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Box {
                material: 2,
                phase: 1,
                min: (10.0, 10.0, 10.0),
                max: (11.0, 11.0, 11.0),
                temperature: 2000.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        assert_eq!(particles.len(), 1);
        assert!((particles[0].vel_temp.w - 2000.0).abs() < 1e-3);
    }

    #[test]
    fn layer_spans_full_grid() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Layer {
                material: 0,
                phase: 0,
                y_min: 2.0,
                y_max: 3.0,
                temperature: 20.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        let side = GRID_SIZE as i32 - 4; // margin=2 on each side
        assert_eq!(particles.len() as i32, side * 1 * side);
    }

    #[test]
    fn default_temperature_not_stored() {
        let scene = SceneDef {
            name: "test".to_string(),
            objects: vec![SceneObject::Box {
                material: 0,
                phase: 0,
                min: (10.0, 10.0, 10.0),
                max: (11.0, 11.0, 11.0),
                temperature: 20.0,
                ppc: 1,
            }],
            camera: None,
        };
        let particles = scene.spawn_particles();
        // Default temp 20.0 is special: vel_temp.w stays 0.0 (Particle::new default)
        // Actually, we set it when temp != 20.0. For temp == 20.0, vel_temp.w = 0.0
        assert_eq!(particles[0].vel_temp.w, 0.0);
    }
}
