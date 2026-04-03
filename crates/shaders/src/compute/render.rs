//! Render compute shader: ray-marches through voxel grid and writes pixels.
//!
//! Dispatched with (ceil(width/8), ceil(height/8), 1) workgroups.
//! Each thread computes one pixel via DDA ray marching through the 3D voxel grid.

use spirv_std::glam::{UVec3, UVec4, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

/// Push constants for the render shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RenderPushConstants {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Padding.
    pub _pad: u32,
    /// Camera eye position (xyz) + padding (w).
    pub eye: Vec4,
    /// Camera target position (xyz) + padding (w).
    pub target: Vec4,
}

/// Normalize a Vec3 without using the `length()` method (more portable for SPIR-V).
fn normalize_vec3(v: Vec3) -> Vec3 {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if len_sq < 1e-12 {
        return Vec3::new(0.0, 1.0, 0.0);
    }
    let inv_len = 1.0 / len_sq.sqrt();
    Vec3::new(v.x * inv_len, v.y * inv_len, v.z * inv_len)
}

/// Cross product of two Vec3s.
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Dot product of two Vec3s.
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Compute camera ray direction for a given pixel coordinate.
///
/// Returns a normalized ray direction using a simple perspective camera
/// with vertical FOV of ~60 degrees.
fn compute_ray(
    px: u32,
    py: u32,
    width: u32,
    height: u32,
    eye: Vec3,
    target: Vec3,
) -> Vec3 {
    let w = width as f32;
    let h = height as f32;

    // Normalized device coordinates: [-1, 1]
    let ndc_x = (2.0 * (px as f32 + 0.5) / w) - 1.0;
    let ndc_y = 1.0 - (2.0 * (py as f32 + 0.5) / h);

    // Aspect ratio
    let aspect = w / h;

    // FOV ~60 degrees -> tan(30) ~= 0.577
    let fov_tan = 0.577;

    // Camera basis
    let forward = normalize_vec3(Vec3::new(
        target.x - eye.x,
        target.y - eye.y,
        target.z - eye.z,
    ));
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = normalize_vec3(cross(forward, world_up));
    let up = cross(right, forward);

    // Ray direction in world space
    let dir = Vec3::new(
        forward.x + right.x * ndc_x * aspect * fov_tan + up.x * ndc_y * fov_tan,
        forward.y + right.y * ndc_x * aspect * fov_tan + up.y * ndc_y * fov_tan,
        forward.z + right.z * ndc_x * aspect * fov_tan + up.z * ndc_y * fov_tan,
    );

    normalize_vec3(dir)
}

/// Pack RGBA color components (0-255) into a u32 for B8G8R8A8 swapchain format.
///
/// Byte order: R in lowest byte, then G, B, A (little-endian 0xAABBGGRR).
fn pack_bgra(r: u32, g: u32, b: u32, a: u32) -> u32 {
    // For B8G8R8A8_SRGB on little-endian: bytes are B, G, R, A
    // But vkCmdCopyBufferToImage interprets buffer bytes linearly,
    // so for B8G8R8A8_SRGB we need: byte0=B, byte1=G, byte2=R, byte3=A
    // In u32 little-endian: 0xAARRGGBB
    b | (g << 8) | (r << 16) | (a << 24)
}

/// Simple 3D hash for procedural texturing.
///
/// Returns a deterministic pseudo-random value in [0, 1] for integer coordinates.
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let n = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(z.wrapping_mul(1274126177));
    let n = ((n as u32) >> 13) ^ (n as u32);
    let n = n.wrapping_mul(
        n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303))
            .wrapping_add(1376312589),
    );
    (n >> 16) as f32 / 65535.0
}

/// Smooth value noise via trilinear interpolation of hashed lattice points.
///
/// Input coordinates are continuous; output is in [0, 1] with smooth transitions.
fn value_noise(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let fz = z - iz as f32;
    // Smoothstep for continuous first derivative
    let fx = fx * fx * (3.0 - 2.0 * fx);
    let fy = fy * fy * (3.0 - 2.0 * fy);
    let fz = fz * fz * (3.0 - 2.0 * fz);

    // Trilinear interpolation of 8 corner hashes
    let c000 = hash3(ix, iy, iz);
    let c100 = hash3(ix + 1, iy, iz);
    let c010 = hash3(ix, iy + 1, iz);
    let c110 = hash3(ix + 1, iy + 1, iz);
    let c001 = hash3(ix, iy, iz + 1);
    let c101 = hash3(ix + 1, iy, iz + 1);
    let c011 = hash3(ix, iy + 1, iz + 1);
    let c111 = hash3(ix + 1, iy + 1, iz + 1);

    let c00 = c000 + (c100 - c000) * fx;
    let c10 = c010 + (c110 - c010) * fx;
    let c01 = c001 + (c101 - c001) * fx;
    let c11 = c011 + (c111 - c011) * fx;

    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;

    c0 + (c1 - c0) * fz
}

/// Get procedurally textured color for a material at a given voxel position.
///
/// Each material uses noise/hash functions seeded by voxel coordinates to produce
/// visual variation instead of flat color. Returns (r, g, b) as floats in [0, 1].
fn textured_material_color(material_id: u32, vx: i32, vy: i32, vz: i32, grid_size: u32) -> Vec3 {
    if material_id == 0 {
        // Stone: dark veins through lighter base with per-voxel grain
        let noise = value_noise(vx as f32 * 0.7, vy as f32 * 0.7, vz as f32 * 0.7);
        let noise2 = value_noise(vx as f32 * 2.3, vy as f32 * 2.3, vz as f32 * 2.3);
        let base = Vec3::new(0.5, 0.48, 0.45);
        let vein = Vec3::new(0.35, 0.33, 0.30);
        let t = noise * 0.7 + noise2 * 0.3;
        let inv_t = 1.0 - t;
        let inv_t = if inv_t < 0.0 { 0.0 } else { inv_t };
        let color = Vec3::new(
            base.x + (vein.x - base.x) * inv_t,
            base.y + (vein.y - base.y) * inv_t,
            base.z + (vein.z - base.z) * inv_t,
        );
        // Subtle per-voxel variation
        let h = hash3(vx, vy, vz);
        let offset = (h - 0.5) * 0.06;
        Vec3::new(color.x + offset, color.y + offset, color.z + offset)
    } else if material_id == 1 {
        // Water: depth-based blue variation
        let gs = grid_size as f32;
        let depth_factor = 1.0 - (vy as f32 / gs);
        let depth_factor = if depth_factor < 0.0 { 0.0 } else { depth_factor };
        let base = Vec3::new(0.1, 0.35, 0.85);
        let deep = Vec3::new(0.05, 0.15, 0.5);
        let t = depth_factor * 0.5;
        Vec3::new(
            base.x + (deep.x - base.x) * t,
            base.y + (deep.y - base.y) * t,
            base.z + (deep.z - base.z) * t,
        )
    } else if material_id == 2 {
        // Lava: turbulent bright/dark patches
        let noise = value_noise(vx as f32 * 1.5, vy as f32 * 1.5, vz as f32 * 1.5);
        let bright = Vec3::new(1.0, 0.7, 0.1);
        let dark = Vec3::new(0.8, 0.2, 0.0);
        Vec3::new(
            dark.x + (bright.x - dark.x) * noise,
            dark.y + (bright.y - dark.y) * noise,
            dark.z + (bright.z - dark.z) * noise,
        )
    } else if material_id == 3 {
        // Wood: grain pattern with brown variation
        let noise = value_noise(vx as f32 * 0.5, vy as f32 * 2.0, vz as f32 * 0.5);
        let noise2 = value_noise(vx as f32 * 3.0, vy as f32 * 0.3, vz as f32 * 3.0);
        let light = Vec3::new(0.6, 0.4, 0.2);
        let dark = Vec3::new(0.4, 0.25, 0.1);
        let t = noise * 0.6 + noise2 * 0.4;
        let color = Vec3::new(
            dark.x + (light.x - dark.x) * t,
            dark.y + (light.y - dark.y) * t,
            dark.z + (light.z - dark.z) * t,
        );
        let h = hash3(vx, vy, vz);
        let offset = (h - 0.5) * 0.04;
        Vec3::new(color.x + offset, color.y + offset, color.z + offset)
    } else if material_id == 4 {
        // Ash: gray with subtle variation
        let h = hash3(vx, vy, vz);
        let base = 0.4 + (h - 0.5) * 0.08;
        Vec3::new(base, base, base)
    } else {
        // Unknown: magenta
        Vec3::new(1.0, 0.0, 1.0)
    }
}

/// Sun direction constant used for both diffuse lighting and shadow rays.
const SUN_DIR_RAW: Vec3 = Vec3::new(0.4, 0.8, 0.3);

/// Return the normalized sun direction.
fn sun_direction() -> Vec3 {
    normalize_vec3(SUN_DIR_RAW)
}

/// Compute voxel-based ambient occlusion.
///
/// Samples 8 neighbors (4 edge + 4 corner) in the plane perpendicular to the
/// hit face normal. Each occupied edge neighbor adds 0.12 occlusion, each
/// occupied corner neighbor adds 0.08. Total is clamped to 0.6 max.
/// Returns the AO factor (1.0 - occlusion).
fn compute_ao(
    ix: i32,
    iy: i32,
    iz: i32,
    normal: Vec3,
    voxels: &[UVec4],
    grid_size: u32,
) -> f32 {
    let gs = grid_size as i32;

    // Determine two tangent axes perpendicular to the hit face normal.
    // normal is axis-aligned (one component is +/-1, others 0).
    let (t1x, t1y, t1z, t2x, t2y, t2z) = if normal.x.abs() > 0.5 {
        // Hit X face -> tangent plane is YZ
        (0i32, 1i32, 0i32, 0i32, 0i32, 1i32)
    } else if normal.y.abs() > 0.5 {
        // Hit Y face -> tangent plane is XZ
        (1, 0, 0, 0, 0, 1)
    } else {
        // Hit Z face -> tangent plane is XY
        (1, 0, 0, 0, 1, 0)
    };

    // Offset one step in the normal direction (sample neighbors on the outer face side)
    let nx = if normal.x > 0.5 { 1i32 } else if normal.x < -0.5 { -1i32 } else { 0i32 };
    let ny = if normal.y > 0.5 { 1i32 } else if normal.y < -0.5 { -1i32 } else { 0i32 };
    let nz = if normal.z > 0.5 { 1i32 } else if normal.z < -0.5 { -1i32 } else { 0i32 };

    let mut occlusion = 0.0_f32;

    // 4 edge neighbors (along tangent axes): +t1, -t1, +t2, -t2
    let edge_offsets: [(i32, i32, i32); 4] = [
        (t1x, t1y, t1z),
        (-t1x, -t1y, -t1z),
        (t2x, t2y, t2z),
        (-t2x, -t2y, -t2z),
    ];
    let mut ei = 0u32;
    while ei < 4 {
        let ox = ix + nx + edge_offsets[ei as usize].0;
        let oy = iy + ny + edge_offsets[ei as usize].1;
        let oz = iz + nz + edge_offsets[ei as usize].2;
        if ox >= 0 && ox < gs && oy >= 0 && oy < gs && oz >= 0 && oz < gs {
            let idx = (oz as u32 * grid_size * grid_size + oy as u32 * grid_size + ox as u32) as usize;
            if voxels[idx].w > 0 {
                occlusion += 0.12;
            }
        }
        ei += 1;
    }

    // 4 corner neighbors (diagonal combinations of tangent axes)
    let corner_offsets: [(i32, i32, i32); 4] = [
        (t1x + t2x, t1y + t2y, t1z + t2z),
        (t1x - t2x, t1y - t2y, t1z - t2z),
        (-t1x + t2x, -t1y + t2y, -t1z + t2z),
        (-t1x - t2x, -t1y - t2y, -t1z - t2z),
    ];
    let mut ci = 0u32;
    while ci < 4 {
        let ox = ix + nx + corner_offsets[ci as usize].0;
        let oy = iy + ny + corner_offsets[ci as usize].1;
        let oz = iz + nz + corner_offsets[ci as usize].2;
        if ox >= 0 && ox < gs && oy >= 0 && oy < gs && oz >= 0 && oz < gs {
            let idx = (oz as u32 * grid_size * grid_size + oy as u32 * grid_size + ox as u32) as usize;
            if voxels[idx].w > 0 {
                occlusion += 0.08;
            }
        }
        ci += 1;
    }

    // Clamp max occlusion to 0.6
    if occlusion > 0.6 {
        occlusion = 0.6;
    }

    1.0 - occlusion
}

/// March a shadow ray through the voxel grid using DDA.
///
/// Returns `true` if any occupied voxel is hit between the start position and
/// the grid boundary in the given direction (i.e. the point is in shadow).
fn march_shadow(
    start: Vec3,
    dir: Vec3,
    voxels: &[UVec4],
    grid_size: u32,
) -> bool {
    let gs = grid_size as i32;
    let grid = grid_size as f32;

    // Starting voxel
    let mut vx = start.x as i32;
    let mut vy = start.y as i32;
    let mut vz = start.z as i32;

    // Clamp to grid
    if vx < 0 { vx = 0; }
    if vy < 0 { vy = 0; }
    if vz < 0 { vz = 0; }
    if vx >= gs { vx = gs - 1; }
    if vy >= gs { vy = gs - 1; }
    if vz >= gs { vz = gs - 1; }

    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    let t_delta_x = if dir.x.abs() > 1e-8 { (1.0 / dir.x).abs() } else { 1e20 };
    let t_delta_y = if dir.y.abs() > 1e-8 { (1.0 / dir.y).abs() } else { 1e20 };
    let t_delta_z = if dir.z.abs() > 1e-8 { (1.0 / dir.z).abs() } else { 1e20 };

    let next_x = if step_x > 0 { (vx + 1) as f32 } else { vx as f32 };
    let next_y = if step_y > 0 { (vy + 1) as f32 } else { vy as f32 };
    let next_z = if step_z > 0 { (vz + 1) as f32 } else { vz as f32 };

    let mut t_max_x = if dir.x.abs() > 1e-8 { (next_x - start.x) / dir.x } else { 1e20 };
    let mut t_max_y = if dir.y.abs() > 1e-8 { (next_y - start.y) / dir.y } else { 1e20 };
    let mut t_max_z = if dir.z.abs() > 1e-8 { (next_z - start.z) / dir.z } else { 1e20 };

    let max_steps = grid_size * 2;
    let mut i: u32 = 0;

    // Step once before checking to skip the start voxel
    if t_max_x < t_max_y {
        if t_max_x < t_max_z {
            vx += step_x;
            t_max_x += t_delta_x;
        } else {
            vz += step_z;
            t_max_z += t_delta_z;
        }
    } else if t_max_y < t_max_z {
        vy += step_y;
        t_max_y += t_delta_y;
    } else {
        vz += step_z;
        t_max_z += t_delta_z;
    }
    i += 1;

    while i < max_steps {
        if vx < 0 || vx >= gs || vy < 0 || vy >= gs || vz < 0 || vz >= gs {
            return false;
        }

        let idx = (vz as u32 * grid_size * grid_size + vy as u32 * grid_size + vx as u32) as usize;
        if voxels[idx].w > 0 {
            return true;
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                vx += step_x;
                t_max_x += t_delta_x;
            } else {
                vz += step_z;
                t_max_z += t_delta_z;
            }
        } else if t_max_y < t_max_z {
            vy += step_y;
            t_max_y += t_delta_y;
        } else {
            vz += step_z;
            t_max_z += t_delta_z;
        }

        i += 1;
    }

    false
}

/// Apply simple diffuse lighting to a surface color.
///
/// Uses a directional light from the upper right and ambient term.
fn apply_lighting(color: Vec3, normal: Vec3) -> Vec3 {
    let light_dir = sun_direction();
    let ndotl = dot(normal, light_dir);
    let diffuse = if ndotl > 0.0 { ndotl } else { 0.0 };

    let ambient = 0.5;
    let intensity = ambient + (1.0 - ambient) * diffuse;

    Vec3::new(
        color.x * intensity,
        color.y * intensity,
        color.z * intensity,
    )
}

/// Compute point light contribution from nearby lava voxels.
///
/// Searches a 5x5x5 neighborhood (±2 cells) around the hit voxel for lava
/// (material_id == 2). Each lava voxel contributes warm orange light with
/// inverse-distance-squared attenuation. This illuminates cave interiors
/// that receive no sunlight.
fn compute_lava_light(
    vx: i32,
    vy: i32,
    vz: i32,
    voxels: &[UVec4],
    grid_size: u32,
) -> Vec3 {
    let gs = grid_size as i32;
    let mut light = Vec3::new(0.0, 0.0, 0.0);
    let search = 2i32; // check ±2 cells
    let mut dx = -search;
    while dx <= search {
        let mut dy = -search;
        while dy <= search {
            let mut dz = -search;
            while dz <= search {
                let nx = vx + dx;
                let ny = vy + dy;
                let nz = vz + dz;
                if nx >= 0 && nx < gs && ny >= 0 && ny < gs && nz >= 0 && nz < gs {
                    let idx = (nz as u32 * grid_size * grid_size
                        + ny as u32 * grid_size
                        + nx as u32) as usize;
                    let voxel = voxels[idx];
                    if voxel.w > 0 && ((voxel.x >> 16) & 0xFF) == 2 {
                        // Lava found — add warm orange light with distance falloff
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        let attenuation = 1.0 / (1.0 + dist_sq * 0.5);
                        light = Vec3::new(
                            light.x + 1.0 * attenuation * 0.6,
                            light.y + 0.5 * attenuation * 0.6,
                            light.z + 0.1 * attenuation * 0.6,
                        );
                    }
                }
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }
    light
}

/// Perform DDA ray march through the voxel grid and write pixel color.
///
/// This is the main per-pixel function, separated from the entry point
/// as required by the rust-gpu linker bug workaround (trap #4a).
pub fn render_pixel(
    px: u32,
    py: u32,
    push: &RenderPushConstants,
    voxels: &[UVec4],
    output: &mut [u32],
) {
    let width = push.width;
    let height = push.height;
    let grid_size = push.grid_size;

    if px >= width || py >= height {
        return;
    }

    let eye = Vec3::new(push.eye.x, push.eye.y, push.eye.z);
    let target = Vec3::new(push.target.x, push.target.y, push.target.z);
    let ray_dir = compute_ray(px, py, width, height, eye, target);

    // DDA ray marching through voxel grid
    let grid = grid_size as f32;

    // Find entry point into the grid [0, grid_size]^3
    // Ray: P = eye + t * dir
    // We need to find t_enter and t_exit for the AABB [0, grid]^3
    let mut t_min = 0.0_f32;
    let mut t_max = 1e20_f32;

    // X axis
    if ray_dir.x.abs() > 1e-8 {
        let t1 = (0.0 - eye.x) / ray_dir.x;
        let t2 = (grid - eye.x) / ray_dir.x;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max = if tb < t_max { tb } else { t_max };
    } else if eye.x < 0.0 || eye.x > grid {
        // Ray parallel to X slab and outside
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    // Y axis
    if ray_dir.y.abs() > 1e-8 {
        let t1 = (0.0 - eye.y) / ray_dir.y;
        let t2 = (grid - eye.y) / ray_dir.y;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max = if tb < t_max { tb } else { t_max };
    } else if eye.y < 0.0 || eye.y > grid {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    // Z axis
    if ray_dir.z.abs() > 1e-8 {
        let t1 = (0.0 - eye.z) / ray_dir.z;
        let t2 = (grid - eye.z) / ray_dir.z;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max = if tb < t_max { tb } else { t_max };
    } else if eye.z < 0.0 || eye.z > grid {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    if t_min > t_max || t_max < 0.0 {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    // Start point: clamp t_min to be >= 0 (eye may be inside the grid)
    if t_min < 0.0 {
        t_min = 0.0;
    }

    // Entry point into the grid
    let entry = Vec3::new(
        eye.x + ray_dir.x * (t_min + 0.001),
        eye.y + ray_dir.y * (t_min + 0.001),
        eye.z + ray_dir.z * (t_min + 0.001),
    );

    // Current voxel
    let gs = grid_size as i32;
    let mut vx = entry.x as i32;
    let mut vy = entry.y as i32;
    let mut vz = entry.z as i32;

    // Clamp to valid range
    if vx < 0 { vx = 0; }
    if vy < 0 { vy = 0; }
    if vz < 0 { vz = 0; }
    if vx >= gs { vx = gs - 1; }
    if vy >= gs { vy = gs - 1; }
    if vz >= gs { vz = gs - 1; }

    // Step direction
    let step_x: i32 = if ray_dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if ray_dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if ray_dir.z >= 0.0 { 1 } else { -1 };

    // t_delta: how far along the ray to cross one voxel in each axis
    let t_delta_x = if ray_dir.x.abs() > 1e-8 { (1.0 / ray_dir.x).abs() } else { 1e20 };
    let t_delta_y = if ray_dir.y.abs() > 1e-8 { (1.0 / ray_dir.y).abs() } else { 1e20 };
    let t_delta_z = if ray_dir.z.abs() > 1e-8 { (1.0 / ray_dir.z).abs() } else { 1e20 };

    // t_max_axis: distance to next voxel boundary
    let next_x = if step_x > 0 { (vx + 1) as f32 } else { vx as f32 };
    let next_y = if step_y > 0 { (vy + 1) as f32 } else { vy as f32 };
    let next_z = if step_z > 0 { (vz + 1) as f32 } else { vz as f32 };

    let mut t_max_x = if ray_dir.x.abs() > 1e-8 { (next_x - entry.x) / ray_dir.x } else { 1e20 };
    let mut t_max_y = if ray_dir.y.abs() > 1e-8 { (next_y - entry.y) / ray_dir.y } else { 1e20 };
    let mut t_max_z = if ray_dir.z.abs() > 1e-8 { (next_z - entry.z) / ray_dir.z } else { 1e20 };

    // DDA traversal
    let max_steps = grid_size * 3; // sqrt(3) * grid_size roughly
    // Track which axis was last stepped for face normal
    let mut last_axis: u32 = 0; // 0=x, 1=y, 2=z

    let mut i: u32 = 0;
    let mut hit = false;
    let mut hit_material: u32 = 0;

    while i < max_steps {
        // Check current voxel
        if vx >= 0 && vx < gs && vy >= 0 && vy < gs && vz >= 0 && vz < gs {
            let idx = (vz as u32 * grid_size * grid_size + vy as u32 * grid_size + vx as u32) as usize;
            let voxel = voxels[idx];
            if voxel.w > 0 {
                hit = true;
                hit_material = (voxel.x >> 16) & 0xFF;
                break;
            }
        } else {
            // Exited grid
            break;
        }

        // Step to next voxel (DDA)
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                vx += step_x;
                t_max_x += t_delta_x;
                last_axis = 0;
            } else {
                vz += step_z;
                t_max_z += t_delta_z;
                last_axis = 2;
            }
        } else if t_max_y < t_max_z {
            vy += step_y;
            t_max_y += t_delta_y;
            last_axis = 1;
        } else {
            vz += step_z;
            t_max_z += t_delta_z;
            last_axis = 2;
        }

        i += 1;
    }

    let pixel_idx = (py * width + px) as usize;

    if hit {
        // Determine face normal from the last DDA step axis
        let normal = if last_axis == 0 {
            Vec3::new(-(step_x as f32), 0.0, 0.0)
        } else if last_axis == 1 {
            Vec3::new(0.0, -(step_y as f32), 0.0)
        } else {
            Vec3::new(0.0, 0.0, -(step_z as f32))
        };

        // Compute ambient occlusion from neighboring voxels
        let ao_factor = compute_ao(vx, vy, vz, normal, voxels, grid_size);

        // Shadow ray: offset start position by half a voxel along the normal
        let shadow_origin = Vec3::new(
            vx as f32 + 0.5 + normal.x * 0.5,
            vy as f32 + 0.5 + normal.y * 0.5,
            vz as f32 + 0.5 + normal.z * 0.5,
        );
        let in_shadow = march_shadow(shadow_origin, sun_direction(), voxels, grid_size);

        // Read phase from packed voxel data (.x component, bits 8-15)
        let hit_idx = (vz as u32 * grid_size * grid_size + vy as u32 * grid_size + vx as u32) as usize;
        let hit_phase = (voxels[hit_idx].x >> 8) & 0xFF;

        // Override color for gas/steam (phase == 2): white-ish wispy appearance
        let base_color = if hit_phase == 2 {
            Vec3::new(0.9, 0.9, 0.95)
        } else {
            textured_material_color(hit_material, vx, vy, vz, grid_size)
        };

        // Lighting: if in shadow, only ambient contributes; otherwise full lighting
        let lit_color = if in_shadow {
            let ambient = 0.4;
            Vec3::new(
                base_color.x * ambient * ao_factor,
                base_color.y * ambient * ao_factor,
                base_color.z * ambient * ao_factor,
            )
        } else {
            let full_lit = apply_lighting(base_color, normal);
            Vec3::new(
                full_lit.x * ao_factor,
                full_lit.y * ao_factor,
                full_lit.z * ao_factor,
            )
        };

        // Add lava point lighting from nearby lava voxels
        let lava_light = compute_lava_light(vx, vy, vz, voxels, grid_size);
        let lit_color = Vec3::new(
            lit_color.x + base_color.x * lava_light.x,
            lit_color.y + base_color.y * lava_light.y,
            lit_color.z + base_color.z * lava_light.z,
        );

        // Add emissive glow for lava (material_id == 2) and burning wood (material_id == 3)
        let final_color = if hit_material == 2 {
            let emissive = Vec3::new(1.0, 0.6, 0.1) * 0.8;
            Vec3::new(
                lit_color.x + emissive.x,
                lit_color.y + emissive.y,
                lit_color.z + emissive.z,
            )
        } else if hit_material == 3 {
            // Burning wood: emissive orange glow based on temperature
            // Temperature is encoded in the voxel; approximate by using
            // a moderate glow since we know wood on fire has T > 300
            let emissive = Vec3::new(1.0, 0.4, 0.05) * 0.5;
            Vec3::new(
                lit_color.x + emissive.x,
                lit_color.y + emissive.y,
                lit_color.z + emissive.z,
            )
        } else {
            lit_color
        };

        // Clamp and convert to u8
        let r = ((final_color.x * 255.0) as u32).min(255);
        let g = ((final_color.y * 255.0) as u32).min(255);
        let b = ((final_color.z * 255.0) as u32).min(255);

        output[pixel_idx] = pack_bgra(r, g, b, 255);
    } else {
        // Sky gradient based on ray direction Y
        let t = (ray_dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
        // Lerp from dark navy (bottom) to slightly lighter (top)
        let sky = Vec3::new(
            0.05 + (0.1 - 0.05) * t,
            0.05 + (0.1 - 0.05) * t,
            0.12 + (0.2 - 0.12) * t,
        );
        let r = ((sky.x * 255.0) as u32).min(255);
        let g = ((sky.y * 255.0) as u32).min(255);
        let b = ((sky.z * 255.0) as u32).min(255);
        output[pixel_idx] = pack_bgra(r, g, b, 255);
    }
}

/// Compute shader entry point: render voxels via ray marching.
///
/// Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
/// Descriptor set 0, binding 0: storage buffer of `UVec4` (voxel grid, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (pixel output, write).
/// Push constants: `RenderPushConstants`.
#[spirv(compute(threads(8, 8, 1)))]
pub fn render_voxels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &RenderPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] voxels: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
) {
    render_pixel(id.x, id.y, push, voxels, output);
}
