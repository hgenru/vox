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

/// Get color for a material ID.
///
/// Returns (r, g, b) as floats in [0, 1].
fn material_color(material_id: u32) -> Vec3 {
    if material_id == 0 {
        // Stone: warm gray
        Vec3::new(0.55, 0.52, 0.5)
    } else if material_id == 1 {
        // Water: brighter blue
        Vec3::new(0.15, 0.4, 0.95)
    } else if material_id == 2 {
        // Lava: orange-red base (emissive added in shading)
        Vec3::new(1.0, 0.3, 0.05)
    } else {
        // Unknown: magenta
        Vec3::new(1.0, 0.0, 1.0)
    }
}

/// Apply simple diffuse lighting to a surface color.
///
/// Uses a directional light from the upper right and ambient term.
fn apply_lighting(color: Vec3, normal: Vec3) -> Vec3 {
    let light_dir = normalize_vec3(Vec3::new(0.4, 0.8, 0.3));
    let ndotl = dot(normal, light_dir);
    let diffuse = if ndotl > 0.0 { ndotl } else { 0.0 };

    let ambient = 0.3;
    let intensity = ambient + (1.0 - ambient) * diffuse;

    Vec3::new(
        color.x * intensity,
        color.y * intensity,
        color.z * intensity,
    )
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
                hit_material = voxel.x;
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

        let base_color = material_color(hit_material);
        let lit_color = apply_lighting(base_color, normal);

        // Add emissive glow for lava (material_id == 2)
        let final_color = if hit_material == 2 {
            let emissive = Vec3::new(1.0, 0.6, 0.1) * 0.8;
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
