//! Far-field render compute shader: fills in sky pixels with far-field chunk data.
//!
//! Runs as a second pass AFTER the near-field render. For each pixel that was
//! left as sky (alpha channel marker), casts a ray into far-field chunks and
//! writes the hit color. This avoids touching the performance-critical DDA loop
//! in the main render shader.

use crate::compute::render::math::{compute_ray, pack_bgra};
use crate::compute::render::sky::compute_sky_color;
use crate::types::{ChunkTableEntry, FAR_CHUNK_SIZE, MaterialParams};
use spirv_std::glam::{UVec3, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

/// Push constants for the far-field render shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FarFieldPushConstants {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Number of valid entries in the chunk table.
    pub chunk_count: u32,
    /// Grid dimension of the sim domain (cells per axis).
    pub grid_size: u32,
    /// Camera eye position (xyz) + padding (w).
    pub eye: Vec4,
    /// Camera target position (xyz) + padding (w).
    pub target: Vec4,
    /// World-space origin of the sim domain (xyz) + padding (w).
    /// Grid voxel [0,0,0] maps to this world position.
    pub sim_origin: Vec4,
}

/// Compute simple shading for a far-field voxel hit.
///
/// Uses the material color from the table with a basic directional light + ambient.
/// Separated into a helper to satisfy trap #4a (entry points must call helpers).
fn shade_far_field_hit(material_id: u32, normal: Vec3, materials: &[MaterialParams]) -> Vec3 {
    let base_color = if (material_id as usize) < materials.len() {
        let c = materials[material_id as usize].color;
        Vec3::new(c.x, c.y, c.z)
    } else {
        Vec3::new(0.5, 0.5, 0.5)
    };

    // Simple directional light (sun from upper-right)
    let sun_dir = Vec3::new(0.4, 0.7, 0.3);
    let sun_len_sq = sun_dir.x * sun_dir.x + sun_dir.y * sun_dir.y + sun_dir.z * sun_dir.z;
    let inv_sun_len = 1.0 / sun_len_sq.sqrt();
    let sun_norm = Vec3::new(sun_dir.x * inv_sun_len, sun_dir.y * inv_sun_len, sun_dir.z * inv_sun_len);

    let ndotl = normal.x * sun_norm.x + normal.y * sun_norm.y + normal.z * sun_norm.z;
    let ndotl = if ndotl > 0.0 { ndotl } else { 0.0 };

    let ambient = 0.35;
    let diffuse = 0.65 * ndotl;
    let factor = ambient + diffuse;

    Vec3::new(
        (base_color.x * factor).min(1.0),
        (base_color.y * factor).min(1.0),
        (base_color.z * factor).min(1.0),
    )
}

/// Read a u8 material ID from the packed u32 voxel buffer.
///
/// The far-field voxel buffer stores u8 values packed 4 per u32.
/// Returns the material ID at the given voxel offset.
fn read_voxel_u8(far_voxels: &[u32], voxel_offset: u32) -> u32 {
    let word_idx = (voxel_offset / 4) as usize;
    let byte_idx = voxel_offset % 4;
    let word = far_voxels[word_idx];
    (word >> (byte_idx * 8)) & 0xFF
}

/// DDA ray march through a single far-field chunk.
///
/// Returns (hit, material_id, normal) where hit is true if a non-empty voxel was found.
/// The normal is the face normal of the last DDA step axis.
fn march_chunk(
    ray_origin: Vec3,
    ray_dir: Vec3,
    chunk_origin: Vec3,
    voxel_base_offset: u32,
    far_voxels: &[u32],
) -> (bool, u32, Vec3) {
    let chunk_size = FAR_CHUNK_SIZE as f32;

    // Ray-AABB intersection with the chunk
    let inv_dir_x = if ray_dir.x.abs() > 1e-8 { 1.0 / ray_dir.x } else { 1e20 };
    let inv_dir_y = if ray_dir.y.abs() > 1e-8 { 1.0 / ray_dir.y } else { 1e20 };
    let inv_dir_z = if ray_dir.z.abs() > 1e-8 { 1.0 / ray_dir.z } else { 1e20 };

    let t1x = (chunk_origin.x - ray_origin.x) * inv_dir_x;
    let t2x = (chunk_origin.x + chunk_size - ray_origin.x) * inv_dir_x;
    let t1y = (chunk_origin.y - ray_origin.y) * inv_dir_y;
    let t2y = (chunk_origin.y + chunk_size - ray_origin.y) * inv_dir_y;
    let t1z = (chunk_origin.z - ray_origin.z) * inv_dir_z;
    let t2z = (chunk_origin.z + chunk_size - ray_origin.z) * inv_dir_z;

    let tmin_x = if t1x < t2x { t1x } else { t2x };
    let tmax_x = if t1x > t2x { t1x } else { t2x };
    let tmin_y = if t1y < t2y { t1y } else { t2y };
    let tmax_y = if t1y > t2y { t1y } else { t2y };
    let tmin_z = if t1z < t2z { t1z } else { t2z };
    let tmax_z = if t1z > t2z { t1z } else { t2z };

    let mut t_enter = tmin_x;
    if tmin_y > t_enter { t_enter = tmin_y; }
    if tmin_z > t_enter { t_enter = tmin_z; }

    let mut t_exit = tmax_x;
    if tmax_y < t_exit { t_exit = tmax_y; }
    if tmax_z < t_exit { t_exit = tmax_z; }

    if t_enter > t_exit || t_exit < 0.0 {
        return (false, 0, Vec3::new(0.0, 1.0, 0.0));
    }
    if t_enter < 0.0 { t_enter = 0.0; }

    // Entry point into chunk local coords
    let entry = Vec3::new(
        ray_origin.x + ray_dir.x * (t_enter + 0.001) - chunk_origin.x,
        ray_origin.y + ray_dir.y * (t_enter + 0.001) - chunk_origin.y,
        ray_origin.z + ray_dir.z * (t_enter + 0.001) - chunk_origin.z,
    );

    let cs = FAR_CHUNK_SIZE as i32;
    let mut vx = entry.x as i32;
    let mut vy = entry.y as i32;
    let mut vz = entry.z as i32;

    // Clamp
    if vx < 0 { vx = 0; }
    if vy < 0 { vy = 0; }
    if vz < 0 { vz = 0; }
    if vx >= cs { vx = cs - 1; }
    if vy >= cs { vy = cs - 1; }
    if vz >= cs { vz = cs - 1; }

    let step_x: i32 = if ray_dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if ray_dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if ray_dir.z >= 0.0 { 1 } else { -1 };

    let t_delta_x = if ray_dir.x.abs() > 1e-8 { (1.0 / ray_dir.x).abs() } else { 1e20 };
    let t_delta_y = if ray_dir.y.abs() > 1e-8 { (1.0 / ray_dir.y).abs() } else { 1e20 };
    let t_delta_z = if ray_dir.z.abs() > 1e-8 { (1.0 / ray_dir.z).abs() } else { 1e20 };

    let next_x = if step_x > 0 { (vx + 1) as f32 } else { vx as f32 };
    let next_y = if step_y > 0 { (vy + 1) as f32 } else { vy as f32 };
    let next_z = if step_z > 0 { (vz + 1) as f32 } else { vz as f32 };

    let mut t_max_x = if ray_dir.x.abs() > 1e-8 { (next_x - entry.x) / ray_dir.x } else { 1e20 };
    let mut t_max_y = if ray_dir.y.abs() > 1e-8 { (next_y - entry.y) / ray_dir.y } else { 1e20 };
    let mut t_max_z = if ray_dir.z.abs() > 1e-8 { (next_z - entry.z) / ray_dir.z } else { 1e20 };

    // Max steps = 3 * 64 = 192
    let max_steps = FAR_CHUNK_SIZE * 3;
    let mut last_axis: u32 = 0;
    let mut i: u32 = 0;

    while i < max_steps {
        if vx < 0 || vx >= cs || vy < 0 || vy >= cs || vz < 0 || vz >= cs {
            break;
        }

        let local_offset = (vz as u32) * FAR_CHUNK_SIZE * FAR_CHUNK_SIZE
            + (vy as u32) * FAR_CHUNK_SIZE
            + (vx as u32);
        let mat_id = read_voxel_u8(far_voxels, voxel_base_offset + local_offset);

        if mat_id != 0 {
            let normal = compute_hit_normal(last_axis, step_x, step_y, step_z);
            return (true, mat_id, normal);
        }

        // DDA step
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

    (false, 0, Vec3::new(0.0, 1.0, 0.0))
}

/// Compute hit normal from DDA last axis and step direction.
///
/// Separated to keep branch count low per function (trap #15).
fn compute_hit_normal(last_axis: u32, step_x: i32, step_y: i32, step_z: i32) -> Vec3 {
    if last_axis == 0 {
        Vec3::new(-(step_x as f32), 0.0, 0.0)
    } else if last_axis == 1 {
        Vec3::new(0.0, -(step_y as f32), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -(step_z as f32))
    }
}

/// Per-pixel far-field rendering logic.
///
/// Checks if the pixel is sky, then casts a ray through far-field chunks.
/// Helper for the entry point (trap #4a).
fn render_far_field_pixel(
    px: u32,
    py: u32,
    push: &FarFieldPushConstants,
    render_output: &mut [u32],
    far_field_voxels: &[u32],
    far_field_table: &[ChunkTableEntry],
    materials: &[MaterialParams],
) {
    let width = push.width;
    let height = push.height;

    if px >= width || py >= height {
        return;
    }

    let pixel_idx = (py * width + px) as usize;
    let existing = render_output[pixel_idx];

    // Check if pixel is sky: alpha byte (bits 24-31) marks fully opaque (0xFF)
    // for near-field hits. If the pixel was written by near-field render, skip it.
    // Near-field render writes alpha=0xFF (255). Sky pixels also get 0xFF from pack_bgra.
    // We use a special marker: if the existing color matches the sky color exactly,
    // it's a sky pixel. But that's fragile. Instead, we use a simpler approach:
    // the near-field render writes the sky color (dark navy 0x2812120C with pack_bgra).
    // We'll check if the pixel's RGB matches the sky gradient range.
    //
    // Simpler approach: always try far-field for ALL pixels. If we get a hit closer
    // than the sim domain, we won't know. But since far-field is behind the sim domain,
    // we only run this for pixels that got sky color from the main render.
    //
    // The most robust approach: check if the pixel's low bits match the sky pack pattern.
    // Near-field sky: pack_bgra(18, 18, 40, 255) = 0xFF284012 (approx).
    // Let's just check if the pixel matches sky color exactly OR if alpha==0.
    //
    // Actually, the simplest: compare to the sky gradient. The near-field render writes
    // specific sky colors via compute_sky_color + pack_bgra. These are deterministic
    // per-pixel from the ray direction. If the existing pixel equals what sky would produce,
    // it's sky. But that requires re-computing sky. Too expensive.
    //
    // Best approach: use a flag. The simplest flag is alpha < 255 for sky.
    // But current render always writes alpha=255. We'd need to change the near-field
    // render to write alpha=0 for sky pixels. That's a small change.
    //
    // For now, use the simplest possible heuristic: re-compute what the sky color
    // would be for this pixel and compare. If it matches, this is a sky pixel.
    // This works because sky colors are deterministic per ray direction.

    // Check if chunk_count is 0 - nothing to render
    if push.chunk_count == 0 {
        return;
    }

    let eye = Vec3::new(push.eye.x, push.eye.y, push.eye.z);
    let target = Vec3::new(push.target.x, push.target.y, push.target.z);
    let ray_dir = compute_ray(px, py, width, height, eye, target);

    // Re-compute expected sky color for this pixel
    let sky = compute_sky_color(ray_dir);
    let sky_r = ((sky.x * 255.0) as u32).min(255);
    let sky_g = ((sky.y * 255.0) as u32).min(255);
    let sky_b = ((sky.z * 255.0) as u32).min(255);
    let expected_sky = pack_bgra(sky_r, sky_g, sky_b, 255);

    // If existing pixel doesn't match sky, it was hit by near-field - skip
    if existing != expected_sky {
        return;
    }

    // This pixel is sky. Try to find a far-field chunk hit.
    // World-space ray: eye is in sim-domain coords, convert to world coords.
    let sim_origin = Vec3::new(push.sim_origin.x, push.sim_origin.y, push.sim_origin.z);
    let world_eye = Vec3::new(
        eye.x + sim_origin.x,
        eye.y + sim_origin.y,
        eye.z + sim_origin.z,
    );

    // Find nearest chunk hit
    let mut best_t = 1e20_f32;
    let mut best_color = sky;
    let mut found_hit = false;

    let count = push.chunk_count;
    // Linear scan over chunk table (fine for <128 chunks, avoids >3 branches)
    let mut ci: u32 = 0;
    while ci < count {
        let entry_origin = far_field_table[ci as usize].origin;
        let entry_offset = far_field_table[ci as usize].offset_pad.x;

        // Skip invalid entries
        if entry_origin.w < 0.5 {
            ci += 1;
            continue;
        }

        let chunk_origin = Vec3::new(entry_origin.x, entry_origin.y, entry_origin.z);

        let (hit, mat_id, normal) = march_chunk(
            world_eye, ray_dir, chunk_origin, entry_offset, far_field_voxels,
        );

        if hit {
            // Compute approximate t for sorting (distance from eye to chunk origin)
            let dx = chunk_origin.x + 32.0 - world_eye.x;
            let dy = chunk_origin.y + 32.0 - world_eye.y;
            let dz = chunk_origin.z + 32.0 - world_eye.z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < best_t {
                best_t = dist_sq;
                best_color = shade_far_field_hit(mat_id, normal, materials);
                found_hit = true;
            }
        }

        ci += 1;
    }

    if found_hit {
        let r = ((best_color.x * 255.0) as u32).min(255);
        let g = ((best_color.y * 255.0) as u32).min(255);
        let b = ((best_color.z * 255.0) as u32).min(255);
        render_output[pixel_idx] = pack_bgra(r, g, b, 255);
    }
}

/// Compute shader entry point: render far-field chunks behind the near-field.
///
/// Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
/// Runs after the main render pass. For each pixel that shows sky, casts a ray
/// into far-field chunks and writes the hit color.
///
/// Descriptor set 0:
/// - binding 0: render_output (u32, read/write)
/// - binding 1: far_field_voxels (u32, packed u8s, read)
/// - binding 2: far_field_table (ChunkTableEntry, read)
/// - binding 3: materials (MaterialParams, read)
///
/// Push constants: `FarFieldPushConstants`.
#[spirv(compute(threads(8, 8, 1)))]
pub fn render_far_field(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &FarFieldPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] render_output: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] far_field_voxels: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] far_field_table: &[ChunkTableEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] materials: &[MaterialParams],
) {
    render_far_field_pixel(
        id.x, id.y, push,
        render_output, far_field_voxels, far_field_table, materials,
    );
}
