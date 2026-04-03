//! Render compute shader: ray-marches through voxel grid and writes pixels.
//!
//! Dispatched with (ceil(width/8), ceil(height/8), 1) workgroups.
//! Each thread computes one pixel via DDA ray marching through the 3D voxel grid.

pub mod lighting;
pub mod materials;
pub mod math;
pub mod sky;

use crate::types::{MaterialParams, MAT_WATER};
use spirv_std::glam::{UVec3, UVec4, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

use self::lighting::{apply_lighting, compute_ao, compute_lava_light, march_shadow, sun_direction};
use self::materials::{apply_emissive, textured_material_color};
use self::math::{compute_ray, pack_bgra};
use self::sky::compute_sky_color;

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

/// Shade a hit voxel: compute lighting, AO, shadows, lava light, and emissive.
///
/// Separated from render_pixel to keep function complexity manageable and
/// avoid the rust-gpu branch-dropping trap (#15).
fn shade_voxel(
    vx: i32,
    vy: i32,
    vz: i32,
    last_axis: u32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    hit_material: u32,
    voxels: &[UVec4],
    grid_size: u32,
    materials: &[MaterialParams],
) -> Vec3 {
    // Determine face normal from last DDA step axis
    let normal = if last_axis == 0 {
        Vec3::new(-(step_x as f32), 0.0, 0.0)
    } else if last_axis == 1 {
        Vec3::new(0.0, -(step_y as f32), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -(step_z as f32))
    };

    let ao_factor = compute_ao(vx, vy, vz, normal, voxels, grid_size);

    // Shadow ray
    let shadow_origin = Vec3::new(
        vx as f32 + 0.5 + normal.x * 0.5,
        vy as f32 + 0.5 + normal.y * 0.5,
        vz as f32 + 0.5 + normal.z * 0.5,
    );
    let in_shadow = march_shadow(shadow_origin, sun_direction(), voxels, grid_size);

    let hit_idx = (vz as u32 * grid_size * grid_size + vy as u32 * grid_size + vx as u32) as usize;
    let hit_phase = (voxels[hit_idx].x >> 8) & 0xFF;

    // Base color: gas/steam gets white-ish, others from material table
    let base_color = if hit_phase == 2 {
        Vec3::new(0.9, 0.9, 0.95)
    } else {
        textured_material_color(hit_material, vx, vy, vz, materials)
    };

    // Diffuse + ambient lighting with shadow
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

    // Add lava point lighting
    let lava_light = compute_lava_light(vx, vy, vz, voxels, grid_size);
    let lit_color = Vec3::new(
        lit_color.x + base_color.x * lava_light.x,
        lit_color.y + base_color.y * lava_light.y,
        lit_color.z + base_color.z * lava_light.z,
    );

    // Temperature-based emissive
    let temperature = f32::from_bits(voxels[hit_idx].y);
    apply_emissive(lit_color, hit_material, hit_phase, temperature)
}

/// Compute the number of voxel steps to skip to exit the current brick along one axis.
///
/// Given the current voxel coordinate and step direction, returns the distance
/// (in voxels) to the far edge of the 8-voxel brick boundary.
fn brick_exit_steps(voxel_coord: i32, step: i32) -> i32 {
    if step > 0 {
        let next_boundary = ((voxel_coord >> 3) + 1) << 3;
        next_boundary - voxel_coord
    } else {
        let brick_start = (voxel_coord >> 3) << 3;
        voxel_coord - brick_start + 1
    }
}

/// Compute the number of voxel steps to skip to exit the current super-brick (64³) along one axis.
///
/// Same logic as `brick_exit_steps` but for 64-voxel boundaries instead of 8.
fn super_brick_exit_steps(voxel_coord: i32, step: i32) -> i32 {
    if step > 0 {
        let next_boundary = ((voxel_coord >> 6) + 1) << 6;
        next_boundary - voxel_coord
    } else {
        let sb_start = (voxel_coord >> 6) << 6;
        voxel_coord - sb_start + 1
    }
}

/// Check if a voxel is inside an empty super-brick (64³ = 8x8x8 bricks) and compute skip offsets.
///
/// Returns (delta_vx, delta_vy, delta_vz, delta_tx, delta_ty, delta_tz).
/// All zeros if the super-brick is occupied (no skip). Otherwise, the deltas advance
/// the ray past the empty super-brick along the axis with the nearest exit.
fn check_super_brick_skip(
    vx: i32,
    vy: i32,
    vz: i32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    t_max_x: f32,
    t_max_y: f32,
    t_max_z: f32,
    t_delta_x: f32,
    t_delta_y: f32,
    t_delta_z: f32,
    super_brick_occupied: &[u32],
    grid_size: u32,
) -> (i32, i32, i32, f32, f32, f32) {
    let sbpa = grid_size / 64; // super-bricks per axis
    if sbpa == 0 {
        return (0, 0, 0, 0.0, 0.0, 0.0);
    }
    let sbx = (vx as u32) / 64;
    let sby = (vy as u32) / 64;
    let sbz = (vz as u32) / 64;

    if sbx >= sbpa || sby >= sbpa || sbz >= sbpa {
        return (0, 0, 0, 0.0, 0.0, 0.0);
    }

    let sb_idx = (sbz * sbpa * sbpa + sby * sbpa + sbx) as usize;
    if super_brick_occupied[sb_idx] != 0 {
        return (0, 0, 0, 0.0, 0.0, 0.0);
    }

    // Super-brick is empty: advance ray to exit the 64³ region
    compute_skip_deltas_64(vx, vy, vz, step_x, step_y, step_z, t_max_x, t_max_y, t_max_z, t_delta_x, t_delta_y, t_delta_z)
}

/// Compute skip deltas to exit a 64³ super-brick. Separated to avoid >3 branches (trap #15).
fn compute_skip_deltas_64(
    vx: i32,
    vy: i32,
    vz: i32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    t_max_x: f32,
    t_max_y: f32,
    t_max_z: f32,
    t_delta_x: f32,
    t_delta_y: f32,
    t_delta_z: f32,
) -> (i32, i32, i32, f32, f32, f32) {
    let sx = super_brick_exit_steps(vx, step_x);
    let sy = super_brick_exit_steps(vy, step_y);
    let sz = super_brick_exit_steps(vz, step_z);

    let t_exit_x = t_max_x + (sx - 1) as f32 * t_delta_x;
    let t_exit_y = t_max_y + (sy - 1) as f32 * t_delta_y;
    let t_exit_z = t_max_z + (sz - 1) as f32 * t_delta_z;

    if t_exit_x <= t_exit_y && t_exit_x <= t_exit_z {
        (sx * step_x, 0, 0, sx as f32 * t_delta_x, 0.0, 0.0)
    } else if t_exit_y <= t_exit_z {
        (0, sy * step_y, 0, 0.0, sy as f32 * t_delta_y, 0.0)
    } else {
        (0, 0, sz * step_z, 0.0, 0.0, sz as f32 * t_delta_z)
    }
}

/// Check if a voxel is inside an empty brick and compute skip offsets.
///
/// Returns (delta_vx, delta_vy, delta_vz, delta_tx, delta_ty, delta_tz).
/// All zeros if the brick is occupied (no skip). Otherwise, the deltas advance
/// the ray past the empty brick along the axis with the nearest exit.
fn check_brick_skip(
    vx: i32,
    vy: i32,
    vz: i32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    t_max_x: f32,
    t_max_y: f32,
    t_max_z: f32,
    t_delta_x: f32,
    t_delta_y: f32,
    t_delta_z: f32,
    bricks_per_axis: u32,
    brick_occupied: &[u32],
) -> (i32, i32, i32, f32, f32, f32) {
    let bx = (vx as u32) >> 3;
    let by = (vy as u32) >> 3;
    let bz = (vz as u32) >> 3;
    let bpa = bricks_per_axis;

    if bx >= bpa || by >= bpa || bz >= bpa {
        return (0, 0, 0, 0.0, 0.0, 0.0);
    }

    let brick_idx = (bz * bpa * bpa + by * bpa + bx) as usize;
    if brick_occupied[brick_idx] != 0 {
        return (0, 0, 0, 0.0, 0.0, 0.0);
    }

    // Brick is empty: advance ray to exit the brick.
    let sx = brick_exit_steps(vx, step_x);
    let sy = brick_exit_steps(vy, step_y);
    let sz = brick_exit_steps(vz, step_z);

    let t_exit_x = t_max_x + (sx - 1) as f32 * t_delta_x;
    let t_exit_y = t_max_y + (sy - 1) as f32 * t_delta_y;
    let t_exit_z = t_max_z + (sz - 1) as f32 * t_delta_z;

    if t_exit_x <= t_exit_y && t_exit_x <= t_exit_z {
        (sx * step_x, 0, 0, sx as f32 * t_delta_x, 0.0, 0.0)
    } else if t_exit_y <= t_exit_z {
        (0, sy * step_y, 0, 0.0, sy as f32 * t_delta_y, 0.0)
    } else {
        (0, 0, sz * step_z, 0.0, 0.0, sz as f32 * t_delta_z)
    }
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
    materials: &[MaterialParams],
    brick_occupied: &[u32],
    super_brick_occupied: &[u32],
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
    let mut t_min = 0.0_f32;
    let mut t_max_bound = 1e20_f32;

    // X axis slab test
    if ray_dir.x.abs() > 1e-8 {
        let t1 = (0.0 - eye.x) / ray_dir.x;
        let t2 = (grid - eye.x) / ray_dir.x;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max_bound = if tb < t_max_bound { tb } else { t_max_bound };
    } else if eye.x < 0.0 || eye.x > grid {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    // Y axis slab test
    if ray_dir.y.abs() > 1e-8 {
        let t1 = (0.0 - eye.y) / ray_dir.y;
        let t2 = (grid - eye.y) / ray_dir.y;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max_bound = if tb < t_max_bound { tb } else { t_max_bound };
    } else if eye.y < 0.0 || eye.y > grid {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    // Z axis slab test
    if ray_dir.z.abs() > 1e-8 {
        let t1 = (0.0 - eye.z) / ray_dir.z;
        let t2 = (grid - eye.z) / ray_dir.z;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = if ta > t_min { ta } else { t_min };
        t_max_bound = if tb < t_max_bound { tb } else { t_max_bound };
    } else if eye.z < 0.0 || eye.z > grid {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

    if t_min > t_max_bound || t_max_bound < 0.0 {
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = pack_bgra(18, 18, 40, 255);
        return;
    }

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

    // DDA traversal with water transparency support.
    // When a water voxel is hit, we accumulate its tint and continue marching.
    // When a solid (non-water) voxel is hit, we shade it and blend with
    // accumulated water color. If the ray exits without hitting solid,
    // we blend water with sky.
    let max_steps = grid_size * 3;
    let mut last_axis: u32 = 0;

    let mut i: u32 = 0;
    let mut hit = false;
    let mut hit_material: u32 = 0;

    // Water transparency accumulation
    let mut water_transmittance = 1.0_f32; // remaining light (1.0 = no water traversed)
    let mut water_tint = Vec3::new(0.0, 0.0, 0.0); // accumulated water color contribution

    // Read water opacity from MaterialParams
    let water_opacity = if (MAT_WATER as usize) < materials.len() {
        materials[MAT_WATER as usize].visual.z
    } else {
        0.3
    };
    // Water base color from materials
    let water_base = if (MAT_WATER as usize) < materials.len() {
        let c = materials[MAT_WATER as usize].color;
        Vec3::new(c.x, c.y, c.z)
    } else {
        Vec3::new(0.2, 0.4, 0.8)
    };

    let bricks_per_axis = grid_size >> 3; // grid_size / 8

    while i < max_steps {
        // Check current voxel
        if vx >= 0 && vx < gs && vy >= 0 && vy < gs && vz >= 0 && vz < gs {
            // Super-brick-level skip: if the entire 64³ region is empty, advance past it
            let (sb_vx, sb_vy, sb_vz, sb_tx, sb_ty, sb_tz) = check_super_brick_skip(
                vx, vy, vz,
                step_x, step_y, step_z,
                t_max_x, t_max_y, t_max_z,
                t_delta_x, t_delta_y, t_delta_z,
                super_brick_occupied,
                grid_size,
            );
            if sb_vx != 0 || sb_vy != 0 || sb_vz != 0 {
                vx += sb_vx;
                vy += sb_vy;
                vz += sb_vz;
                t_max_x += sb_tx;
                t_max_y += sb_ty;
                t_max_z += sb_tz;
                i += 1;
                continue;
            }

            // Brick-level skip: if the entire 8x8x8 brick is empty, advance past it
            let (skip_vx, skip_vy, skip_vz, skip_tx, skip_ty, skip_tz) = check_brick_skip(
                vx, vy, vz,
                step_x, step_y, step_z,
                t_max_x, t_max_y, t_max_z,
                t_delta_x, t_delta_y, t_delta_z,
                bricks_per_axis,
                brick_occupied,
            );
            if skip_vx != 0 || skip_vy != 0 || skip_vz != 0 {
                vx += skip_vx;
                vy += skip_vy;
                vz += skip_vz;
                t_max_x += skip_tx;
                t_max_y += skip_ty;
                t_max_z += skip_tz;
                i += 1;
                continue;
            }

            let idx = (vz as u32 * grid_size * grid_size + vy as u32 * grid_size + vx as u32) as usize;
            let voxel = voxels[idx];
            if voxel.w > 0 {
                let mat_id = (voxel.x >> 16) & 0xFF;
                if mat_id == MAT_WATER && water_transmittance > 0.05 {
                    // Water voxel: accumulate tint and continue marching.
                    // Per-voxel absorption: water_color * opacity blended into tint.
                    let absorption = water_opacity * water_transmittance;
                    water_tint = Vec3::new(
                        water_tint.x + water_base.x * absorption,
                        water_tint.y + water_base.y * absorption,
                        water_tint.z + water_base.z * absorption,
                    );
                    water_transmittance *= 1.0 - water_opacity;
                    // Do NOT break — continue DDA to find what's behind the water
                } else {
                    hit = true;
                    hit_material = mat_id;
                    break;
                }
            }
        } else {
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

    // Determine the background color (either shaded solid or sky)
    let background = if hit {
        shade_voxel(
            vx, vy, vz, last_axis, step_x, step_y, step_z,
            hit_material, voxels, grid_size, materials,
        )
    } else {
        compute_sky_color(ray_dir)
    };

    // Blend water tint with background using accumulated transmittance
    let final_color = Vec3::new(
        water_tint.x + background.x * water_transmittance,
        water_tint.y + background.y * water_transmittance,
        water_tint.z + background.z * water_transmittance,
    );

    // Clamp and convert to u8
    let r = ((final_color.x * 255.0) as u32).min(255);
    let g = ((final_color.y * 255.0) as u32).min(255);
    let b = ((final_color.z * 255.0) as u32).min(255);

    output[pixel_idx] = pack_bgra(r, g, b, 255);
}

/// Tile size in pixels for dirty-tile rendering.
const TILE_SIZE: u32 = 16;

/// Check if the tile containing this pixel is dirty, and if clean, copy from
/// the previous frame buffer.
///
/// Returns `true` if the pixel needs full ray march (dirty tile),
/// `false` if the pixel was copied from the previous frame (clean tile).
///
/// Helper function to satisfy trap #4a.
fn handle_dirty_tile(
    px: u32,
    py: u32,
    width: u32,
    height: u32,
    dirty_tiles: &[u32],
    prev_output: &[u32],
    output: &mut [u32],
) -> bool {
    if px >= width || py >= height {
        return false;
    }

    let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tile_x = px / TILE_SIZE;
    let tile_y = py / TILE_SIZE;
    let tile_idx = tile_y * tiles_x + tile_x;

    if dirty_tiles[tile_idx as usize] == 0 {
        // Clean tile: copy from previous frame
        let pixel_idx = (py * width + px) as usize;
        output[pixel_idx] = prev_output[pixel_idx];
        return false;
    }

    true
}

/// Compute shader entry point: render voxels via ray marching.
///
/// Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
/// Descriptor set 0, binding 0: storage buffer of `UVec4` (voxel grid, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (pixel output, write).
/// Descriptor set 0, binding 2: storage buffer of `MaterialParams` (material table, read).
/// Descriptor set 0, binding 3: storage buffer of `u32` (brick_occupied map, read).
/// Descriptor set 0, binding 4: storage buffer of `u32` (super_brick_occupied map, read).
/// Descriptor set 0, binding 5: storage buffer of `u32` (dirty_tile_buffer, read).
/// Descriptor set 0, binding 6: storage buffer of `u32` (prev_render_output, read).
/// Push constants: `RenderPushConstants`.
#[spirv(compute(threads(8, 8, 1)))]
pub fn render_voxels(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &RenderPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] voxels: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] materials: &[MaterialParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] brick_occupied: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] super_brick_occupied: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dirty_tiles: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] prev_output: &[u32],
) {
    // Check dirty tile first; if clean, pixel is already copied from prev frame
    if !handle_dirty_tile(id.x, id.y, push.width, push.height, dirty_tiles, prev_output, output) {
        return;
    }
    render_pixel(id.x, id.y, push, voxels, output, materials, brick_occupied, super_brick_occupied);
}
