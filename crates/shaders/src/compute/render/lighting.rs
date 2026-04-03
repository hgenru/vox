//! Lighting functions: ambient occlusion, shadow rays, diffuse lighting, lava point lights.

use spirv_std::glam::{UVec4, Vec3};

use super::math::{dot, normalize_vec3};

/// Sun direction constant used for both diffuse lighting and shadow rays.
const SUN_DIR_RAW: Vec3 = Vec3::new(0.4, 0.8, 0.3);

/// Return the normalized sun direction.
pub(crate) fn sun_direction() -> Vec3 {
    normalize_vec3(SUN_DIR_RAW)
}

/// Compute voxel-based ambient occlusion.
///
/// Samples 8 neighbors (4 edge + 4 corner) in the plane perpendicular to the
/// hit face normal. Each occupied edge neighbor adds 0.12 occlusion, each
/// occupied corner neighbor adds 0.08. Total is clamped to 0.6 max.
/// Returns the AO factor (1.0 - occlusion).
pub(crate) fn compute_ao(
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
pub(crate) fn march_shadow(
    start: Vec3,
    dir: Vec3,
    voxels: &[UVec4],
    grid_size: u32,
) -> bool {
    let gs = grid_size as i32;

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
pub(crate) fn apply_lighting(color: Vec3, normal: Vec3) -> Vec3 {
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
/// Searches a 5x5x5 neighborhood (+-2 cells) around the hit voxel for lava
/// (material_id == 2). Each lava voxel contributes warm orange light with
/// inverse-distance-squared attenuation. This illuminates cave interiors
/// that receive no sunlight.
pub(crate) fn compute_lava_light(
    vx: i32,
    vy: i32,
    vz: i32,
    voxels: &[UVec4],
    grid_size: u32,
) -> Vec3 {
    let gs = grid_size as i32;
    let mut light = Vec3::new(0.0, 0.0, 0.0);
    let search = 2i32; // check +-2 cells
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
                        // Lava found — add warm orange-red light with distance falloff
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        let attenuation = 1.0 / (1.0 + dist_sq * 0.5);
                        light = Vec3::new(
                            light.x + 1.0 * attenuation * 0.5,
                            light.y + 0.25 * attenuation * 0.5,
                            light.z + 0.05 * attenuation * 0.5,
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
