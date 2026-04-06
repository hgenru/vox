//! Chunk-aware render shader: ray-marches directly through the chunk pool gigabuffer.
//!
//! Instead of copying CA voxels into a flat buffer, this shader reads voxels
//! directly from the chunk pool using a chunk table for world-to-slot mapping.
//! This removes the 128^3 world size limit and eliminates the flat-buffer copy.
//!
//! Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
//! Each thread computes one pixel via DDA ray marching through the chunked world.

use crate::ca_types;
use crate::types::MaterialParams;
use spirv_std::glam::{UVec3, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

use super::render::lighting::{apply_lighting, sun_direction};
use super::render::materials::{apply_emissive, textured_material_color};
use super::render::math::{compute_ray, dot, normalize_vec3, pack_bgra};
use super::render::sky::compute_sky_color;

/// Push constants for the chunk-aware CA render shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaRenderPush {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Total world size in voxels (X axis = chunks_x * 32).
    pub world_size_x: u32,
    /// Total world size in voxels (Y axis = chunks_y * 32).
    pub world_size_y: u32,
    /// Total world size in voxels (Z axis = chunks_z * 32).
    pub world_size_z: u32,
    /// Number of chunks along X axis.
    pub chunks_x: u32,
    /// Number of chunks along Y axis.
    pub chunks_y: u32,
    /// Number of chunks along Z axis.
    pub chunks_z: u32,
    /// Camera eye position (xyz) + padding (w).
    pub eye: Vec4,
    /// Camera target position (xyz) + padding (w).
    pub target: Vec4,
}

// ---------------------------------------------------------------------------
// CA voxel field extraction (mirroring ca_to_render.rs constants)
// ---------------------------------------------------------------------------

/// Bit mask for material ID (bits 0-9).
const MATERIAL_ID_MASK: u32 = 0x3FF;

/// Bit shift for temperature field.
const TEMPERATURE_SHIFT: u32 = 10;

/// Bit mask for temperature (bits 10-17, 8 bits).
const TEMPERATURE_MASK: u32 = 0xFF;

/// Extract material_id from a packed CA voxel.
fn ca_material_id(voxel: u32) -> u32 {
    voxel & MATERIAL_ID_MASK
}

/// Extract temperature from a packed CA voxel.
fn ca_temperature(voxel: u32) -> u32 {
    (voxel >> TEMPERATURE_SHIFT) & TEMPERATURE_MASK
}

/// Remap CA material ID to render material ID.
///
/// CA IDs: 0=air, 1=stone, 2=sand, 3=water, 4=lava, 5=steam, 6=ice
/// Render: CA 4 (lava) -> render 2 (MAT_LAVA), CA 2 (sand) -> render 4
fn remap_material_id(ca_mat: u32) -> u32 {
    if ca_mat == 4 {
        2
    } else if ca_mat == 2 {
        4
    } else {
        ca_mat
    }
}

/// Remap CA phase to render phase.
///
/// CA:     0=solid, 1=powder, 2=liquid, 3=gas
/// Render: 0=solid, 1=liquid, 2=gas
fn remap_phase(ca_phase: u32) -> u32 {
    if ca_phase <= 1 {
        0
    } else if ca_phase == 2 {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// World voxel access via chunk table
// ---------------------------------------------------------------------------

/// Read a voxel from the chunked world at world coordinates (wx, wy, wz).
///
/// Returns 0 (air) if the position is out of bounds or the chunk is not loaded.
fn read_world_voxel(
    chunk_pool: &[u32],
    chunk_table: &[u32],
    wx: i32,
    wy: i32,
    wz: i32,
    chunks_x: u32,
    chunks_y: u32,
    chunks_z: u32,
    world_size_x: u32,
    world_size_y: u32,
    world_size_z: u32,
) -> u32 {
    // Bounds check — return air for out-of-world positions
    if wx < 0 || wy < 0 || wz < 0 {
        return 0;
    }
    let wx = wx as u32;
    let wy = wy as u32;
    let wz = wz as u32;
    if wx >= world_size_x || wy >= world_size_y || wz >= world_size_z {
        return 0;
    }

    // Chunk coordinate
    let cx = wx / 32;
    let cy = wy / 32;
    let cz = wz / 32;

    // Lookup slot from chunk table
    let table_idx = (cz * chunks_y * chunks_x + cy * chunks_x + cx) as usize;
    let slot_id = chunk_table[table_idx];
    if slot_id == 0xFFFFFFFF {
        return 0; // chunk not loaded
    }

    // Local coord within chunk
    let lx = wx % 32;
    let ly = wy % 32;
    let lz = wz % 32;

    // Read from gigabuffer
    let addr = ca_types::voxel_addr(slot_id, lx, ly, lz);
    chunk_pool[addr as usize]
}

/// Check if a world voxel is non-air (occupied).
fn is_occupied(
    chunk_pool: &[u32],
    chunk_table: &[u32],
    wx: i32,
    wy: i32,
    wz: i32,
    chunks_x: u32,
    chunks_y: u32,
    chunks_z: u32,
    world_size_x: u32,
    world_size_y: u32,
    world_size_z: u32,
) -> bool {
    let v = read_world_voxel(
        chunk_pool,
        chunk_table,
        wx,
        wy,
        wz,
        chunks_x,
        chunks_y,
        chunks_z,
        world_size_x,
        world_size_y,
        world_size_z,
    );
    ca_material_id(v) != 0
}

// ---------------------------------------------------------------------------
// Ambient occlusion for chunk world
// ---------------------------------------------------------------------------

/// Compute ambient occlusion by sampling neighbors around the hit voxel.
///
/// Checks 8 neighbors (4 edge + 4 corner) in the plane perpendicular to the
/// hit face normal.
fn compute_ao_chunked(
    vx: i32,
    vy: i32,
    vz: i32,
    normal: Vec3,
    chunk_pool: &[u32],
    chunk_table: &[u32],
    chunks_x: u32,
    chunks_y: u32,
    chunks_z: u32,
    world_size_x: u32,
    world_size_y: u32,
    world_size_z: u32,
) -> f32 {
    // Determine two tangent axes perpendicular to the hit face
    let (t1x, t1y, t1z, t2x, t2y, t2z) = if normal.x.abs() > 0.5 {
        (0i32, 1i32, 0i32, 0i32, 0i32, 1i32)
    } else if normal.y.abs() > 0.5 {
        (1, 0, 0, 0, 0, 1)
    } else {
        (1, 0, 0, 0, 1, 0)
    };

    let nx = if normal.x > 0.5 { 1i32 } else if normal.x < -0.5 { -1i32 } else { 0i32 };
    let ny = if normal.y > 0.5 { 1i32 } else if normal.y < -0.5 { -1i32 } else { 0i32 };
    let nz = if normal.z > 0.5 { 1i32 } else if normal.z < -0.5 { -1i32 } else { 0i32 };

    let mut occlusion = 0.0_f32;

    // 4 edge neighbors
    if is_occupied(chunk_pool, chunk_table, vx + nx + t1x, vy + ny + t1y, vz + nz + t1z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.12; }
    if is_occupied(chunk_pool, chunk_table, vx + nx - t1x, vy + ny - t1y, vz + nz - t1z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.12; }
    if is_occupied(chunk_pool, chunk_table, vx + nx + t2x, vy + ny + t2y, vz + nz + t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.12; }
    if is_occupied(chunk_pool, chunk_table, vx + nx - t2x, vy + ny - t2y, vz + nz - t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.12; }

    // 4 corner neighbors
    if is_occupied(chunk_pool, chunk_table, vx + nx + t1x + t2x, vy + ny + t1y + t2y, vz + nz + t1z + t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.08; }
    if is_occupied(chunk_pool, chunk_table, vx + nx + t1x - t2x, vy + ny + t1y - t2y, vz + nz + t1z - t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.08; }
    if is_occupied(chunk_pool, chunk_table, vx + nx - t1x + t2x, vy + ny - t1y + t2y, vz + nz - t1z + t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.08; }
    if is_occupied(chunk_pool, chunk_table, vx + nx - t1x - t2x, vy + ny - t1y - t2y, vz + nz - t1z - t2z, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) { occlusion += 0.08; }

    if occlusion > 0.6 {
        occlusion = 0.6;
    }
    1.0 - occlusion
}

// ---------------------------------------------------------------------------
// Shadow ray marching
// ---------------------------------------------------------------------------

/// March a shadow ray through the chunked world using DDA.
///
/// Returns `true` if any non-air voxel is hit.
fn march_shadow_chunked(
    start: Vec3,
    dir: Vec3,
    chunk_pool: &[u32],
    chunk_table: &[u32],
    chunks_x: u32,
    chunks_y: u32,
    chunks_z: u32,
    world_size_x: u32,
    world_size_y: u32,
    world_size_z: u32,
) -> bool {
    let gs_x = world_size_x as i32;
    let gs_y = world_size_y as i32;
    let gs_z = world_size_z as i32;

    let mut vx = start.x as i32;
    let mut vy = start.y as i32;
    let mut vz = start.z as i32;

    if vx < 0 { vx = 0; }
    if vy < 0 { vy = 0; }
    if vz < 0 { vz = 0; }
    if vx >= gs_x { vx = gs_x - 1; }
    if vy >= gs_y { vy = gs_y - 1; }
    if vz >= gs_z { vz = gs_z - 1; }

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

    // Use a reasonable max step count based on max world dimension
    let max_dim = world_size_x.max(world_size_y).max(world_size_z);
    let max_steps = max_dim * 2;
    let mut i: u32 = 0;

    // Skip start voxel
    if t_max_x < t_max_y {
        if t_max_x < t_max_z { vx += step_x; t_max_x += t_delta_x; }
        else { vz += step_z; t_max_z += t_delta_z; }
    } else if t_max_y < t_max_z {
        vy += step_y; t_max_y += t_delta_y;
    } else {
        vz += step_z; t_max_z += t_delta_z;
    }
    i += 1;

    while i < max_steps {
        if vx < 0 || vx >= gs_x || vy < 0 || vy >= gs_y || vz < 0 || vz >= gs_z {
            return false;
        }

        if is_occupied(chunk_pool, chunk_table, vx, vy, vz, chunks_x, chunks_y, chunks_z, world_size_x, world_size_y, world_size_z) {
            return true;
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z { vx += step_x; t_max_x += t_delta_x; }
            else { vz += step_z; t_max_z += t_delta_z; }
        } else if t_max_y < t_max_z {
            vy += step_y; t_max_y += t_delta_y;
        } else {
            vz += step_z; t_max_z += t_delta_z;
        }

        i += 1;
    }

    false
}

// ---------------------------------------------------------------------------
// Lava point lighting
// ---------------------------------------------------------------------------

/// Compute point light contribution from nearby lava voxels.
///
/// Searches a 5x5x5 neighborhood around the hit voxel for lava (CA mat_id=4).
fn compute_lava_light_chunked(
    vx: i32,
    vy: i32,
    vz: i32,
    chunk_pool: &[u32],
    chunk_table: &[u32],
    chunks_x: u32,
    chunks_y: u32,
    chunks_z: u32,
    world_size_x: u32,
    world_size_y: u32,
    world_size_z: u32,
) -> Vec3 {
    let mut light = Vec3::new(0.0, 0.0, 0.0);
    let search = 2i32;
    let mut dx = -search;
    while dx <= search {
        let mut dy = -search;
        while dy <= search {
            let mut dz = -search;
            while dz <= search {
                let nx = vx + dx;
                let ny = vy + dy;
                let nz = vz + dz;
                let v = read_world_voxel(
                    chunk_pool, chunk_table, nx, ny, nz,
                    chunks_x, chunks_y, chunks_z,
                    world_size_x, world_size_y, world_size_z,
                );
                // CA lava = material_id 4
                if ca_material_id(v) == 4 {
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    let attenuation = 1.0 / (1.0 + dist_sq * 0.5);
                    light = Vec3::new(
                        light.x + 1.0 * attenuation * 0.5,
                        light.y + 0.25 * attenuation * 0.5,
                        light.z + 0.05 * attenuation * 0.5,
                    );
                }
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }
    light
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

/// Shade a hit voxel: diffuse, AO, shadows, lava light, emissive.
///
/// Separated from the main ray march to manage function complexity (trap #15).
fn shade_voxel_chunked(
    vx: i32,
    vy: i32,
    vz: i32,
    last_axis: u32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    ca_voxel: u32,
    chunk_pool: &[u32],
    chunk_table: &[u32],
    ca_materials: &[u32],
    render_materials: &[MaterialParams],
    push: &CaRenderPush,
) -> Vec3 {
    let mat_id = ca_material_id(ca_voxel);
    let render_mat = remap_material_id(mat_id);

    // Get CA phase from material table
    let mat_table_base = (mat_id * 16) as usize;
    let ca_phase = if mat_table_base < ca_materials.len() {
        ca_materials[mat_table_base]
    } else {
        0
    };
    let render_phase = remap_phase(ca_phase);

    // Face normal from DDA step axis
    let normal = if last_axis == 0 {
        Vec3::new(-(step_x as f32), 0.0, 0.0)
    } else if last_axis == 1 {
        Vec3::new(0.0, -(step_y as f32), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -(step_z as f32))
    };

    // Base color: gas/steam gets white-ish, others from material table
    let base_color = if render_phase == 2 {
        Vec3::new(0.9, 0.9, 0.95)
    } else {
        textured_material_color(render_mat, vx, vy, vz, render_materials)
    };

    // AO
    let ao_factor = compute_ao_chunked(
        vx, vy, vz, normal,
        chunk_pool, chunk_table,
        push.chunks_x, push.chunks_y, push.chunks_z,
        push.world_size_x, push.world_size_y, push.world_size_z,
    );

    // Shadow ray
    let shadow_origin = Vec3::new(
        vx as f32 + 0.5 + normal.x * 0.5,
        vy as f32 + 0.5 + normal.y * 0.5,
        vz as f32 + 0.5 + normal.z * 0.5,
    );
    let in_shadow = march_shadow_chunked(
        shadow_origin, sun_direction(),
        chunk_pool, chunk_table,
        push.chunks_x, push.chunks_y, push.chunks_z,
        push.world_size_x, push.world_size_y, push.world_size_z,
    );

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

    // Lava point lighting
    let lava_light = compute_lava_light_chunked(
        vx, vy, vz,
        chunk_pool, chunk_table,
        push.chunks_x, push.chunks_y, push.chunks_z,
        push.world_size_x, push.world_size_y, push.world_size_z,
    );
    let lit_color = Vec3::new(
        lit_color.x + base_color.x * lava_light.x,
        lit_color.y + base_color.y * lava_light.y,
        lit_color.z + base_color.z * lava_light.z,
    );

    // Temperature-based emissive
    let temp = ca_temperature(ca_voxel);
    let temp_f32 = if mat_id == 4 {
        temp as f32 * 10.0 // lava: hot, will glow
    } else {
        0.0
    };
    apply_emissive(lit_color, render_mat, render_phase, temp_f32)
}

// ---------------------------------------------------------------------------
// AABB slab test (split per-axis to avoid >3 branches per function)
// ---------------------------------------------------------------------------

/// Slab test result for one axis.
struct SlabResult {
    /// Updated t_min.
    t_min: f32,
    /// Updated t_max.
    t_max: f32,
    /// Whether the ray misses (parallel outside slab).
    miss: bool,
}

/// Slab test for X axis.
fn slab_test_x(eye_x: f32, dir_x: f32, grid_max: f32, t_min: f32, t_max: f32) -> SlabResult {
    if dir_x.abs() > 1e-8 {
        let t1 = (0.0 - eye_x) / dir_x;
        let t2 = (grid_max - eye_x) / dir_x;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        let new_min = if ta > t_min { ta } else { t_min };
        let new_max = if tb < t_max { tb } else { t_max };
        SlabResult { t_min: new_min, t_max: new_max, miss: false }
    } else if eye_x < 0.0 || eye_x > grid_max {
        SlabResult { t_min, t_max, miss: true }
    } else {
        SlabResult { t_min, t_max, miss: false }
    }
}

/// Slab test for Y axis.
fn slab_test_y(eye_y: f32, dir_y: f32, grid_max: f32, t_min: f32, t_max: f32) -> SlabResult {
    if dir_y.abs() > 1e-8 {
        let t1 = (0.0 - eye_y) / dir_y;
        let t2 = (grid_max - eye_y) / dir_y;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        let new_min = if ta > t_min { ta } else { t_min };
        let new_max = if tb < t_max { tb } else { t_max };
        SlabResult { t_min: new_min, t_max: new_max, miss: false }
    } else if eye_y < 0.0 || eye_y > grid_max {
        SlabResult { t_min, t_max, miss: true }
    } else {
        SlabResult { t_min, t_max, miss: false }
    }
}

/// Slab test for Z axis.
fn slab_test_z(eye_z: f32, dir_z: f32, grid_max: f32, t_min: f32, t_max: f32) -> SlabResult {
    if dir_z.abs() > 1e-8 {
        let t1 = (0.0 - eye_z) / dir_z;
        let t2 = (grid_max - eye_z) / dir_z;
        let (ta, tb) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        let new_min = if ta > t_min { ta } else { t_min };
        let new_max = if tb < t_max { tb } else { t_max };
        SlabResult { t_min: new_min, t_max: new_max, miss: false }
    } else if eye_z < 0.0 || eye_z > grid_max {
        SlabResult { t_min, t_max, miss: true }
    } else {
        SlabResult { t_min, t_max, miss: false }
    }
}

// ---------------------------------------------------------------------------
// DDA ray march through chunked world
// ---------------------------------------------------------------------------

/// Background color packed as u32 (dark navy).
const SKY_BG: u32 = 0xFF281212; // pack_bgra(18, 18, 40, 255)

/// Perform DDA ray march through the chunked world and write pixel color.
///
/// This is the main per-pixel function, separated from the entry point
/// as required by the rust-gpu linker bug workaround (trap #4a).
fn render_pixel_chunked(
    px: u32,
    py: u32,
    push: &CaRenderPush,
    chunk_pool: &[u32],
    output: &mut [u32],
    ca_materials: &[u32],
    chunk_table: &[u32],
    render_materials: &[MaterialParams],
) {
    let width = push.width;
    let height = push.height;

    if px >= width || py >= height {
        return;
    }

    let eye = Vec3::new(push.eye.x, push.eye.y, push.eye.z);
    let target = Vec3::new(push.target.x, push.target.y, push.target.z);
    let ray_dir = compute_ray(px, py, width, height, eye, target);

    let grid_x = push.world_size_x as f32;
    let grid_y = push.world_size_y as f32;
    let grid_z = push.world_size_z as f32;

    // AABB slab test for world bounds [0, world_size]^3
    let mut t_min = 0.0_f32;
    let mut t_max_bound = 1e20_f32;

    let sx = slab_test_x(eye.x, ray_dir.x, grid_x, t_min, t_max_bound);
    if sx.miss { output[(py * width + px) as usize] = SKY_BG; return; }
    t_min = sx.t_min; t_max_bound = sx.t_max;

    let sy = slab_test_y(eye.y, ray_dir.y, grid_y, t_min, t_max_bound);
    if sy.miss { output[(py * width + px) as usize] = SKY_BG; return; }
    t_min = sy.t_min; t_max_bound = sy.t_max;

    let sz = slab_test_z(eye.z, ray_dir.z, grid_z, t_min, t_max_bound);
    if sz.miss { output[(py * width + px) as usize] = SKY_BG; return; }
    t_min = sz.t_min; t_max_bound = sz.t_max;

    if t_min > t_max_bound || t_max_bound < 0.0 {
        output[(py * width + px) as usize] = SKY_BG;
        return;
    }
    if t_min < 0.0 { t_min = 0.0; }

    // Entry point into the grid
    let entry = Vec3::new(
        eye.x + ray_dir.x * (t_min + 0.001),
        eye.y + ray_dir.y * (t_min + 0.001),
        eye.z + ray_dir.z * (t_min + 0.001),
    );

    let gs_x = push.world_size_x as i32;
    let gs_y = push.world_size_y as i32;
    let gs_z = push.world_size_z as i32;

    let mut vx = entry.x as i32;
    let mut vy = entry.y as i32;
    let mut vz = entry.z as i32;

    // Clamp to valid range
    if vx < 0 { vx = 0; }
    if vy < 0 { vy = 0; }
    if vz < 0 { vz = 0; }
    if vx >= gs_x { vx = gs_x - 1; }
    if vy >= gs_y { vy = gs_y - 1; }
    if vz >= gs_z { vz = gs_z - 1; }

    // Step direction
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

    let max_dim = push.world_size_x.max(push.world_size_y).max(push.world_size_z);
    let max_steps = max_dim * 3;
    let mut last_axis: u32 = 0;
    let mut i: u32 = 0;
    let mut hit = false;
    let mut hit_voxel: u32 = 0;

    // Water transparency accumulation
    let mut water_transmittance = 1.0_f32;
    let mut water_tint = Vec3::new(0.0, 0.0, 0.0);
    let water_opacity = 0.3_f32;
    let water_base = Vec3::new(0.2, 0.4, 0.8);

    while i < max_steps {
        if vx >= 0 && vx < gs_x && vy >= 0 && vy < gs_y && vz >= 0 && vz < gs_z {
            let voxel = read_world_voxel(
                chunk_pool, chunk_table, vx, vy, vz,
                push.chunks_x, push.chunks_y, push.chunks_z,
                push.world_size_x, push.world_size_y, push.world_size_z,
            );
            let mat_id = ca_material_id(voxel);
            if mat_id != 0 {
                // CA water = material_id 3
                if mat_id == 3 && water_transmittance > 0.05 {
                    let absorption = water_opacity * water_transmittance;
                    water_tint = Vec3::new(
                        water_tint.x + water_base.x * absorption,
                        water_tint.y + water_base.y * absorption,
                        water_tint.z + water_base.z * absorption,
                    );
                    water_transmittance *= 1.0 - water_opacity;
                } else {
                    hit = true;
                    hit_voxel = voxel;
                    break;
                }
            }
        } else {
            break;
        }

        // DDA step
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                vx += step_x; t_max_x += t_delta_x; last_axis = 0;
            } else {
                vz += step_z; t_max_z += t_delta_z; last_axis = 2;
            }
        } else if t_max_y < t_max_z {
            vy += step_y; t_max_y += t_delta_y; last_axis = 1;
        } else {
            vz += step_z; t_max_z += t_delta_z; last_axis = 2;
        }

        i += 1;
    }

    let pixel_idx = (py * width + px) as usize;

    let background = if hit {
        shade_voxel_chunked(
            vx, vy, vz, last_axis, step_x, step_y, step_z,
            hit_voxel,
            chunk_pool, chunk_table, ca_materials, render_materials, push,
        )
    } else {
        compute_sky_color(ray_dir)
    };

    // Blend water tint with background
    let final_color = Vec3::new(
        water_tint.x + background.x * water_transmittance,
        water_tint.y + background.y * water_transmittance,
        water_tint.z + background.z * water_transmittance,
    );

    let r = ((final_color.x * 255.0) as u32).min(255);
    let g = ((final_color.y * 255.0) as u32).min(255);
    let b = ((final_color.z * 255.0) as u32).min(255);

    output[pixel_idx] = pack_bgra(r, g, b, 255);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compute shader entry point: chunk-aware CA render via ray marching.
///
/// Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
///
/// Descriptor set 0:
/// - binding 0: chunk_pool gigabuffer (CA voxels, read)
/// - binding 1: render output pixel buffer (write)
/// - binding 2: CA material table (MaterialPropertiesCA as u32[], read)
/// - binding 3: chunk table (slot_id lookup, read)
/// - binding 4: material render buffer (MaterialParams for colors, read)
///
/// Push constants: [`CaRenderPush`].
#[spirv(compute(threads(8, 8, 1)))]
pub fn ca_render(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &CaRenderPush,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] chunk_pool: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ca_materials: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] chunk_table: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] render_materials: &[MaterialParams],
) {
    render_pixel_chunked(
        id.x,
        id.y,
        push,
        chunk_pool,
        output,
        ca_materials,
        chunk_table,
        render_materials,
    );
}
