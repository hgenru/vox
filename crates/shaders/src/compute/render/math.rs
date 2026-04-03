//! Math utilities for the render shader: vector ops, ray computation, color packing, noise.

use spirv_std::glam::Vec3;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Normalize a Vec3 without using the `length()` method (more portable for SPIR-V).
pub(crate) fn normalize_vec3(v: Vec3) -> Vec3 {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if len_sq < 1e-12 {
        return Vec3::new(0.0, 1.0, 0.0);
    }
    let inv_len = 1.0 / len_sq.sqrt();
    Vec3::new(v.x * inv_len, v.y * inv_len, v.z * inv_len)
}

/// Cross product of two Vec3s.
pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Dot product of two Vec3s.
pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Compute camera ray direction for a given pixel coordinate.
///
/// Returns a normalized ray direction using a simple perspective camera
/// with vertical FOV of ~60 degrees.
pub(crate) fn compute_ray(
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
pub(crate) fn pack_bgra(r: u32, g: u32, b: u32, a: u32) -> u32 {
    // For B8G8R8A8_SRGB on little-endian: bytes are B, G, R, A
    // But vkCmdCopyBufferToImage interprets buffer bytes linearly,
    // so for B8G8R8A8_SRGB we need: byte0=B, byte1=G, byte2=R, byte3=A
    // In u32 little-endian: 0xAARRGGBB
    b | (g << 8) | (r << 16) | (a << 24)
}

/// Simple 3D hash for procedural texturing.
///
/// Returns a deterministic pseudo-random value in [0, 1] for integer coordinates.
pub(crate) fn hash3(x: i32, y: i32, z: i32) -> f32 {
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
pub(crate) fn value_noise(x: f32, y: f32, z: f32) -> f32 {
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
