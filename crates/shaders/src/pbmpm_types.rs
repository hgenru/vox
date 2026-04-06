//! GPU-side type definitions for the PB-MPM (Position-Based MPM) simulation.
//!
//! These types are SEPARATE from the CA voxel types and the existing MLS-MPM types.
//! PB-MPM handles on-demand physics zones (explosions, fluid dynamics, structural
//! collapse) and is iterated 2-4 times per frame for unconditional stability.
//!
//! All buffer access uses `&[u32]` / `&mut [u32]` with manual f32-to-bits conversion
//! to enable float atomics in P2G and to match the `ca_types.rs` accessor pattern.
//!
//! Particle layout: 96 bytes (24 u32s)
//!   [0..3]   pos_mass:  Vec4 (xyz=position, w=mass)
//!   [4..7]   vel_temp:  Vec4 (xyz=velocity, w=temperature)
//!   [8..11]  F_col0:    Vec4 (deformation gradient column 0 + pad)
//!   [12..15] F_col1:    Vec4 (deformation gradient column 1 + pad)
//!   [16..19] F_col2:    Vec4 (deformation gradient column 2 + pad)
//!   [20..23] ids:       UVec4 (x=material_id, y=phase, z=flags, w=pad)
//!
//! GridCell layout: 32 bytes (8 u32s)
//!   [0..3]   velocity_mass: Vec4 (xyz=momentum, w=mass)
//!   [4..7]   force_pad:     Vec4 (xyz=force, w=pad)

// ---- Constants ----

/// Maximum number of PB-MPM particles across all zones (256K).
pub const PBMPM_MAX_PARTICLES: u32 = 262144;

/// Maximum number of PB-MPM grid cells across all zones (256K).
pub const PBMPM_MAX_GRID_CELLS: u32 = 262144;

/// Size of one grid cell in u32 units (32 bytes / 4).
pub const PBMPM_GRID_CELL_SIZE_U32: u32 = 8;

/// Size of one particle in u32 units (96 bytes / 4).
pub const PBMPM_PARTICLE_SIZE_U32: u32 = 24;

// ---- Particle field offsets (in u32 units from particle start) ----

/// Offset to pos_mass.x within a particle.
const P_POS_X: u32 = 0;
/// Offset to pos_mass.y within a particle.
const P_POS_Y: u32 = 1;
/// Offset to pos_mass.z within a particle.
const P_POS_Z: u32 = 2;
/// Offset to pos_mass.w (mass) within a particle.
const P_MASS: u32 = 3;
/// Offset to vel_temp.x within a particle.
const P_VEL_X: u32 = 4;
/// Offset to vel_temp.y within a particle.
const P_VEL_Y: u32 = 5;
/// Offset to vel_temp.z within a particle.
const P_VEL_Z: u32 = 6;
/// Offset to vel_temp.w (temperature) within a particle.
const P_TEMP: u32 = 7;
/// Offset to F_col0.x within a particle.
const P_F0_X: u32 = 8;
/// Offset to F_col0.y within a particle.
const P_F0_Y: u32 = 9;
/// Offset to F_col0.z within a particle.
const P_F0_Z: u32 = 10;
/// Offset to F_col1.x within a particle.
const P_F1_X: u32 = 12;
/// Offset to F_col1.y within a particle.
const P_F1_Y: u32 = 13;
/// Offset to F_col1.z within a particle.
const P_F1_Z: u32 = 14;
/// Offset to F_col2.x within a particle.
const P_F2_X: u32 = 16;
/// Offset to F_col2.y within a particle.
const P_F2_Y: u32 = 17;
/// Offset to F_col2.z within a particle.
const P_F2_Z: u32 = 18;
/// Offset to ids.x (material_id) within a particle.
const P_MAT_ID: u32 = 20;
/// Offset to ids.y (phase) within a particle.
const P_PHASE: u32 = 21;
/// Offset to ids.z (flags) within a particle.
const P_FLAGS: u32 = 22;

// ---- Grid cell field offsets (in u32 units from cell start) ----

/// Offset to velocity_mass.x (momentum x) within a grid cell.
const G_MOM_X: u32 = 0;
/// Offset to velocity_mass.y (momentum y) within a grid cell.
const G_MOM_Y: u32 = 1;
/// Offset to velocity_mass.z (momentum z) within a grid cell.
const G_MOM_Z: u32 = 2;
/// Offset to velocity_mass.w (mass) within a grid cell.
const G_MASS: u32 = 3;
/// Offset to force_pad.x within a grid cell.
const G_FORCE_X: u32 = 4;
/// Offset to force_pad.y within a grid cell.
const G_FORCE_Y: u32 = 5;
/// Offset to force_pad.z within a grid cell.
const G_FORCE_Z: u32 = 6;

// ---- Particle accessors ----

/// Reads a f32 from the particle buffer at the given particle index and field offset.
#[inline]
fn read_particle_f32(particles: &[u32], idx: u32, field: u32) -> f32 {
    let offset = idx * PBMPM_PARTICLE_SIZE_U32 + field;
    f32::from_bits(particles[offset as usize])
}

/// Writes a f32 to the particle buffer at the given particle index and field offset.
#[inline]
fn write_particle_f32(particles: &mut [u32], idx: u32, field: u32, val: f32) {
    let offset = idx * PBMPM_PARTICLE_SIZE_U32 + field;
    particles[offset as usize] = val.to_bits();
}

/// Reads a u32 from the particle buffer at the given particle index and field offset.
#[inline]
fn read_particle_u32(particles: &[u32], idx: u32, field: u32) -> u32 {
    let offset = idx * PBMPM_PARTICLE_SIZE_U32 + field;
    particles[offset as usize]
}

/// Returns the particle's X position.
pub fn particle_pos_x(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_POS_X)
}

/// Returns the particle's Y position.
pub fn particle_pos_y(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_POS_Y)
}

/// Returns the particle's Z position.
pub fn particle_pos_z(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_POS_Z)
}

/// Returns the particle's mass.
pub fn particle_mass(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_MASS)
}

/// Returns the particle's X velocity.
pub fn particle_vel_x(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_VEL_X)
}

/// Returns the particle's Y velocity.
pub fn particle_vel_y(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_VEL_Y)
}

/// Returns the particle's Z velocity.
pub fn particle_vel_z(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_VEL_Z)
}

/// Returns the particle's temperature.
pub fn particle_temp(particles: &[u32], idx: u32) -> f32 {
    read_particle_f32(particles, idx, P_TEMP)
}

/// Returns the particle's material ID.
pub fn particle_material_id(particles: &[u32], idx: u32) -> u32 {
    read_particle_u32(particles, idx, P_MAT_ID)
}

/// Returns the particle's phase (0=solid, 1=liquid, 2=gas).
pub fn particle_phase(particles: &[u32], idx: u32) -> u32 {
    read_particle_u32(particles, idx, P_PHASE)
}

/// Returns the particle's flags.
pub fn particle_flags(particles: &[u32], idx: u32) -> u32 {
    read_particle_u32(particles, idx, P_FLAGS)
}

/// Sets the particle's position (xyz).
pub fn set_particle_pos(particles: &mut [u32], idx: u32, x: f32, y: f32, z: f32) {
    write_particle_f32(particles, idx, P_POS_X, x);
    write_particle_f32(particles, idx, P_POS_Y, y);
    write_particle_f32(particles, idx, P_POS_Z, z);
}

/// Sets the particle's velocity (xyz).
pub fn set_particle_vel(particles: &mut [u32], idx: u32, x: f32, y: f32, z: f32) {
    write_particle_f32(particles, idx, P_VEL_X, x);
    write_particle_f32(particles, idx, P_VEL_Y, y);
    write_particle_f32(particles, idx, P_VEL_Z, z);
}

/// Sets the particle's temperature.
pub fn set_particle_temp(particles: &mut [u32], idx: u32, temp: f32) {
    write_particle_f32(particles, idx, P_TEMP, temp);
}

/// Sets the particle's mass.
pub fn set_particle_mass(particles: &mut [u32], idx: u32, mass: f32) {
    write_particle_f32(particles, idx, P_MASS, mass);
}

/// Sets the deformation gradient to identity (F = I).
///
/// Used on initialization and after phase transitions (trap #8).
pub fn set_particle_f_identity(particles: &mut [u32], idx: u32) {
    // F_col0 = (1, 0, 0, 0)
    write_particle_f32(particles, idx, P_F0_X, 1.0);
    write_particle_f32(particles, idx, P_F0_Y, 0.0);
    write_particle_f32(particles, idx, P_F0_Z, 0.0);
    write_particle_f32(particles, idx, P_F0_X + 3, 0.0); // pad
    // F_col1 = (0, 1, 0, 0)
    write_particle_f32(particles, idx, P_F1_X, 0.0);
    write_particle_f32(particles, idx, P_F1_Y, 1.0);
    write_particle_f32(particles, idx, P_F1_Z, 0.0);
    write_particle_f32(particles, idx, P_F1_X + 3, 0.0); // pad
    // F_col2 = (0, 0, 1, 0)
    write_particle_f32(particles, idx, P_F2_X, 0.0);
    write_particle_f32(particles, idx, P_F2_Y, 0.0);
    write_particle_f32(particles, idx, P_F2_Z, 1.0);
    write_particle_f32(particles, idx, P_F2_X + 3, 0.0); // pad
}

// ---- APIC C matrix accessors ----
//
// The APIC affine momentum matrix C is stored in the F_col fields.
// For PB-MPM POC we don't use elastic deformation gradient, so F_col0/1/2
// are repurposed as C_col0/1/2.
//
// C is a 3x3 matrix stored column-major:
//   C_col0 = (C00, C10, C20) at offsets P_F0_X, P_F0_Y, P_F0_Z
//   C_col1 = (C01, C11, C21) at offsets P_F1_X, P_F1_Y, P_F1_Z
//   C_col2 = (C02, C12, C22) at offsets P_F2_X, P_F2_Y, P_F2_Z

/// Reads C matrix element at (row, col) from particle.
///
/// C is the APIC affine momentum matrix, stored in F_col fields.
/// Row/col must be 0, 1, or 2.
pub fn particle_c_element(particles: &[u32], idx: u32, row: u32, col: u32) -> f32 {
    let col_offset = match col {
        0 => P_F0_X,
        1 => P_F1_X,
        _ => P_F2_X,
    };
    read_particle_f32(particles, idx, col_offset + row)
}

/// Reads C matrix column 0: (C00, C10, C20).
pub fn particle_c_col0(particles: &[u32], idx: u32) -> [f32; 3] {
    [
        read_particle_f32(particles, idx, P_F0_X),
        read_particle_f32(particles, idx, P_F0_Y),
        read_particle_f32(particles, idx, P_F0_Z),
    ]
}

/// Reads C matrix column 1: (C01, C11, C21).
pub fn particle_c_col1(particles: &[u32], idx: u32) -> [f32; 3] {
    [
        read_particle_f32(particles, idx, P_F1_X),
        read_particle_f32(particles, idx, P_F1_Y),
        read_particle_f32(particles, idx, P_F1_Z),
    ]
}

/// Reads C matrix column 2: (C02, C12, C22).
pub fn particle_c_col2(particles: &[u32], idx: u32) -> [f32; 3] {
    [
        read_particle_f32(particles, idx, P_F2_X),
        read_particle_f32(particles, idx, P_F2_Y),
        read_particle_f32(particles, idx, P_F2_Z),
    ]
}

/// Sets the full C matrix (3x3, column-major) for a particle.
///
/// `c` is laid out as [c00, c10, c20, c01, c11, c21, c02, c12, c22].
pub fn set_particle_c_matrix(particles: &mut [u32], idx: u32, c: &[f32; 9]) {
    // Column 0
    write_particle_f32(particles, idx, P_F0_X, c[0]);
    write_particle_f32(particles, idx, P_F0_Y, c[1]);
    write_particle_f32(particles, idx, P_F0_Z, c[2]);
    // Column 1
    write_particle_f32(particles, idx, P_F1_X, c[3]);
    write_particle_f32(particles, idx, P_F1_Y, c[4]);
    write_particle_f32(particles, idx, P_F1_Z, c[5]);
    // Column 2
    write_particle_f32(particles, idx, P_F2_X, c[6]);
    write_particle_f32(particles, idx, P_F2_Y, c[7]);
    write_particle_f32(particles, idx, P_F2_Z, c[8]);
}

/// Zeros the APIC C matrix for a particle.
///
/// Used on initialization when no affine momentum is present.
pub fn set_particle_c_zero(particles: &mut [u32], idx: u32) {
    set_particle_c_matrix(particles, idx, &[0.0; 9]);
}

/// Computes C * offset, where C is the particle's APIC affine matrix and
/// offset is a 3D vector. Returns [result_x, result_y, result_z].
///
/// C is column-major: result = col0*ox + col1*oy + col2*oz.
pub fn apply_c_matrix(particles: &[u32], idx: u32, ox: f32, oy: f32, oz: f32) -> [f32; 3] {
    let c0 = particle_c_col0(particles, idx);
    let c1 = particle_c_col1(particles, idx);
    let c2 = particle_c_col2(particles, idx);
    [
        c0[0] * ox + c1[0] * oy + c2[0] * oz,
        c0[1] * ox + c1[1] * oy + c2[1] * oz,
        c0[2] * ox + c1[2] * oy + c2[2] * oz,
    ]
}

// ---- Grid cell accessors ----

/// Reads a f32 from the grid buffer at the given cell index and field offset.
#[inline]
fn read_grid_f32(grid: &[u32], cell_idx: u32, field: u32) -> f32 {
    let offset = cell_idx * PBMPM_GRID_CELL_SIZE_U32 + field;
    f32::from_bits(grid[offset as usize])
}

/// Writes a f32 to the grid buffer at the given cell index and field offset.
#[inline]
fn write_grid_f32(grid: &mut [u32], cell_idx: u32, field: u32, val: f32) {
    let offset = cell_idx * PBMPM_GRID_CELL_SIZE_U32 + field;
    grid[offset as usize] = val.to_bits();
}

/// Returns the grid cell's momentum X component.
pub fn grid_momentum_x(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_X)
}

/// Returns the grid cell's momentum Y component.
pub fn grid_momentum_y(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_Y)
}

/// Returns the grid cell's momentum Z component.
pub fn grid_momentum_z(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_Z)
}

/// Returns the grid cell's accumulated mass.
pub fn grid_mass(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MASS)
}

/// Returns the grid cell's velocity X (after normalization by mass).
pub fn grid_velocity_x(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_X)
}

/// Returns the grid cell's velocity Y (after normalization by mass).
pub fn grid_velocity_y(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_Y)
}

/// Returns the grid cell's velocity Z (after normalization by mass).
pub fn grid_velocity_z(grid: &[u32], cell_idx: u32) -> f32 {
    read_grid_f32(grid, cell_idx, G_MOM_Z)
}

/// Sets the grid cell's velocity (xyz) and mass (w).
pub fn set_grid_velocity_mass(grid: &mut [u32], cell_idx: u32, vx: f32, vy: f32, vz: f32, mass: f32) {
    write_grid_f32(grid, cell_idx, G_MOM_X, vx);
    write_grid_f32(grid, cell_idx, G_MOM_Y, vy);
    write_grid_f32(grid, cell_idx, G_MOM_Z, vz);
    write_grid_f32(grid, cell_idx, G_MASS, mass);
}

/// Sets the grid cell's force (xyz).
pub fn set_grid_force(grid: &mut [u32], cell_idx: u32, fx: f32, fy: f32, fz: f32) {
    write_grid_f32(grid, cell_idx, G_FORCE_X, fx);
    write_grid_f32(grid, cell_idx, G_FORCE_Y, fy);
    write_grid_f32(grid, cell_idx, G_FORCE_Z, fz);
}

// ---- Grid cell index helpers ----

/// Converts 3D grid coordinates to a flat cell index.
///
/// Returns the 1D index for the cell at (x, y, z) in a grid of `grid_size` cells per axis.
pub fn grid_cell_index(x: u32, y: u32, z: u32, grid_size: u32) -> u32 {
    z * grid_size * grid_size + y * grid_size + x
}

/// Atomically adds a float value to a grid buffer bound as `&mut [f32]`.
///
/// Uses `spirv_std::arch::atomic_f_add` on SPIR-V targets.
/// Falls back to non-atomic addition on CPU (for testing).
///
/// The P2G pass binds the grid as `&mut [f32]` (same physical buffer)
/// to enable float atomics without pointer casts, matching the existing
/// MLS-MPM P2G pattern.
///
/// # Safety
/// The caller must ensure `grid[index]` is a valid location and that
/// concurrent atomic access is properly synchronized via SPIR-V semantics.
#[inline]
pub unsafe fn grid_atomic_add_f32(grid: &mut [f32], index: usize, value: f32) {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::atomic_f_add::<f32, 1u32, 0x0u32>(&mut grid[index], value);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        grid[index] += value;
    }
}
