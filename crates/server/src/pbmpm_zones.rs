//! PB-MPM zone manager: activation, spawn/sleep lifecycle.
//!
//! Manages on-demand physics zones that activate when the CA substrate triggers
//! dynamic events (explosions, structural collapse, fluid bursts). Each zone
//! gets a region of the shared particle and grid buffers, and dispatches the
//! PB-MPM compute passes (clear_grid, P2G, grid_update, G2P) iteratively.
//!
//! Up to [`MAX_ZONES`] zones can be active simultaneously. Zones are deactivated
//! (slept) after particles settle below a velocity threshold for [`SLEEP_FRAMES`]
//! consecutive frames.

use std::mem;

use ash::vk;
use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
    pipeline,
};

use crate::passes::{self, ComputePass};

/// Compiled SPIR-V shader module bytes, included at compile time.
const SHADER_BYTES: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));

/// Maximum simultaneous PB-MPM zones.
const MAX_ZONES: usize = 4;

/// PB-MPM iterations per frame (PB-MPM iterates the full loop for stability).
/// 4 iterations provides better constraint convergence and more stable physics.
const PBMPM_ITERATIONS: u32 = 4;

/// Frames below velocity threshold before sleeping a zone.
const SLEEP_FRAMES: u32 = 30;

/// Maximum particles across all zones (256K).
const MAX_PARTICLES: u32 = 262_144;

/// Maximum grid cells across all zones (256K).
const MAX_GRID_CELLS: u32 = 262_144;

/// Bytes per PB-MPM particle (24 u32s = 96 bytes).
const PARTICLE_BYTES: u64 = 96;

/// Bytes per PB-MPM grid cell (8 u32s = 32 bytes).
const GRID_CELL_BYTES: u64 = 32;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// State of a PB-MPM zone slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZoneState {
    /// Slot is available for allocation.
    Free,
    /// Zone is actively simulated each frame.
    Active,
    /// Zone particles are below velocity threshold, counting down to sleep.
    Sleeping,
}

/// A single PB-MPM physics zone occupying a region of the shared buffers.
pub struct PbmpmZone {
    /// World-space origin of this zone (integer voxel coordinates).
    pub world_origin: [i32; 3],
    /// Grid dimension (cells per axis, e.g. 32 or 64).
    pub grid_size: u32,
    /// Number of live particles in this zone.
    pub particle_count: u32,
    /// Offset into the global particle buffer (in particles, not bytes).
    pub particle_offset: u32,
    /// Offset into the global grid buffer (in cells, not bytes).
    pub grid_offset: u32,
    /// Current zone lifecycle state.
    pub state: ZoneState,
    /// Number of consecutive frames all particles have been below threshold.
    pub frames_quiet: u32,
    /// Whether this zone is a rigid body (shape matching constraint).
    pub rigid_body: bool,
    /// Rigid body state tracker (only set when rigid_body == true).
    pub rigid_body_state: Option<crate::rigid_body_tracker::RigidBodyState>,
}

impl PbmpmZone {
    /// Create a new free zone slot.
    fn free() -> Self {
        Self {
            world_origin: [0; 3],
            grid_size: 0,
            particle_count: 0,
            particle_offset: 0,
            grid_offset: 0,
            state: ZoneState::Free,
            frames_quiet: 0,
            rigid_body: false,
            rigid_body_state: None,
        }
    }
}

/// Push constants matching the shader-side `PbmpmPush` (32 bytes).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PbmpmPush {
    /// Grid dimension: cells per axis.
    pub grid_size: u32,
    /// Total number of particles in this zone.
    pub particle_count: u32,
    /// Offset into the global particle buffer (in particles).
    pub particle_offset: u32,
    /// Offset into the global grid buffer (in cells).
    pub grid_offset: u32,
    /// Simulation timestep for this iteration.
    pub dt: f32,
    /// Gravity acceleration (grid-units/s^2).
    pub gravity: f32,
    /// Current PB-MPM iteration index (0..N).
    pub iteration: u32,
    /// Padding to 32-byte alignment.
    pub _pad: u32,
}

// SAFETY: #[repr(C)] struct with only primitive fields.
unsafe impl bytemuck::Zeroable for PbmpmPush {}
unsafe impl bytemuck::Pod for PbmpmPush {}

/// Activation trigger from the CA world.
pub struct ActivationTrigger {
    /// World position (center of the activation region).
    pub center: [f32; 3],
    /// Activation radius in voxels.
    pub radius: f32,
    /// Initial velocity impulse magnitude.
    pub impulse: f32,
}

/// PB-MPM zone manager. Allocates and dispatches physics zones.
pub struct PbmpmZoneManager {
    /// Zone slots (fixed-size array).
    zones: [PbmpmZone; MAX_ZONES],

    // GPU resources
    /// Shared particle storage for all zones.
    particle_buffer: GpuBuffer,
    /// Shared grid storage for all zones.
    grid_buffer: GpuBuffer,

    // Compute passes
    clear_grid_pass: ComputePass,
    p2g_pass: ComputePass,
    grid_update_pass: ComputePass,
    g2p_pass: ComputePass,

    // Shader module (kept alive for pipeline lifetime)
    shader_module: vk::ShaderModule,

    // Descriptor pool (owns all descriptor sets)
    descriptor_pool: vk::DescriptorPool,

    // Capacity tracking
    max_particles: u32,
    max_grid_cells: u32,
    next_particle_offset: u32,
    next_grid_offset: u32,

    // Cached device handle for recording commands
    device: ash::Device,
}

impl PbmpmZoneManager {
    /// Create a new PB-MPM zone manager with GPU resources.
    ///
    /// Allocates particle and grid buffers, creates compute pipelines for all
    /// four PB-MPM passes, and initializes all zone slots as [`ZoneState::Free`].
    pub fn new(ctx: &VulkanContext) -> anyhow::Result<Self> {
        tracing::info!(
            "Creating PbmpmZoneManager: max_particles={}, max_grid_cells={}, max_zones={}",
            MAX_PARTICLES,
            MAX_GRID_CELLS,
            MAX_ZONES,
        );

        // Create GPU buffers
        let particle_buf_size = MAX_PARTICLES as u64 * PARTICLE_BYTES;
        let grid_buf_size = MAX_GRID_CELLS as u64 * GRID_CELL_BYTES;

        let particle_buffer = buffer::create_device_local_buffer(
            ctx,
            particle_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "pbmpm-particle-buffer",
        )?;

        let grid_buffer = buffer::create_device_local_buffer(
            ctx,
            grid_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "pbmpm-grid-buffer",
        )?;

        // Load SPIR-V shader module
        let shader_module =
            pipeline::create_shader_module(ctx, SHADER_BYTES, "pbmpm-shaders")?;

        // Create descriptor pool
        // Total storage buffer bindings across all passes:
        // clear_grid: 1, p2g: 2, grid_update: 1, g2p: 2 = 6 total
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 8, // some headroom
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 4, "pbmpm-descriptor-pool")?;

        let push_size = mem::size_of::<PbmpmPush>() as u32;

        // clear_grid: binding 0 = grid_buffer
        let clear_grid_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::pbmpm_clear_grid::pbmpm_clear_grid",
            &[&grid_buffer],
            push_size,
            descriptor_pool,
            "pbmpm-clear-grid",
        )?;

        // p2g: binding 0 = particle_buffer, binding 1 = grid_buffer (as f32 for atomics)
        let p2g_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::pbmpm_p2g::pbmpm_p2g",
            &[&particle_buffer, &grid_buffer],
            push_size,
            descriptor_pool,
            "pbmpm-p2g",
        )?;

        // grid_update: binding 0 = grid_buffer
        let grid_update_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::pbmpm_grid_update::pbmpm_grid_update",
            &[&grid_buffer],
            push_size,
            descriptor_pool,
            "pbmpm-grid-update",
        )?;

        // g2p: binding 0 = particle_buffer, binding 1 = grid_buffer
        let g2p_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::pbmpm_g2p::pbmpm_g2p",
            &[&particle_buffer, &grid_buffer],
            push_size,
            descriptor_pool,
            "pbmpm-g2p",
        )?;

        // Initialize all zones as Free
        let zones = [
            PbmpmZone::free(),
            PbmpmZone::free(),
            PbmpmZone::free(),
            PbmpmZone::free(),
        ];

        Ok(Self {
            zones,
            particle_buffer,
            grid_buffer,
            clear_grid_pass,
            p2g_pass,
            grid_update_pass,
            g2p_pass,
            shader_module,
            descriptor_pool,
            max_particles: MAX_PARTICLES,
            max_grid_cells: MAX_GRID_CELLS,
            next_particle_offset: 0,
            next_grid_offset: 0,
            device: ctx.device.clone(),
        })
    }

    /// Try to activate a new PB-MPM zone. Returns zone index if successful.
    ///
    /// Finds a free slot, reserves particle/grid buffer space, and sets the
    /// zone to [`ZoneState::Active`]. Returns `None` if no free slots or
    /// insufficient buffer capacity.
    pub fn activate_zone(
        &mut self,
        trigger: &ActivationTrigger,
        grid_size: u32,
    ) -> Option<usize> {
        // Find a free zone slot
        let zone_idx = self.zones.iter().position(|z| z.state == ZoneState::Free)?;

        // Check capacity: max 1 particle per voxel
        let max_particles_for_zone = grid_size * grid_size * grid_size;
        let grid_cells = grid_size * grid_size * grid_size;

        if self.next_particle_offset + max_particles_for_zone > self.max_particles {
            tracing::warn!(
                "PB-MPM: insufficient particle capacity for zone (need {}, have {})",
                max_particles_for_zone,
                self.max_particles - self.next_particle_offset,
            );
            return None;
        }
        if self.next_grid_offset + grid_cells > self.max_grid_cells {
            tracing::warn!(
                "PB-MPM: insufficient grid capacity for zone (need {}, have {})",
                grid_cells,
                self.max_grid_cells - self.next_grid_offset,
            );
            return None;
        }

        let zone = &mut self.zones[zone_idx];
        zone.world_origin = [
            trigger.center[0] as i32 - grid_size as i32 / 2,
            trigger.center[1] as i32 - grid_size as i32 / 2,
            trigger.center[2] as i32 - grid_size as i32 / 2,
        ];
        zone.grid_size = grid_size;
        zone.particle_count = 0; // set during spawn
        zone.particle_offset = self.next_particle_offset;
        zone.grid_offset = self.next_grid_offset;
        zone.state = ZoneState::Active;
        zone.frames_quiet = 0;

        self.next_particle_offset += max_particles_for_zone;
        self.next_grid_offset += grid_cells;

        tracing::info!(
            "PB-MPM: activated zone {} at origin {:?}, grid_size={}, particle_offset={}, grid_offset={}",
            zone_idx,
            zone.world_origin,
            grid_size,
            zone.particle_offset,
            zone.grid_offset,
        );

        Some(zone_idx)
    }

    /// Spawn particles from voxel data for a zone.
    ///
    /// Iterates over the zone's grid region, converting non-air voxels into
    /// PB-MPM particles with radial impulse from the trigger center.
    /// Returns the actual number of particles spawned.
    pub fn spawn_particles_from_voxels(
        &mut self,
        ctx: &VulkanContext,
        zone_idx: usize,
        voxel_data: &[u32],
        _chunk_origin: [i32; 3],
        trigger: &ActivationTrigger,
    ) -> anyhow::Result<u32> {
        use shared::voxel::Voxel;

        let zone = &self.zones[zone_idx];
        let grid_size = zone.grid_size;
        let mut particles: Vec<f32> = Vec::new();
        let mut count = 0u32;

        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let voxel_idx = (z * 32 * 32 + y * 32 + x) as usize;
                    if voxel_idx >= voxel_data.len() {
                        continue;
                    }

                    let v = Voxel(voxel_data[voxel_idx]);
                    if v.is_air() {
                        continue;
                    }

                    // Position: center of voxel in grid coordinates
                    let px = x as f32 + 0.5;
                    let py = y as f32 + 0.5;
                    let pz = z as f32 + 0.5;

                    // Velocity: radial impulse from trigger center, scaled by distance
                    let dx = px - (trigger.center[0] - zone.world_origin[0] as f32);
                    let dy = py - (trigger.center[1] - zone.world_origin[1] as f32);
                    let dz = pz - (trigger.center[2] - zone.world_origin[2] as f32);
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);

                    // Inverse-square falloff: stronger near center, weaker far away
                    // Normalize direction, then scale by impulse / (1 + dist^2)
                    let inv_dist = 1.0 / dist;
                    let dir_x = dx * inv_dist;
                    let dir_y = dy * inv_dist;
                    let dir_z = dz * inv_dist;
                    let falloff = trigger.impulse / (1.0 + dist * dist * 0.1);

                    // Hash-based per-particle speed variation (+-20%)
                    let hash = ((x.wrapping_mul(73856093))
                        ^ (y.wrapping_mul(19349663))
                        ^ (z.wrapping_mul(83492791)))
                        & 0xFF;
                    let variation = 0.8 + 0.4 * (hash as f32 / 255.0);

                    let speed = falloff * variation;
                    let vx = dir_x * speed;
                    // Upward bias: add 50% of impulse magnitude as vertical boost
                    let vy = dir_y * speed + trigger.impulse * 0.5;
                    let vz = dir_z * speed;

                    let mass = 1.0_f32;
                    let temp = v.temperature() as f32;

                    // Write particle: 24 f32s (96 bytes)
                    // pos_mass
                    particles.extend_from_slice(&[px, py, pz, mass]);
                    // vel_temp
                    particles.extend_from_slice(&[vx, vy, vz, temp]);
                    // C_col0: zero (APIC affine momentum matrix, no initial rotation)
                    particles.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                    // C_col1: zero
                    particles.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                    // C_col2: zero
                    particles.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                    // ids: material_id, phase, flags, padding (as u32 bits)
                    let mat_id = v.material_id() as u32;
                    let phase = 1u32; // treat as liquid for PB-MPM
                    particles.extend_from_slice(&[
                        f32::from_bits(mat_id),
                        f32::from_bits(phase),
                        f32::from_bits(0u32),
                        f32::from_bits(0u32),
                    ]);

                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(0);
        }

        // Upload particle data at the correct offset in particle buffer
        let byte_offset = zone.particle_offset as u64 * PARTICLE_BYTES;
        let byte_size = count as u64 * PARTICLE_BYTES;

        let particle_bytes: &[u8] = bytemuck::cast_slice(&particles);
        let mut staging =
            buffer::create_upload_staging_buffer(ctx, byte_size, "pbmpm-particle-staging")?;

        {
            let mapped = staging
                .mapped_slice_mut()
                .ok_or_else(|| anyhow::anyhow!("failed to map staging buffer"))?;
            mapped[..particle_bytes.len()].copy_from_slice(particle_bytes);
        }

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(byte_offset)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    self.particle_buffer.buffer,
                    &[region],
                );
            }
        })?;

        buffer::destroy_buffer(ctx, staging);
        self.zones[zone_idx].particle_count = count;

        tracing::info!(
            "PB-MPM: spawned {} particles for zone {} (byte_offset={})",
            count,
            zone_idx,
            byte_offset,
        );

        Ok(count)
    }

    /// Record PB-MPM compute dispatches for all active zones into the command buffer.
    ///
    /// For each active zone with particles, dispatches `PBMPM_ITERATIONS` rounds of:
    /// clear_grid -> P2G -> grid_update -> G2P, with barriers between each pass.
    pub fn step(&self, cmd: vk::CommandBuffer, _ctx: &VulkanContext, dt: f32) {
        let gravity = -196.0_f32; // scaled for voxel size (trap #17)

        for zone_idx in 0..MAX_ZONES {
            let zone = &self.zones[zone_idx];
            if zone.state != ZoneState::Active {
                continue;
            }
            if zone.particle_count == 0 {
                continue;
            }

            let grid_cells = zone.grid_size * zone.grid_size * zone.grid_size;

            for iter in 0..PBMPM_ITERATIONS {
                let push = PbmpmPush {
                    grid_size: zone.grid_size,
                    particle_count: zone.particle_count,
                    particle_offset: zone.particle_offset,
                    grid_offset: zone.grid_offset,
                    dt,
                    gravity,
                    iteration: iter,
                    _pad: 0,
                };
                let push_bytes = bytemuck::bytes_of(&push);

                // 1. Clear grid
                let clear_wg = (grid_cells + 255) / 256;
                passes::dispatch(
                    &self.device,
                    cmd,
                    &self.clear_grid_pass,
                    clear_wg,
                    1,
                    1,
                    push_bytes,
                );
                passes::barrier(cmd, &self.device);

                // 2. P2G
                let p2g_wg = (zone.particle_count + 255) / 256;
                passes::dispatch(
                    &self.device,
                    cmd,
                    &self.p2g_pass,
                    p2g_wg,
                    1,
                    1,
                    push_bytes,
                );
                passes::barrier(cmd, &self.device);

                // 3. Grid update
                passes::dispatch(
                    &self.device,
                    cmd,
                    &self.grid_update_pass,
                    clear_wg,
                    1,
                    1,
                    push_bytes,
                );
                passes::barrier(cmd, &self.device);

                // 4. G2P
                passes::dispatch(
                    &self.device,
                    cmd,
                    &self.g2p_pass,
                    p2g_wg,
                    1,
                    1,
                    push_bytes,
                );

                // Barrier between iterations (skip after last)
                if iter + 1 < PBMPM_ITERATIONS {
                    passes::barrier(cmd, &self.device);
                }
            }

            // Final barrier after all iterations for this zone
            passes::barrier(cmd, &self.device);
        }
    }

    /// Check zones for sleeping particles and deactivate settled zones.
    ///
    /// Call after [`step()`] each frame. For POC, uses a simple frame counter;
    /// a real implementation would readback max velocity and check threshold.
    pub fn check_sleep(&mut self) {
        for zone_idx in 0..MAX_ZONES {
            let zone = &mut self.zones[zone_idx];
            if zone.state != ZoneState::Active {
                continue;
            }

            // POC: count frames and deactivate after N frames
            zone.frames_quiet += 1;
            if zone.frames_quiet >= SLEEP_FRAMES {
                tracing::info!(
                    "PB-MPM: zone {} sleeping after {} frames",
                    zone_idx,
                    zone.frames_quiet,
                );
                zone.state = ZoneState::Free;
                // Reclaim buffer space if this was the last allocated zone
                self.try_reclaim_buffers();
            }
        }
    }

    /// Download particle data for a zone (for readback/testing).
    ///
    /// Returns the raw f32 data (24 floats per particle).
    pub fn download_particles(
        &self,
        ctx: &VulkanContext,
        zone_idx: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let zone = &self.zones[zone_idx];
        let byte_offset = zone.particle_offset as u64 * PARTICLE_BYTES;
        let byte_size = zone.particle_count as u64 * PARTICLE_BYTES;

        if byte_size == 0 {
            return Ok(Vec::new());
        }

        let staging =
            buffer::create_readback_staging_buffer(ctx, byte_size, "pbmpm-readback-staging")?;

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(byte_offset)
                .dst_offset(0)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    self.particle_buffer.buffer,
                    staging.buffer,
                    &[region],
                );
            }
        })?;

        let result = {
            let mapped = staging
                .mapped_slice()
                .ok_or_else(|| anyhow::anyhow!("failed to map readback staging buffer"))?;
            let typed: &[f32] = bytemuck::cast_slice(&mapped[..byte_size as usize]);
            typed.to_vec()
        };

        buffer::destroy_buffer(ctx, staging);
        Ok(result)
    }

    /// Try to activate a new rigid body PB-MPM zone from an [`IslandResult`].
    ///
    /// Similar to [`activate_zone`] but marks the zone as a rigid body with shape
    /// matching constraint. Returns zone index if successful.
    pub fn activate_rigid_body_zone(
        &mut self,
        island: &crate::island_detector::IslandResult,
    ) -> Option<usize> {
        let grid_size = island.grid_size;

        // Find a free zone slot
        let zone_idx = self.zones.iter().position(|z| z.state == ZoneState::Free)?;

        // Check capacity
        let max_particles_for_zone = grid_size * grid_size * grid_size;
        let grid_cells = grid_size * grid_size * grid_size;

        if self.next_particle_offset + max_particles_for_zone > self.max_particles {
            tracing::warn!(
                "PB-MPM: insufficient particle capacity for rigid body zone (need {}, have {})",
                max_particles_for_zone,
                self.max_particles - self.next_particle_offset,
            );
            return None;
        }
        if self.next_grid_offset + grid_cells > self.max_grid_cells {
            tracing::warn!(
                "PB-MPM: insufficient grid capacity for rigid body zone (need {}, have {})",
                grid_cells,
                self.max_grid_cells - self.next_grid_offset,
            );
            return None;
        }

        // Zone origin: AABB min - 2 padding
        let zone = &mut self.zones[zone_idx];
        zone.world_origin = [
            island.aabb_min[0] - 2,
            island.aabb_min[1] - 2,
            island.aabb_min[2] - 2,
        ];
        zone.grid_size = grid_size;
        zone.particle_count = 0; // set during spawn
        zone.particle_offset = self.next_particle_offset;
        zone.grid_offset = self.next_grid_offset;
        zone.state = ZoneState::Active;
        zone.frames_quiet = 0;
        zone.rigid_body = true;
        zone.rigid_body_state = None; // set after spawning

        self.next_particle_offset += max_particles_for_zone;
        self.next_grid_offset += grid_cells;

        tracing::info!(
            "PB-MPM: activated rigid body zone {} at origin {:?}, grid_size={}, voxels={}",
            zone_idx,
            zone.world_origin,
            grid_size,
            island.voxels.len(),
        );

        Some(zone_idx)
    }

    /// Spawn rigid body particles from an [`IslandResult`].
    ///
    /// Unlike [`spawn_particles_from_voxels`], particles keep phase=0 (solid),
    /// start with zero velocity, and rigid body state is initialized.
    /// Returns the actual number of particles spawned.
    pub fn spawn_rigid_body_particles(
        &mut self,
        ctx: &VulkanContext,
        zone_idx: usize,
        island: &crate::island_detector::IslandResult,
    ) -> anyhow::Result<u32> {
        let zone = &self.zones[zone_idx];
        let world_origin = zone.world_origin;
        let mut particles: Vec<f32> = Vec::new();
        let mut positions_local: Vec<[f32; 3]> = Vec::new();
        let mut count = 0u32;

        for (i, &world_pos) in island.voxels.iter().enumerate() {
            let raw = island.voxel_data[i];
            let v = shared::voxel::Voxel(raw);
            if v.is_air() {
                continue;
            }

            // Position in zone-local grid coordinates
            let px = (world_pos[0] - world_origin[0]) as f32 + 0.5;
            let py = (world_pos[1] - world_origin[1]) as f32 + 0.5;
            let pz = (world_pos[2] - world_origin[2]) as f32 + 0.5;

            positions_local.push([px, py, pz]);

            let mass = 1.0_f32;
            let temp = v.temperature() as f32;

            // Write particle: 24 f32s (96 bytes)
            // pos_mass
            particles.extend_from_slice(&[px, py, pz, mass]);
            // vel_temp — zero velocity, gravity handles fall
            particles.extend_from_slice(&[0.0, 0.0, 0.0, temp]);
            // C_col0, C_col1, C_col2 — zero
            particles.extend_from_slice(&[0.0; 12]);
            // ids: material_id, phase=0 (solid), flags, padding
            let mat_id = v.material_id() as u32;
            let phase = 0u32; // solid — rigid body
            particles.extend_from_slice(&[
                f32::from_bits(mat_id),
                f32::from_bits(phase),
                f32::from_bits(0u32),
                f32::from_bits(0u32),
            ]);

            count += 1;
        }

        if count == 0 {
            return Ok(0);
        }

        // Upload particle data
        let byte_offset = zone.particle_offset as u64 * PARTICLE_BYTES;
        let byte_size = count as u64 * PARTICLE_BYTES;

        let particle_bytes: &[u8] = bytemuck::cast_slice(&particles);
        let mut staging =
            buffer::create_upload_staging_buffer(ctx, byte_size, "pbmpm-rb-particle-staging")?;

        {
            let mapped = staging
                .mapped_slice_mut()
                .ok_or_else(|| anyhow::anyhow!("failed to map staging buffer"))?;
            mapped[..particle_bytes.len()].copy_from_slice(particle_bytes);
        }

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(byte_offset)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    self.particle_buffer.buffer,
                    &[region],
                );
            }
        })?;

        buffer::destroy_buffer(ctx, staging);
        self.zones[zone_idx].particle_count = count;

        // Compute zone-local center of mass
        let mut com = [0.0f32; 3];
        for p in &positions_local {
            com[0] += p[0];
            com[1] += p[1];
            com[2] += p[2];
        }
        let n = positions_local.len() as f32;
        com[0] /= n;
        com[1] /= n;
        com[2] /= n;

        // Initialize rigid body state
        let rb_state =
            crate::rigid_body_tracker::RigidBodyState::new(&positions_local, com);
        self.zones[zone_idx].rigid_body_state = Some(rb_state);

        tracing::info!(
            "PB-MPM: spawned {} rigid body particles for zone {} (com={:?})",
            count,
            zone_idx,
            com,
        );

        Ok(count)
    }

    /// Upload modified particle data back to the GPU for a zone.
    ///
    /// Used after CPU-side shape matching to push updated positions/velocities.
    pub fn upload_particles(
        &self,
        ctx: &VulkanContext,
        zone_idx: usize,
        particle_data: &[f32],
    ) -> anyhow::Result<()> {
        let zone = &self.zones[zone_idx];
        let byte_offset = zone.particle_offset as u64 * PARTICLE_BYTES;
        let byte_size = zone.particle_count as u64 * PARTICLE_BYTES;

        if byte_size == 0 {
            return Ok(());
        }

        let particle_bytes: &[u8] = bytemuck::cast_slice(particle_data);
        let upload_size = (particle_bytes.len() as u64).min(byte_size);

        let mut staging =
            buffer::create_upload_staging_buffer(ctx, upload_size, "pbmpm-rb-upload-staging")?;

        {
            let mapped = staging
                .mapped_slice_mut()
                .ok_or_else(|| anyhow::anyhow!("failed to map staging buffer"))?;
            mapped[..upload_size as usize].copy_from_slice(&particle_bytes[..upload_size as usize]);
        }

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(byte_offset)
                .size(upload_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    self.particle_buffer.buffer,
                    &[region],
                );
            }
        })?;

        buffer::destroy_buffer(ctx, staging);
        Ok(())
    }

    /// Step rigid body zones: integrate, shape match, check fracture/sleep.
    ///
    /// Called after [`step()`] each frame. Downloads particles for rigid body zones,
    /// applies CPU-side shape matching, and uploads modified data back.
    pub fn step_rigid_bodies(&mut self, ctx: &VulkanContext, dt: f32) {
        let gravity = -196.0_f32; // scaled for voxel size (trap #17)

        for zone_idx in 0..MAX_ZONES {
            let zone = &self.zones[zone_idx];
            if zone.state != ZoneState::Active || !zone.rigid_body {
                continue;
            }
            if zone.particle_count == 0 || zone.rigid_body_state.is_none() {
                continue;
            }

            // Download particles
            let particle_data = match self.download_particles(ctx, zone_idx) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("PB-MPM: failed to download particles for RB zone {}: {}", zone_idx, e);
                    continue;
                }
            };

            // Get mutable reference to rigid body state
            let rb_state = self.zones[zone_idx].rigid_body_state.as_mut().expect("checked above");

            // Integrate under gravity
            crate::rigid_body_tracker::integrate(rb_state, dt, gravity);

            // Check fracture
            if crate::rigid_body_tracker::check_fracture(rb_state) {
                tracing::info!("PB-MPM: rigid body zone {} fractured on impact", zone_idx);
                let mut data = particle_data;
                let pc = self.zones[zone_idx].particle_count;
                crate::rigid_body_tracker::fracture_to_debris(&mut data, pc, 5.0);
                if let Err(e) = self.upload_particles(ctx, zone_idx, &data) {
                    tracing::warn!("PB-MPM: failed to upload fracture debris: {}", e);
                }
                self.zones[zone_idx].rigid_body = false;
                self.zones[zone_idx].rigid_body_state = None;
                continue;
            }

            // Apply shape matching
            let mut data = particle_data;
            let pc = self.zones[zone_idx].particle_count;
            let rb_state = self.zones[zone_idx].rigid_body_state.as_ref().expect("checked above");
            crate::rigid_body_tracker::apply_shape_matching(&mut data, pc, rb_state, 0.8);

            // Upload modified particles
            if let Err(e) = self.upload_particles(ctx, zone_idx, &data) {
                tracing::warn!("PB-MPM: failed to upload shape-matched particles: {}", e);
            }
        }
    }

    /// Check zones for sleep using velocity readback.
    ///
    /// Every 10 frames, downloads particles and checks max velocity.
    /// After 3 consecutive low-velocity checks, the zone is slept.
    pub fn check_sleep_with_readback(&mut self, ctx: &VulkanContext) {
        for zone_idx in 0..MAX_ZONES {
            if self.zones[zone_idx].state != ZoneState::Active {
                continue;
            }
            if self.zones[zone_idx].particle_count == 0 {
                continue;
            }

            self.zones[zone_idx].frames_quiet += 1;

            // Only check every 10 frames
            if self.zones[zone_idx].frames_quiet % 10 != 0 {
                continue;
            }

            let pc = self.zones[zone_idx].particle_count;
            let particle_data = match self.download_particles(ctx, zone_idx) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let max_vel_sq =
                crate::rigid_body_tracker::compute_max_velocity_sq(&particle_data, pc);

            if max_vel_sq < crate::rigid_body_tracker::SLEEP_VEL_THRESHOLD_SQ {
                let consecutive = self.zones[zone_idx].frames_quiet / 10;
                if consecutive >= 3 {
                    tracing::info!(
                        "PB-MPM: zone {} sleeping (max_vel_sq={:.3} below threshold)",
                        zone_idx,
                        max_vel_sq,
                    );
                    self.zones[zone_idx].state = ZoneState::Free;
                    self.zones[zone_idx].rigid_body = false;
                    self.zones[zone_idx].rigid_body_state = None;
                    self.try_reclaim_buffers();
                }
            } else {
                self.zones[zone_idx].frames_quiet = 0;
            }
        }
    }

    /// Number of active zones.
    pub fn active_zone_count(&self) -> usize {
        self.zones
            .iter()
            .filter(|z| z.state == ZoneState::Active)
            .count()
    }

    /// Get a reference to a zone by index.
    pub fn zone(&self, idx: usize) -> &PbmpmZone {
        &self.zones[idx]
    }

    /// Reclaim buffer space when all zones after a certain point are free.
    fn try_reclaim_buffers(&mut self) {
        // Simple strategy: if all zones are free, reset offsets to 0
        if self.zones.iter().all(|z| z.state == ZoneState::Free) {
            self.next_particle_offset = 0;
            self.next_grid_offset = 0;
            tracing::debug!("PB-MPM: all zones free, reclaimed buffer space");
        }
    }

    /// Free all GPU resources.
    ///
    /// Must be called before dropping the [`VulkanContext`].
    pub fn destroy(self, ctx: &VulkanContext) {
        // Destroy compute passes
        for pass in [
            &self.clear_grid_pass,
            &self.p2g_pass,
            &self.grid_update_pass,
            &self.g2p_pass,
        ] {
            pipeline::destroy_pipeline(ctx, pass.pipeline);
            pipeline::destroy_pipeline_layout(ctx, pass.pipeline_layout);
            pipeline::destroy_descriptor_set_layout(ctx, pass.descriptor_set_layout);
        }

        pipeline::destroy_descriptor_pool(ctx, self.descriptor_pool);
        pipeline::destroy_shader_module(ctx, self.shader_module);

        buffer::destroy_buffer(ctx, self.particle_buffer);
        buffer::destroy_buffer(ctx, self.grid_buffer);

        tracing::info!("PbmpmZoneManager destroyed");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_constant_size() {
        assert_eq!(mem::size_of::<PbmpmPush>(), 32);
    }

    #[test]
    fn test_zone_activation_and_free() {
        // Use a mock-like approach: test zone slot logic without GPU
        let mut zones = [
            PbmpmZone::free(),
            PbmpmZone::free(),
            PbmpmZone::free(),
            PbmpmZone::free(),
        ];

        // Activate 4 zones
        for i in 0..4 {
            let idx = zones.iter().position(|z| z.state == ZoneState::Free);
            assert!(idx.is_some(), "Should find free slot for zone {}", i);
            zones[idx.unwrap()].state = ZoneState::Active;
        }

        // 5th should fail
        let idx = zones.iter().position(|z| z.state == ZoneState::Free);
        assert!(idx.is_none(), "Should have no free slots");

        // Free one, verify re-activation works
        zones[2].state = ZoneState::Free;
        let idx = zones.iter().position(|z| z.state == ZoneState::Free);
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn test_zone_state_default() {
        let zone = PbmpmZone::free();
        assert_eq!(zone.state, ZoneState::Free);
        assert_eq!(zone.particle_count, 0);
        assert_eq!(zone.grid_size, 0);
    }

    // --- GPU tests (require VulkanContext, run with --test-threads=1) ---

    fn init_tracing() {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    }

    #[test]
    fn test_create_and_destroy() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");
        assert_eq!(mgr.active_zone_count(), 0);
        mgr.destroy(&ctx);
    }

    #[test]
    fn test_activate_zone() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");

        let trigger = ActivationTrigger {
            center: [16.0, 16.0, 16.0],
            radius: 8.0,
            impulse: 10.0,
        };
        let zone_idx = mgr.activate_zone(&trigger, 32).expect("should activate");
        assert_eq!(zone_idx, 0);
        assert_eq!(mgr.active_zone_count(), 1);

        // Activate a second zone
        let trigger2 = ActivationTrigger {
            center: [48.0, 16.0, 16.0],
            radius: 8.0,
            impulse: 10.0,
        };
        let zone_idx2 = mgr.activate_zone(&trigger2, 32).expect("should activate second");
        assert_eq!(zone_idx2, 1);
        assert_eq!(mgr.active_zone_count(), 2);

        mgr.destroy(&ctx);
    }

    #[test]
    fn test_activate_max_zones() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");

        // Activate MAX_ZONES zones (use grid_size=32 -> 32768 cells each)
        for i in 0..MAX_ZONES {
            let trigger = ActivationTrigger {
                center: [16.0 + i as f32 * 64.0, 16.0, 16.0],
                radius: 8.0,
                impulse: 10.0,
            };
            let result = mgr.activate_zone(&trigger, 32);
            assert!(result.is_some(), "Zone {} should activate", i);
        }
        assert_eq!(mgr.active_zone_count(), MAX_ZONES);

        // 5th should fail (no free slots)
        let trigger_extra = ActivationTrigger {
            center: [0.0, 0.0, 0.0],
            radius: 8.0,
            impulse: 10.0,
        };
        assert!(mgr.activate_zone(&trigger_extra, 32).is_none());

        mgr.destroy(&ctx);
    }

    #[test]
    fn test_step_empty_zones() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");

        // Step with no active zones should be a no-op
        ctx.execute_one_shot(|cmd| {
            mgr.step(cmd, &ctx, 0.016);
        })
        .expect("Empty step failed");

        mgr.destroy(&ctx);
    }

    #[test]
    fn test_activate_and_step() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");

        let trigger = ActivationTrigger {
            center: [16.0, 16.0, 16.0],
            radius: 8.0,
            impulse: 10.0,
        };
        let zone_idx = mgr.activate_zone(&trigger, 32).expect("should activate");
        assert_eq!(mgr.active_zone_count(), 1);

        // Upload a few test particles manually
        let test_particles: Vec<f32> = {
            let mut p = Vec::new();
            for i in 0..5u32 {
                let x = 10.0 + i as f32;
                let y = 16.0;
                let z = 16.0;
                // pos_mass
                p.extend_from_slice(&[x, y, z, 1.0]);
                // vel_temp
                p.extend_from_slice(&[0.0, 0.0, 0.0, 128.0]);
                // C_col0 (APIC affine momentum, zero initially)
                p.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                // C_col1 (zero)
                p.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                // C_col2 (zero)
                p.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                // ids
                p.extend_from_slice(&[
                    f32::from_bits(1), // material_id = 1
                    f32::from_bits(1), // phase = 1 (liquid)
                    f32::from_bits(0), // flags = 0
                    f32::from_bits(0), // pad
                ]);
            }
            p
        };

        // Upload
        let byte_size = (test_particles.len() * 4) as u64;
        let particle_bytes: &[u8] = bytemuck::cast_slice(&test_particles);
        let mut staging =
            buffer::create_upload_staging_buffer(&ctx, byte_size, "test-particle-staging")
                .expect("staging");
        {
            let mapped = staging.mapped_slice_mut().expect("map");
            mapped[..particle_bytes.len()].copy_from_slice(particle_bytes);
        }
        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    mgr.particle_buffer.buffer,
                    &[region],
                );
            }
        })
        .expect("upload");
        buffer::destroy_buffer(&ctx, staging);

        mgr.zones[zone_idx].particle_count = 5;

        // Step should not crash
        ctx.execute_one_shot(|cmd| {
            mgr.step(cmd, &ctx, 0.016);
        })
        .expect("Simulation step failed");

        // Readback and verify particles are still finite
        let readback = mgr
            .download_particles(&ctx, zone_idx)
            .expect("readback failed");
        assert_eq!(readback.len(), 5 * 24); // 5 particles * 24 f32s each
        for i in 0..5 {
            let base = i * 24;
            let px = readback[base];
            let py = readback[base + 1];
            let pz = readback[base + 2];
            assert!(
                px.is_finite() && py.is_finite() && pz.is_finite(),
                "Particle {} has non-finite position: ({}, {}, {})",
                i,
                px,
                py,
                pz,
            );
        }

        mgr.destroy(&ctx);
    }

    #[test]
    fn test_check_sleep() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut mgr = PbmpmZoneManager::new(&ctx).expect("Failed to create PbmpmZoneManager");

        let trigger = ActivationTrigger {
            center: [16.0, 16.0, 16.0],
            radius: 8.0,
            impulse: 10.0,
        };
        mgr.activate_zone(&trigger, 32).expect("should activate");
        assert_eq!(mgr.active_zone_count(), 1);

        // Check sleep for SLEEP_FRAMES - 1 times: should still be active
        for _ in 0..(SLEEP_FRAMES - 1) {
            mgr.check_sleep();
        }
        assert_eq!(mgr.active_zone_count(), 1);

        // One more: should deactivate
        mgr.check_sleep();
        assert_eq!(mgr.active_zone_count(), 0);

        // Buffer space should be reclaimed (all zones free)
        assert_eq!(mgr.next_particle_offset, 0);
        assert_eq!(mgr.next_grid_offset, 0);

        mgr.destroy(&ctx);
    }
}
