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
const PBMPM_ITERATIONS: u32 = 2;

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

                    // Velocity: radial impulse from trigger center
                    let dx = px - (trigger.center[0] - zone.world_origin[0] as f32);
                    let dy = py - (trigger.center[1] - zone.world_origin[1] as f32);
                    let dz = pz - (trigger.center[2] - zone.world_origin[2] as f32);
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                    let impulse_scale = trigger.impulse / dist;
                    let vx = dx * impulse_scale;
                    let vy = dy * impulse_scale;
                    let vz = dz * impulse_scale;

                    let mass = 1.0_f32;
                    let temp = v.temperature() as f32;

                    // Write particle: 24 f32s (96 bytes)
                    // pos_mass
                    particles.extend_from_slice(&[px, py, pz, mass]);
                    // vel_temp
                    particles.extend_from_slice(&[vx, vy, vz, temp]);
                    // F_col0: identity
                    particles.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
                    // F_col1: identity
                    particles.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
                    // F_col2: identity
                    particles.extend_from_slice(&[0.0, 0.0, 1.0, 0.0]);
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
                // F_col0 (identity)
                p.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
                // F_col1 (identity)
                p.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
                // F_col2 (identity)
                p.extend_from_slice(&[0.0, 0.0, 1.0, 0.0]);
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
