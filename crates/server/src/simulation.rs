//! [`GpuSimulation`] implementation — construction, stepping, rendering, destruction.

use std::mem;

use ash::vk;
use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
    pipeline,
};
use shared::{
    GRID_CELL_COUNT, GRID_SIZE, GridCell, MAX_PARTICLES, Particle, RENDER_HEIGHT, RENDER_WIDTH,
    material::{self, MATERIAL_COUNT, MaterialParams},
    reaction::{self, MAX_PHASE_RULES, PhaseTransitionRule},
};

use crate::{
    Result, ServerError,
    passes::{self, ComputePass},
    push_constants::*,
};

/// Compiled SPIR-V shader module bytes, included at compile time.
const SHADER_BYTES: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));

/// GPU compute simulation orchestrator.
///
/// Manages GPU buffers, compute pipelines, and the dispatch chain for the
/// full MPM simulation pipeline. Create via [`GpuSimulation::new`].
pub struct GpuSimulation {
    // Buffers
    pub(crate) particle_buffer: GpuBuffer,
    pub(crate) grid_buffer: GpuBuffer,
    pub(crate) voxel_buffer: GpuBuffer,
    pub(crate) render_output_buffer: GpuBuffer,
    pub(crate) material_buffer: GpuBuffer,
    pub(crate) reaction_buffer: GpuBuffer,

    // Compute passes
    clear_grid_pass: ComputePass,
    p2g_pass: ComputePass,
    grid_update_pass: ComputePass,
    g2p_pass: ComputePass,
    clear_voxels_pass: ComputePass,
    voxelize_pass: ComputePass,
    render_pass: ComputePass,
    toolbar_overlay_pass: ComputePass,
    react_pass: ComputePass,
    explosion_pass: ComputePass,

    // Shader module (kept alive for pipeline lifetime)
    shader_module: vk::ShaderModule,

    // Descriptor pool (owns all descriptor sets)
    descriptor_pool: vk::DescriptorPool,

    // State
    pub(crate) num_particles: u32,
    num_phase_rules: u32,

    // Reference to context (not owned — caller must keep alive)
    // We store raw device handle for recording commands; the context
    // must outlive GpuSimulation.
    pub(crate) device: ash::Device,
}

impl GpuSimulation {
    /// Create a new GPU simulation orchestrator with default (hardcoded) material tables.
    ///
    /// Loads the compiled SPIR-V shaders, creates GPU buffers and compute
    /// pipelines. The provided [`VulkanContext`] must outlive this struct.
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let material_table = material::default_material_table();
        let (reaction_table, num_phase_rules) = reaction::default_phase_transition_table();
        Self::new_with_materials(ctx, &material_table, &reaction_table, num_phase_rules)
    }

    /// Create a new GPU simulation orchestrator with externally-provided material tables.
    ///
    /// Use this to load data-driven materials from RON files via the `content` crate.
    /// The provided [`VulkanContext`] must outlive this struct.
    pub fn new_with_materials(
        ctx: &VulkanContext,
        material_table: &[MaterialParams; MATERIAL_COUNT],
        reaction_table: &[PhaseTransitionRule; MAX_PHASE_RULES],
        num_phase_rules: usize,
    ) -> Result<Self> {
        tracing::info!("Creating GpuSimulation");

        // -- Shader module --
        let shader_module = pipeline::create_shader_module(ctx, SHADER_BYTES, "mpm-shaders")?;

        // -- Buffers --
        let particle_buf_size =
            (MAX_PARTICLES as usize * mem::size_of::<Particle>()) as vk::DeviceSize;
        let particle_buffer = buffer::create_device_local_buffer(
            ctx,
            particle_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "particle-buffer",
        )?;

        // Grid buffer: must fit both GridCell view and flat f32 view.
        // GridCell = 48 bytes = 12 * f32, so GRID_CELL_COUNT * sizeof(GridCell)
        // equals GRID_CELL_COUNT * 12 * sizeof(f32). Same size, no issue.
        let grid_buf_size =
            (GRID_CELL_COUNT as usize * mem::size_of::<GridCell>()) as vk::DeviceSize;
        let grid_buffer = buffer::create_device_local_buffer(
            ctx,
            grid_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "grid-buffer",
        )?;

        let voxel_buf_size =
            (GRID_CELL_COUNT as usize * mem::size_of::<glam::UVec4>()) as vk::DeviceSize;
        let voxel_buffer = buffer::create_device_local_buffer(
            ctx,
            voxel_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "voxel-buffer",
        )?;

        // Render output buffer: RGBA u32 per pixel, needs TRANSFER_SRC for copy to swapchain
        let render_output_size =
            (RENDER_WIDTH as usize * RENDER_HEIGHT as usize * mem::size_of::<u32>())
                as vk::DeviceSize;
        let render_output_buffer = buffer::create_device_local_buffer(
            ctx,
            render_output_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "render-output-buffer",
        )?;

        // Material params buffer: MATERIAL_COUNT * 64 bytes
        let material_buf_size =
            (MATERIAL_COUNT * mem::size_of::<MaterialParams>()) as vk::DeviceSize;
        let material_buffer = buffer::create_device_local_buffer(
            ctx,
            material_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "material-buffer",
        )?;

        // Upload material table
        buffer::upload(ctx, material_table, &material_buffer)?;
        tracing::info!(
            "Uploaded {} materials ({} bytes) to GPU",
            MATERIAL_COUNT,
            material_buf_size
        );

        // Phase transition rules buffer: MAX_PHASE_RULES * 32 bytes
        let reaction_buf_size =
            (MAX_PHASE_RULES * mem::size_of::<PhaseTransitionRule>()) as vk::DeviceSize;
        let reaction_buffer = buffer::create_device_local_buffer(
            ctx,
            reaction_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "reaction-buffer",
        )?;

        // Upload phase transition table
        buffer::upload(ctx, reaction_table, &reaction_buffer)?;
        tracing::info!(
            "Uploaded {} phase transition rules ({} bytes) to GPU",
            num_phase_rules,
            reaction_buf_size
        );

        // -- Descriptor pool --
        // 10 passes, max 4 bindings each = up to 40 storage buffer descriptors
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 40,
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 10, "sim-descriptor-pool")?;

        // -- Create each compute pass --
        // Entry point names are fully qualified module paths in rust-gpu SPIR-V.
        let clear_grid_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::clear_grid::clear_grid",
            &[&grid_buffer],
            0, // no push constants
            descriptor_pool,
            "clear-grid",
        )?;

        let p2g_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::p2g::p2g",
            &[&particle_buffer, &grid_buffer, &material_buffer],
            mem::size_of::<P2gPushConstants>() as u32,
            descriptor_pool,
            "p2g",
        )?;

        let grid_update_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::grid_update::grid_update",
            &[&grid_buffer],
            mem::size_of::<GridUpdatePushConstants>() as u32,
            descriptor_pool,
            "grid-update",
        )?;

        let g2p_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::g2p::g2p",
            &[&particle_buffer, &grid_buffer, &voxel_buffer, &reaction_buffer],
            mem::size_of::<G2pPushConstants>() as u32,
            descriptor_pool,
            "g2p",
        )?;

        let clear_voxels_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::clear_voxels",
            &[&voxel_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "clear-voxels",
        )?;

        let voxelize_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::voxelize",
            &[&particle_buffer, &voxel_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "voxelize",
        )?;

        let render_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::render::render_voxels",
            &[&voxel_buffer, &render_output_buffer, &material_buffer],
            mem::size_of::<RenderPushConstants>() as u32,
            descriptor_pool,
            "render",
        )?;

        let toolbar_overlay_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::toolbar_overlay::toolbar_overlay",
            &[&render_output_buffer],
            mem::size_of::<ToolbarPushConstants>() as u32,
            descriptor_pool,
            "toolbar-overlay",
        )?;

        let react_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::react::react",
            &[&particle_buffer, &voxel_buffer],
            mem::size_of::<ReactPushConstants>() as u32,
            descriptor_pool,
            "react",
        )?;

        let explosion_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::explosion::explosion",
            &[&particle_buffer],
            mem::size_of::<ExplosionPushConstants>() as u32,
            descriptor_pool,
            "explosion",
        )?;

        tracing::info!("GpuSimulation created successfully");

        Ok(Self {
            particle_buffer,
            grid_buffer,
            voxel_buffer,
            render_output_buffer,
            material_buffer,
            reaction_buffer,
            clear_grid_pass,
            p2g_pass,
            grid_update_pass,
            g2p_pass,
            clear_voxels_pass,
            voxelize_pass,
            render_pass,
            toolbar_overlay_pass,
            react_pass,
            explosion_pass,
            shader_module,
            descriptor_pool,
            num_particles: 0,
            num_phase_rules: num_phase_rules as u32,
            device: ctx.device.clone(),
        })
    }

    /// Upload initial particle data to the GPU.
    ///
    /// The number of particles must not exceed [`MAX_PARTICLES`].
    pub fn init_particles(&mut self, ctx: &VulkanContext, particles: &[Particle]) -> Result<()> {
        if particles.len() > MAX_PARTICLES as usize {
            return Err(ServerError::TooManyParticles(particles.len()));
        }
        self.num_particles = particles.len() as u32;
        buffer::upload(ctx, particles, &self.particle_buffer)?;
        tracing::info!("Uploaded {} particles to GPU", self.num_particles);
        Ok(())
    }

    /// Record the full simulation dispatch chain into the given command buffer.
    ///
    /// The command buffer must already be in the recording state.
    /// The dispatch chain is:
    /// `clear_grid -> P2G -> grid_update -> G2P -> clear_voxels -> voxelize`.
    ///
    /// Memory barriers are inserted between each dispatch to ensure
    /// correct ordering of shader reads and writes.
    /// Run physics only (no reactions). Call in substep loop.
    pub fn step_physics(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let dt = shared::DT;
        let gravity = shared::GRAVITY;

        // Workgroup counts for grid dispatches (4x4x4 workgroups)
        let grid_wg = grid_size / 4;
        // Workgroup count for particle dispatches (64 threads)
        let particle_wg = (num_particles + 63) / 64;

        // 1. Clear grid
        passes::dispatch(&self.device, cmd, &self.clear_grid_pass, grid_wg, grid_wg, grid_wg, &[]);
        passes::barrier(cmd, &self.device);

        // 2. P2G
        let p2g_pc = P2gPushConstants {
            grid_size,
            dt,
            num_particles,
            _pad: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.p2g_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&p2g_pc),
        );
        passes::barrier(cmd, &self.device);

        // 3. Grid update
        let grid_pc = GridUpdatePushConstants {
            grid_size,
            dt,
            gravity,
            _pad: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.grid_update_pass,
            grid_wg,
            grid_wg,
            grid_wg,
            bytemuck::bytes_of(&grid_pc),
        );
        passes::barrier(cmd, &self.device);

        // 4. G2P
        let g2p_pc = G2pPushConstants {
            grid_size,
            dt,
            num_particles,
            num_rules: self.num_phase_rules,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.g2p_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&g2p_pc),
        );
        passes::barrier(cmd, &self.device);

        // 5. Clear voxels
        let vox_pc = VoxelizePushConstants {
            grid_size,
            num_particles,
            _pad0: 0,
            _pad1: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.clear_voxels_pass,
            grid_wg,
            grid_wg,
            grid_wg,
            bytemuck::bytes_of(&vox_pc),
        );
        passes::barrier(cmd, &self.device);

        // 6. Voxelize
        passes::dispatch(
            &self.device,
            cmd,
            &self.voxelize_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&vox_pc),
        );
        passes::barrier(cmd, &self.device);
    }

    /// Run chemical reactions only. Call once per frame after all substeps.
    pub fn step_react(&self, cmd: vk::CommandBuffer) {
        let grid_size = GRID_SIZE;
        let num_particles = self.num_particles;
        let particle_wg = (num_particles + 63) / 64;

        let react_pc = ReactPushConstants {
            grid_size,
            num_particles,
            _pad0: 0,
            _pad1: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.react_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&react_pc),
        );
    }

    /// Convenience: run one full step (physics + react).
    pub fn step(&self, cmd: vk::CommandBuffer) {
        self.step_physics(cmd);
        self.step_react(cmd);
    }

    /// Append new particles to the existing simulation.
    ///
    /// Reads back current particles from the GPU, appends the new ones,
    /// and re-uploads the full set. The total count must not exceed
    /// [`MAX_PARTICLES`]. This is not efficient but works for MVP interactive
    /// spawning where we add small batches infrequently.
    pub fn add_particles(&mut self, ctx: &VulkanContext, new_particles: &[Particle]) -> Result<()> {
        let new_total = self.num_particles as usize + new_particles.len();
        if new_total > MAX_PARTICLES as usize {
            return Err(ServerError::TooManyParticles(new_total));
        }

        let mut all_particles = self.readback_particles(ctx)?;
        all_particles.extend_from_slice(new_particles);

        self.num_particles = all_particles.len() as u32;
        buffer::upload(ctx, &all_particles, &self.particle_buffer)?;
        tracing::info!(
            "Added {} particles, total now {}",
            new_particles.len(),
            self.num_particles
        );
        Ok(())
    }

    /// Record a render dispatch into the given command buffer.
    ///
    /// Ray-marches through the voxel grid and writes pixel data to the
    /// render output buffer. The command buffer must be in recording state.
    /// Call this after `step()` (which populates the voxel buffer).
    pub fn render(
        &self,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        eye: [f32; 3],
        target: [f32; 3],
    ) {
        // Barrier: ensure voxelize writes are visible to render shader reads
        passes::barrier(cmd, &self.device);

        let push = RenderPushConstants {
            width,
            height,
            grid_size: GRID_SIZE,
            _pad: 0,
            eye: glam::Vec4::new(eye[0], eye[1], eye[2], 0.0),
            target: glam::Vec4::new(target[0], target[1], target[2], 0.0),
        };

        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        passes::dispatch(
            &self.device,
            cmd,
            &self.render_pass,
            wg_x,
            wg_y,
            1,
            bytemuck::bytes_of(&push),
        );
    }

    /// Record a toolbar overlay dispatch into the given command buffer.
    ///
    /// Draws a material selection toolbar on top of the render output buffer.
    /// Must be called after [`render`]. The command buffer must be in recording state.
    pub fn render_toolbar(&self, cmd: vk::CommandBuffer, push: &ToolbarPushConstants) {
        // Barrier: ensure render writes are visible before toolbar reads/writes
        passes::barrier(cmd, &self.device);

        let wg_x = (push.screen_width + 7) / 8;
        let wg_y = (push.screen_height + 7) / 8;

        passes::dispatch(
            &self.device,
            cmd,
            &self.toolbar_overlay_pass,
            wg_x,
            wg_y,
            1,
            bytemuck::bytes_of(push),
        );
    }

    /// Record an explosion dispatch into the given command buffer.
    ///
    /// Applies a radial impulse to all particles within `radius` of `center`.
    /// Should be called between `step()` and `render()`. The command buffer
    /// must be in recording state.
    pub fn apply_explosion(
        &self,
        cmd: vk::CommandBuffer,
        center: [f32; 3],
        radius: f32,
        strength: f32,
    ) {
        passes::barrier(cmd, &self.device);

        let push = ExplosionPushConstants {
            center: glam::Vec4::new(center[0], center[1], center[2], 0.0),
            params: glam::Vec4::new(radius, strength, shared::DT, self.num_particles as f32),
            num_particles: self.num_particles,
            _pad: [0; 3],
        };

        let particle_wg = (self.num_particles + 63) / 64;
        passes::dispatch(
            &self.device,
            cmd,
            &self.explosion_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&push),
        );
    }

    /// Insert a final barrier to ensure all render output writes are complete
    /// before the buffer is copied to the swapchain.
    ///
    /// Must be called after [`render`] and [`render_toolbar`] (if used).
    pub fn finalize_render(&self, cmd: vk::CommandBuffer) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }

    /// Destroy all GPU resources. Must be called before the [`VulkanContext`] is dropped.
    ///
    /// After calling this, the struct is in an invalid state and must not be used.
    pub fn destroy(&mut self, ctx: &VulkanContext) {
        tracing::info!("Destroying GpuSimulation");

        // Wait for GPU idle before destroying resources
        unsafe {
            let _ = ctx.device.device_wait_idle();
        }

        // Destroy passes
        let passes = [
            &self.clear_grid_pass,
            &self.p2g_pass,
            &self.grid_update_pass,
            &self.g2p_pass,
            &self.clear_voxels_pass,
            &self.voxelize_pass,
            &self.render_pass,
            &self.toolbar_overlay_pass,
            &self.react_pass,
            &self.explosion_pass,
        ];
        for pass in passes {
            pipeline::destroy_pipeline(ctx, pass.pipeline);
            pipeline::destroy_pipeline_layout(ctx, pass.pipeline_layout);
            pipeline::destroy_descriptor_set_layout(ctx, pass.descriptor_set_layout);
        }

        // Descriptor pool frees all allocated sets
        pipeline::destroy_descriptor_pool(ctx, self.descriptor_pool);

        // Shader module
        pipeline::destroy_shader_module(ctx, self.shader_module);

        // Buffers
        // We need to take ownership by swapping with dummy values
        let particle_buf = std::mem::replace(
            &mut self.particle_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let grid_buf = std::mem::replace(
            &mut self.grid_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let voxel_buf = std::mem::replace(
            &mut self.voxel_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let render_output_buf = std::mem::replace(
            &mut self.render_output_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let material_buf = std::mem::replace(
            &mut self.material_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let reaction_buf = std::mem::replace(
            &mut self.reaction_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        buffer::destroy_buffer(ctx, particle_buf);
        buffer::destroy_buffer(ctx, grid_buf);
        buffer::destroy_buffer(ctx, voxel_buf);
        buffer::destroy_buffer(ctx, render_output_buf);
        buffer::destroy_buffer(ctx, material_buf);
        buffer::destroy_buffer(ctx, reaction_buf);
    }
}
