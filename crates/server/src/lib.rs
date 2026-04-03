//! # server
//!
//! GPU compute simulation orchestrator for the VOX voxel physics engine.
//!
//! Owns [`GpuSimulation`], which manages GPU buffers, compute pipelines,
//! and the dispatch chain for the MPM simulation:
//! `clear_grid -> P2G -> grid_update -> G2P -> (clear_voxels -> voxelize)`.
//!
//! All shaders are compiled to a single SPIR-V module at build time via
//! `spirv-builder` in `build.rs`.

use std::{ffi::CStr, mem};

use ash::vk;
use bytemuck::Pod;
use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
    pipeline,
};
use shared::{
    GRID_CELL_COUNT, GRID_SIZE, GridCell, MAX_PARTICLES, Particle, RENDER_HEIGHT, RENDER_WIDTH,
};

/// Compiled SPIR-V shader module bytes, included at compile time.
const SHADER_BYTES: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));

// ---------------------------------------------------------------------------
// Push constant structs (mirroring shader-side definitions)
// ---------------------------------------------------------------------------

/// Push constants for the P2G shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct P2gPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the grid update shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct GridUpdatePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Gravity acceleration (negative Y).
    pub gravity: f32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the G2P shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct G2pPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Simulation timestep.
    pub dt: f32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding to 16-byte alignment.
    pub _pad: u32,
}

/// Push constants for the voxelize / clear_voxels shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct VoxelizePushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

/// Push constants for the render shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
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
    pub eye: glam::Vec4,
    /// Camera target position (xyz) + padding (w).
    pub target: glam::Vec4,
}

/// Push constants for the react shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ReactPushConstants {
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
    /// Total number of active particles.
    pub num_particles: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

/// Maximum number of material slots in the toolbar.
pub const TOOLBAR_MAX_MATERIALS: usize = 8;

/// Push constants for the toolbar overlay shader.
///
/// Must match `ToolbarParams` in `shaders/src/compute/toolbar_overlay.rs`.
/// Total size: 16 bytes (header) + 128 bytes (8 x Vec4 colors) = 144 bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, bytemuck::Zeroable)]
pub struct ToolbarPushConstants {
    /// Render target width in pixels.
    pub screen_width: u32,
    /// Render target height in pixels.
    pub screen_height: u32,
    /// Index of the currently selected material slot (0-based).
    pub selected_index: u32,
    /// Number of material slots to display.
    pub material_count: u32,
    /// RGBA colors for each material slot (up to 8).
    pub colors: [glam::Vec4; TOOLBAR_MAX_MATERIALS],
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in the server simulation.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// GPU-core error.
    #[error("GPU error: {0}")]
    Gpu(#[from] gpu_core::GpuError),

    /// Too many particles.
    #[error("Particle count {0} exceeds maximum {MAX_PARTICLES}")]
    TooManyParticles(usize),
}

/// Result type alias for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

// ---------------------------------------------------------------------------
// Internal pipeline wrapper
// ---------------------------------------------------------------------------

/// A single compute pipeline with its layout, descriptor set layout, and descriptor set.
struct ComputePass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
}

// ---------------------------------------------------------------------------
// GpuSimulation
// ---------------------------------------------------------------------------

/// GPU compute simulation orchestrator.
///
/// Manages GPU buffers, compute pipelines, and the dispatch chain for the
/// full MPM simulation pipeline. Create via [`GpuSimulation::new`].
pub struct GpuSimulation {
    // Buffers
    particle_buffer: GpuBuffer,
    grid_buffer: GpuBuffer,
    voxel_buffer: GpuBuffer,
    render_output_buffer: GpuBuffer,

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

    // Shader module (kept alive for pipeline lifetime)
    shader_module: vk::ShaderModule,

    // Descriptor pool (owns all descriptor sets)
    descriptor_pool: vk::DescriptorPool,

    // State
    num_particles: u32,

    // Reference to context (not owned — caller must keep alive)
    // We store raw device handle for recording commands; the context
    // must outlive GpuSimulation.
    device: ash::Device,
}

impl GpuSimulation {
    /// Create a new GPU simulation orchestrator.
    ///
    /// Loads the compiled SPIR-V shaders, creates GPU buffers and compute
    /// pipelines. The provided [`VulkanContext`] must outlive this struct.
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
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
        // GridCell = 32 bytes = 8 * f32, so GRID_CELL_COUNT * sizeof(GridCell)
        // equals GRID_CELL_COUNT * 8 * sizeof(f32). Same size, no issue.
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

        // -- Descriptor pool --
        // 9 passes, max 2 bindings each = up to 18 storage buffer descriptors
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 18,
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 9, "sim-descriptor-pool")?;

        // -- Create each compute pass --
        // Entry point names are fully qualified module paths in rust-gpu SPIR-V.
        let clear_grid_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::clear_grid::clear_grid",
            &[&grid_buffer],
            0, // no push constants
            descriptor_pool,
            "clear-grid",
        )?;

        let p2g_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::p2g::p2g",
            &[&particle_buffer, &grid_buffer],
            mem::size_of::<P2gPushConstants>() as u32,
            descriptor_pool,
            "p2g",
        )?;

        let grid_update_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::grid_update::grid_update",
            &[&grid_buffer],
            mem::size_of::<GridUpdatePushConstants>() as u32,
            descriptor_pool,
            "grid-update",
        )?;

        let g2p_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::g2p::g2p",
            &[&particle_buffer, &grid_buffer],
            mem::size_of::<G2pPushConstants>() as u32,
            descriptor_pool,
            "g2p",
        )?;

        let clear_voxels_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::clear_voxels",
            &[&voxel_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "clear-voxels",
        )?;

        let voxelize_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::voxelize",
            &[&particle_buffer, &voxel_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "voxelize",
        )?;

        let render_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::render::render_voxels",
            &[&voxel_buffer, &render_output_buffer],
            mem::size_of::<RenderPushConstants>() as u32,
            descriptor_pool,
            "render",
        )?;

        let toolbar_overlay_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::toolbar_overlay::toolbar_overlay",
            &[&render_output_buffer],
            mem::size_of::<ToolbarPushConstants>() as u32,
            descriptor_pool,
            "toolbar-overlay",
        )?;

        let react_pass = Self::create_pass(
            ctx,
            shader_module,
            c"compute::react::react",
            &[&particle_buffer, &voxel_buffer],
            mem::size_of::<ReactPushConstants>() as u32,
            descriptor_pool,
            "react",
        )?;

        tracing::info!("GpuSimulation created successfully");

        Ok(Self {
            particle_buffer,
            grid_buffer,
            voxel_buffer,
            render_output_buffer,
            clear_grid_pass,
            p2g_pass,
            grid_update_pass,
            g2p_pass,
            clear_voxels_pass,
            voxelize_pass,
            render_pass,
            toolbar_overlay_pass,
            react_pass,
            shader_module,
            descriptor_pool,
            num_particles: 0,
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
    pub fn step(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let dt = shared::DT;
        let gravity = shared::GRAVITY;

        // Workgroup counts for grid dispatches (4x4x4 workgroups)
        let grid_wg = grid_size / 4;
        // Workgroup count for particle dispatches (64 threads)
        let particle_wg = (num_particles + 63) / 64;

        // 1. Clear grid
        self.dispatch(cmd, &self.clear_grid_pass, grid_wg, grid_wg, grid_wg, &[]);
        Self::barrier(cmd, &self.device);

        // 2. P2G
        let p2g_pc = P2gPushConstants {
            grid_size,
            dt,
            num_particles,
            _pad: 0,
        };
        self.dispatch(
            cmd,
            &self.p2g_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&p2g_pc),
        );
        Self::barrier(cmd, &self.device);

        // 3. Grid update
        let grid_pc = GridUpdatePushConstants {
            grid_size,
            dt,
            gravity,
            _pad: 0,
        };
        self.dispatch(
            cmd,
            &self.grid_update_pass,
            grid_wg,
            grid_wg,
            grid_wg,
            bytemuck::bytes_of(&grid_pc),
        );
        Self::barrier(cmd, &self.device);

        // 4. G2P
        let g2p_pc = G2pPushConstants {
            grid_size,
            dt,
            num_particles,
            _pad: 0,
        };
        self.dispatch(
            cmd,
            &self.g2p_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&g2p_pc),
        );
        Self::barrier(cmd, &self.device);

        // 5. Clear voxels
        let vox_pc = VoxelizePushConstants {
            grid_size,
            num_particles,
            _pad0: 0,
            _pad1: 0,
        };
        self.dispatch(
            cmd,
            &self.clear_voxels_pass,
            grid_wg,
            grid_wg,
            grid_wg,
            bytemuck::bytes_of(&vox_pc),
        );
        Self::barrier(cmd, &self.device);

        // 6. Voxelize
        self.dispatch(
            cmd,
            &self.voxelize_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&vox_pc),
        );
        Self::barrier(cmd, &self.device);

        // 7. React (chemical reactions via voxel neighbor lookup)
        let react_pc = ReactPushConstants {
            grid_size,
            num_particles,
            _pad0: 0,
            _pad1: 0,
        };
        self.dispatch(
            cmd,
            &self.react_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&react_pc),
        );
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

    /// Read particle data back from the GPU for debugging.
    ///
    /// This is a synchronous operation that stalls the GPU pipeline.
    /// Use only for debugging and testing.
    pub fn readback_particles(&self, ctx: &VulkanContext) -> Result<Vec<Particle>> {
        let particles =
            buffer::readback::<Particle>(ctx, &self.particle_buffer, self.num_particles as usize)?;
        Ok(particles)
    }

    /// Returns the current number of active particles.
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    /// Returns a reference to the particle buffer.
    pub fn particle_buffer(&self) -> &GpuBuffer {
        &self.particle_buffer
    }

    /// Returns a reference to the grid buffer.
    pub fn grid_buffer(&self) -> &GpuBuffer {
        &self.grid_buffer
    }

    /// Returns a reference to the voxel buffer.
    pub fn voxel_buffer(&self) -> &GpuBuffer {
        &self.voxel_buffer
    }

    /// Returns the Vulkan buffer handle for the render output.
    ///
    /// The buffer contains packed BGRA u32 pixels, sized `width * height`.
    pub fn render_output_buffer(&self) -> vk::Buffer {
        self.render_output_buffer.buffer
    }

    /// Returns a reference to the render output [`GpuBuffer`].
    ///
    /// Useful for readback operations (e.g., headless screenshot).
    /// The buffer contains packed BGRA u32 pixels, sized `width * height`.
    pub fn render_output_gpu_buffer(&self) -> &GpuBuffer {
        &self.render_output_buffer
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
        Self::barrier(cmd, &self.device);

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

        self.dispatch(
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
        Self::barrier(cmd, &self.device);

        let wg_x = (push.screen_width + 7) / 8;
        let wg_y = (push.screen_height + 7) / 8;

        self.dispatch(
            cmd,
            &self.toolbar_overlay_pass,
            wg_x,
            wg_y,
            1,
            bytemuck::bytes_of(push),
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

    // -- Internal helpers --

    /// Create a single compute pass: descriptor set layout, pipeline layout,
    /// pipeline, and descriptor set with buffer bindings.
    fn create_pass(
        ctx: &VulkanContext,
        shader_module: vk::ShaderModule,
        entry_point: &CStr,
        buffers: &[&GpuBuffer],
        push_constant_size: u32,
        descriptor_pool: vk::DescriptorPool,
        name: &str,
    ) -> Result<ComputePass> {
        // Descriptor bindings: one STORAGE_BUFFER per buffer
        let bindings: Vec<pipeline::DescriptorBinding> = (0..buffers.len() as u32)
            .map(|i| pipeline::DescriptorBinding {
                binding: i,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            })
            .collect();

        let ds_layout_name = format!("{name}-ds-layout");
        let descriptor_set_layout =
            pipeline::create_descriptor_set_layout(ctx, &bindings, &ds_layout_name)?;

        let pipe_layout_name = format!("{name}-pipe-layout");
        let pipeline_layout = pipeline::create_pipeline_layout(
            ctx,
            &[descriptor_set_layout],
            push_constant_size,
            &pipe_layout_name,
        )?;

        let pipe_name = format!("{name}-pipeline");
        let compute_pipeline = pipeline::create_compute_pipeline(
            ctx,
            shader_module,
            entry_point,
            pipeline_layout,
            &pipe_name,
        )?;

        // Allocate and update descriptor set
        let sets =
            pipeline::allocate_descriptor_sets(ctx, descriptor_pool, &[descriptor_set_layout])?;
        let descriptor_set = sets[0];

        let buffer_bindings: Vec<pipeline::BufferBinding<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| pipeline::BufferBinding {
                set: descriptor_set,
                binding: i as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer: buf,
            })
            .collect();
        pipeline::update_descriptor_sets(ctx, &buffer_bindings);

        Ok(ComputePass {
            pipeline: compute_pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_set,
        })
    }

    /// Record a compute dispatch for a single pass.
    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        pass: &ComputePass,
        group_x: u32,
        group_y: u32,
        group_z: u32,
        push_constants: &[u8],
    ) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pass.pipeline_layout,
                0,
                &[pass.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    cmd,
                    pass.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            self.device.cmd_dispatch(cmd, group_x, group_y, group_z);
        }
    }

    /// Insert a compute-to-compute memory barrier.
    fn barrier(cmd: vk::CommandBuffer, device: &ash::Device) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
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
        buffer::destroy_buffer(ctx, particle_buf);
        buffer::destroy_buffer(ctx, grid_buf);
        buffer::destroy_buffer(ctx, voxel_buf);
        buffer::destroy_buffer(ctx, render_output_buf);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn init_tracing() {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    }

    #[test]
    fn push_constant_sizes() {
        assert_eq!(mem::size_of::<P2gPushConstants>(), 16);
        assert_eq!(mem::size_of::<GridUpdatePushConstants>(), 16);
        assert_eq!(mem::size_of::<G2pPushConstants>(), 16);
        assert_eq!(mem::size_of::<VoxelizePushConstants>(), 16);
        assert_eq!(mem::size_of::<ReactPushConstants>(), 16);
        // RenderPushConstants: 4 u32s (16 bytes) + 2 Vec4s (32 bytes) = 48 bytes
        assert_eq!(mem::size_of::<RenderPushConstants>(), 48);
        // ToolbarPushConstants: 4 u32s (16 bytes) + 8 Vec4s (128 bytes) = 144 bytes
        assert_eq!(mem::size_of::<ToolbarPushConstants>(), 144);
    }

    #[test]
    fn create_and_destroy() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");
        sim.destroy(&ctx);
    }

    #[test]
    fn init_and_readback_particles() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let particles = vec![
            Particle::new(Vec3::new(16.0, 16.0, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(17.0, 16.0, 16.0), 1.0, 1, 1),
        ];

        sim.init_particles(&ctx, &particles)
            .expect("Failed to init particles");
        assert_eq!(sim.num_particles(), 2);

        let readback = sim
            .readback_particles(&ctx)
            .expect("Failed to readback particles");
        assert_eq!(readback.len(), 2);
        assert!((readback[0].position() - Vec3::new(16.0, 16.0, 16.0)).length() < 1e-5);
        assert!((readback[1].position() - Vec3::new(17.0, 16.0, 16.0)).length() < 1e-5);

        sim.destroy(&ctx);
    }

    #[test]
    fn step_does_not_crash() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let particles = vec![
            Particle::new(Vec3::new(16.0, 20.0, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(16.0, 20.5, 16.0), 1.0, 0, 0),
            Particle::new(Vec3::new(16.5, 20.0, 16.0), 1.0, 0, 0),
        ];

        sim.init_particles(&ctx, &particles)
            .expect("Failed to init particles");

        // Execute one simulation step
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd);
        })
        .expect("Simulation step failed");

        let readback = sim
            .readback_particles(&ctx)
            .expect("Failed to readback after step");
        assert_eq!(readback.len(), 3);

        // After one step with gravity, particles should have moved slightly downward
        // (or at least not be NaN/Inf)
        for (i, p) in readback.iter().enumerate() {
            let pos = p.position();
            assert!(
                pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "Particle {i} has non-finite position: {pos:?}"
            );
        }

        sim.destroy(&ctx);
    }

    #[test]
    fn too_many_particles_returns_error() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut sim = GpuSimulation::new(&ctx).expect("Failed to create GpuSimulation");

        let too_many = vec![Particle::new(Vec3::ZERO, 1.0, 0, 0); MAX_PARTICLES as usize + 1];
        let result = sim.init_particles(&ctx, &too_many);
        assert!(result.is_err());

        sim.destroy(&ctx);
    }
}
