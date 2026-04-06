//! CA substrate simulation orchestrator.
//!
//! [`CaSimulation`] manages a [`ChunkPool`] and dispatches CA compute passes
//! (compact, thermal diffusion, Margolus block automaton) on dirty chunks
//! via indirect dispatch.

use std::collections::HashMap;
use std::mem;

use ash::vk;
use bytemuck::Zeroable;

use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
    chunk_pool::ChunkPool,
    pipeline,
};
use shared::{
    chunk_gpu::ChunkGpuMeta,
    constants::*,
    material_ca::MaterialPropertiesCA,
    reaction_ca::ReactionEntry,
    voxel::Voxel,
};

use crate::passes::{self, ComputePass};
use crate::pbmpm_zones::{ActivationTrigger, PbmpmZoneManager};
use crate::push_constants::{DirtyTilesPushConstants, RenderPushConstants};

/// Compiled SPIR-V shader module bytes, included at compile time.
const SHADER_BYTES: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));

/// Workgroups per chunk for thermal pass: 32/4 = 8 per axis, 8^3 = 512.
const THERMAL_WG_PER_CHUNK: u32 = 512;

/// Workgroups per chunk for Margolus pass: 4x4x4 = 64.
const MARGOLUS_WG_PER_CHUNK: u32 = 64;

// ---------------------------------------------------------------------------
// Push constant structs (mirroring shader-side definitions)
// ---------------------------------------------------------------------------

/// Push constants for the CA compact pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CaCompactPush {
    /// Total number of loaded chunks.
    pub total_chunks: u32,
    /// Workgroups-per-chunk for subsequent indirect dispatch.
    pub workgroups_per_chunk: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 2],
}

/// Push constants for the CA thermal pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CaThermalPush {
    /// Current frame number.
    pub frame_number: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
}

/// Push constants for the CA Margolus pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CaMargolusPush {
    /// Current frame number.
    pub frame_number: u32,
    /// Block offset X (0 for even, 1 for odd).
    pub offset_x: u32,
    /// Block offset Y (0 for even, 1 for odd).
    pub offset_y: u32,
    /// Block offset Z (0 for even, 1 for odd).
    pub offset_z: u32,
    /// Number of active reactions.
    pub num_reactions: u32,
    /// Padding to 32-byte alignment.
    pub _pad: [u32; 3],
}

/// Push constants for the CA-to-render conversion shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CaToRenderPush {
    /// Number of loaded chunks to process.
    pub num_chunks: u32,
    /// Flat render grid dimension.
    pub grid_size: u32,
    /// World offset of the flat grid origin (X).
    pub grid_origin_x: i32,
    /// World offset of the flat grid origin (Y).
    pub grid_origin_y: i32,
    /// World offset of the flat grid origin (Z).
    pub grid_origin_z: i32,
    /// Padding to 32-byte alignment.
    pub _pad: [u32; 3],
}

/// Render grid size for the CA-to-render bridge (128^3 voxels).
pub const CA_RENDER_GRID_SIZE: u32 = 128;

/// Number of bricks per axis for the CA render grid (128 / 8 = 16).
const CA_BRICKS_PER_AXIS: u32 = CA_RENDER_GRID_SIZE / 8;
/// Total bricks in the CA render grid.
const CA_TOTAL_BRICKS: u32 = CA_BRICKS_PER_AXIS * CA_BRICKS_PER_AXIS * CA_BRICKS_PER_AXIS;

/// Super-brick size (in bricks per axis).
const CA_SUPER_BRICK_BRICKS: u32 = 8;
/// Super-bricks per axis.
const CA_SUPER_BRICKS_PER_AXIS: u32 = CA_BRICKS_PER_AXIS / CA_SUPER_BRICK_BRICKS;
/// Total super-bricks.
const CA_TOTAL_SUPER_BRICKS: u32 =
    CA_SUPER_BRICKS_PER_AXIS * CA_SUPER_BRICKS_PER_AXIS * CA_SUPER_BRICKS_PER_AXIS;

/// CA substrate simulation. Manages chunk pool and dispatches CA compute passes.
pub struct CaSimulation {
    // GPU resources
    chunk_pool: ChunkPool,
    material_buffer: GpuBuffer,
    reaction_buffer: GpuBuffer,

    // Compute passes
    compact_pass: ComputePass,
    compact_finalize_pass: ComputePass,
    thermal_pass: ComputePass,
    margolus_pass: ComputePass,

    // Shader module (kept alive for pipeline lifetime)
    shader_module: vk::ShaderModule,

    // Descriptor pool (owns all descriptor sets)
    descriptor_pool: vk::DescriptorPool,

    // PB-MPM zone manager (optional, created alongside CA)
    pbmpm: Option<PbmpmZoneManager>,

    // Render bridge
    render_voxel_buffer: GpuBuffer,
    render_output_buffer: GpuBuffer,
    prev_render_output_buffer: GpuBuffer,
    brick_occupancy_buffer: GpuBuffer,
    super_brick_occupancy_buffer: GpuBuffer,
    dirty_tile_buffer: GpuBuffer,
    material_render_buffer: GpuBuffer,
    sleep_state_buffer: GpuBuffer,
    ca_to_render_pass: ComputePass,
    render_pass: ComputePass,
    compute_dirty_tiles_pass: ComputePass,

    // CPU state
    loaded_chunks: HashMap<[i32; 3], u32>,
    frame_number: u32,
    num_reactions: u32,

    // Cached device handle for recording commands
    device: ash::Device,
}

impl CaSimulation {
    /// Create a new CA simulation orchestrator.
    ///
    /// Allocates a chunk pool, uploads material and reaction tables,
    /// and creates compute pipelines for all CA passes.
    pub fn new(
        ctx: &VulkanContext,
        materials: &[MaterialPropertiesCA],
        reactions: &[ReactionEntry],
        chunk_slots: u32,
    ) -> anyhow::Result<Self> {
        tracing::info!(
            "Creating CaSimulation: {} material(s), {} reaction(s), {} chunk slots",
            materials.len(),
            reactions.len(),
            chunk_slots,
        );

        // 1. Create ChunkPool
        let chunk_pool = ChunkPool::new(ctx, chunk_slots)?;

        // 2. Upload material table
        let mat_buf_size =
            (materials.len() * mem::size_of::<MaterialPropertiesCA>()) as vk::DeviceSize;
        let material_buffer = buffer::create_device_local_buffer(
            ctx,
            mat_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ca-material-buffer",
        )?;
        buffer::upload(ctx, materials, &material_buffer)?;

        // 3. Upload reaction table
        // Ensure at least 1 element to avoid zero-size buffer
        let reaction_count = reactions.len().max(1);
        let rxn_buf_size =
            (reaction_count * mem::size_of::<ReactionEntry>()) as vk::DeviceSize;
        let reaction_buffer = buffer::create_device_local_buffer(
            ctx,
            rxn_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ca-reaction-buffer",
        )?;
        if !reactions.is_empty() {
            buffer::upload(ctx, reactions, &reaction_buffer)?;
        }

        // 4. Load SPIR-V shader module
        let shader_module =
            pipeline::create_shader_module(ctx, SHADER_BYTES, "ca-shaders")?;

        // 5. Create descriptor pool
        // Total storage buffer bindings across all passes:
        // compact: 2, compact_finalize: 2, thermal: 4, margolus: 5,
        // ca_to_render: 4, render: 7, compute_dirty_tiles: 2 = 26 total
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 32, // headroom for all passes
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 7, "ca-descriptor-pool")?;

        // 6. Create compute passes

        // compact pass: binding 0 = metadata, binding 1 = dirty_list
        let compact_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::ca_compact::ca_compact",
            &[chunk_pool.metadata_buffer(), chunk_pool.dirty_list_buffer()],
            mem::size_of::<CaCompactPush>() as u32,
            descriptor_pool,
            "ca-compact",
        )?;

        // compact_finalize pass: binding 0 = dirty_list (note: shader only has binding 1,
        // but the entry point uses descriptor_set=0, binding=1 for dirty_list.
        // However, create_pass assigns sequential bindings 0,1,2,...
        // We need to match what the shader expects.)
        //
        // Looking at ca_compact_finalize: it takes binding 1 = dirty_list.
        // But create_pass assigns binding 0,1,2... to the buffers in order.
        // So we need to pass metadata as a dummy at binding 0, and dirty_list at binding 1.
        let compact_finalize_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::ca_compact::ca_compact_finalize",
            &[chunk_pool.metadata_buffer(), chunk_pool.dirty_list_buffer()],
            mem::size_of::<CaCompactPush>() as u32,
            descriptor_pool,
            "ca-compact-finalize",
        )?;

        // thermal pass: binding 0 = voxel_buffer, 1 = metadata, 2 = dirty_list, 3 = materials
        let thermal_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::ca_thermal::ca_thermal",
            &[
                chunk_pool.voxel_buffer(),
                chunk_pool.metadata_buffer(),
                chunk_pool.dirty_list_buffer(),
                &material_buffer,
            ],
            mem::size_of::<CaThermalPush>() as u32,
            descriptor_pool,
            "ca-thermal",
        )?;

        // margolus pass: binding 0 = voxel_buffer, 1 = metadata, 2 = dirty_list,
        //                3 = materials, 4 = reactions
        let margolus_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::ca_margolus::ca_margolus",
            &[
                chunk_pool.voxel_buffer(),
                chunk_pool.metadata_buffer(),
                chunk_pool.dirty_list_buffer(),
                &material_buffer,
                &reaction_buffer,
            ],
            mem::size_of::<CaMargolusPush>() as u32,
            descriptor_pool,
            "ca-margolus",
        )?;

        // 7. Create render bridge buffers

        // Flat render voxel buffer: CA_RENDER_GRID_SIZE^3 * 4 u32s * 4 bytes = 32 MB
        let render_voxel_buf_size = (CA_RENDER_GRID_SIZE as u64)
            .pow(3)
            * 4
            * mem::size_of::<u32>() as u64;
        let render_voxel_buffer = buffer::create_device_local_buffer(
            ctx,
            render_voxel_buf_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-render-voxel-buffer",
        )?;
        tracing::info!(
            "CA render voxel buffer: {:.1}MB ({}^3 grid)",
            render_voxel_buf_size as f64 / (1024.0 * 1024.0),
            CA_RENDER_GRID_SIZE,
        );

        // Render output buffer: BGRA u32 per pixel
        let render_output_size = (shared::RENDER_WIDTH as u64)
            * (shared::RENDER_HEIGHT as u64)
            * mem::size_of::<u32>() as u64;
        let render_output_buffer = buffer::create_device_local_buffer(
            ctx,
            render_output_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "ca-render-output-buffer",
        )?;
        let prev_render_output_buffer = buffer::create_device_local_buffer(
            ctx,
            render_output_size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-prev-render-output-buffer",
        )?;

        // Brick occupancy: 1 u32 per brick
        let brick_occupancy_buffer = buffer::create_device_local_buffer(
            ctx,
            (CA_TOTAL_BRICKS as u64 * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-brick-occupancy-buffer",
        )?;

        // Super-brick occupancy
        let super_brick_count = CA_TOTAL_SUPER_BRICKS.max(1);
        let super_brick_occupancy_buffer = buffer::create_device_local_buffer(
            ctx,
            (super_brick_count as u64 * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-super-brick-occupancy-buffer",
        )?;

        // Dirty tile buffer
        let dirty_tile_buffer = buffer::create_device_local_buffer(
            ctx,
            (shared::DIRTY_TILE_COUNT as u64 * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-dirty-tile-buffer",
        )?;

        // Material render buffer: MaterialParams for the render shader
        // Map CA materials to render-compatible MaterialParams
        let render_materials = ca_materials_to_render_params(materials);
        let mat_render_buf_size =
            (render_materials.len() * mem::size_of::<shared::material::MaterialParams>())
                as vk::DeviceSize;
        let material_render_buffer = buffer::create_device_local_buffer(
            ctx,
            mat_render_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ca-material-render-buffer",
        )?;
        buffer::upload(ctx, &render_materials, &material_render_buffer)?;

        // Sleep state buffer (dummy, filled with zeros = all awake)
        // Needed by compute_dirty_tiles pass
        let sleep_state_buffer = buffer::create_device_local_buffer(
            ctx,
            (CA_TOTAL_BRICKS as u64 * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "ca-sleep-state-buffer",
        )?;
        let zeros = vec![0u32; CA_TOTAL_BRICKS as usize];
        buffer::upload(ctx, &zeros, &sleep_state_buffer)?;

        // 8. Create render bridge compute passes

        // ca_to_render: chunk_pool(r), metadata(r), render_voxels(w), ca_materials(r)
        let ca_to_render_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::ca_to_render::ca_to_render",
            &[
                chunk_pool.voxel_buffer(),
                chunk_pool.metadata_buffer(),
                &render_voxel_buffer,
                &material_buffer,
            ],
            mem::size_of::<CaToRenderPush>() as u32,
            descriptor_pool,
            "ca-to-render",
        )?;

        // compute_dirty_tiles: sleep_state(r), dirty_tiles(w)
        let compute_dirty_tiles_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::compute_dirty_tiles::compute_dirty_tiles",
            &[&sleep_state_buffer, &dirty_tile_buffer],
            mem::size_of::<DirtyTilesPushConstants>() as u32,
            descriptor_pool,
            "ca-compute-dirty-tiles",
        )?;

        // render_pass: voxels(r), output(w), materials(r), brick_occ(r),
        //              super_brick_occ(r), dirty_tiles(r), prev_output(r)
        let render_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::render::render_voxels",
            &[
                &render_voxel_buffer,
                &render_output_buffer,
                &material_render_buffer,
                &brick_occupancy_buffer,
                &super_brick_occupancy_buffer,
                &dirty_tile_buffer,
                &prev_render_output_buffer,
            ],
            mem::size_of::<RenderPushConstants>() as u32,
            descriptor_pool,
            "ca-render",
        )?;

        // Clean up the sleep state buffer (owned by ca_simulation, not stored)
        // Actually we need it for dirty tiles, but the buffer ref is captured in the descriptor set.
        // We must keep it alive. Store it? No -- we just need it for the descriptor set binding.
        // The descriptor set references the buffer, so we MUST keep it alive.
        // Let's store it as an extra field... Actually, let's just leak-proof it by keeping
        // a reference. For simplicity, store it.

        // Create PB-MPM zone manager (non-fatal if it fails)
        let pbmpm = match PbmpmZoneManager::new(ctx) {
            Ok(mgr) => {
                tracing::info!("PB-MPM zone manager created successfully");
                Some(mgr)
            }
            Err(e) => {
                tracing::warn!("Failed to create PB-MPM zone manager: {}", e);
                None
            }
        };

        Ok(Self {
            chunk_pool,
            material_buffer,
            reaction_buffer,
            compact_pass,
            compact_finalize_pass,
            thermal_pass,
            margolus_pass,
            shader_module,
            descriptor_pool,
            pbmpm,
            render_voxel_buffer,
            render_output_buffer,
            prev_render_output_buffer,
            brick_occupancy_buffer,
            super_brick_occupancy_buffer,
            dirty_tile_buffer,
            material_render_buffer,
            sleep_state_buffer,
            ca_to_render_pass,
            render_pass,
            compute_dirty_tiles_pass,
            loaded_chunks: HashMap::new(),
            frame_number: 0,
            num_reactions: reactions.len() as u32,
            device: ctx.device.clone(),
        })
    }

    /// Record one CA simulation step into the given command buffer.
    ///
    /// Dispatches: compact -> finalize(thermal) -> thermal(indirect) ->
    /// finalize(margolus) -> margolus_even(indirect) -> margolus_odd(indirect).
    pub fn step(&mut self, cmd: vk::CommandBuffer, _ctx: &VulkanContext) {
        let total_loaded = self.loaded_chunks.len() as u32;
        if total_loaded == 0 {
            self.frame_number += 1;
            return;
        }

        // 1. Clear dirty_list counter to 0 (first 4 bytes)
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                self.chunk_pool.dirty_list_buffer().buffer,
                0,
                4,
                0,
            );
        }

        // Barrier: transfer -> compute
        let transfer_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[transfer_barrier],
                &[],
                &[],
            );
        }

        // 2. Dispatch compact pass
        let compact_wg = (total_loaded + 255) / 256;
        let compact_push = CaCompactPush {
            total_chunks: total_loaded,
            workgroups_per_chunk: THERMAL_WG_PER_CHUNK, // will be overridden by finalize
            _pad: [0; 2],
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.compact_pass,
            compact_wg,
            1,
            1,
            bytemuck::bytes_of(&compact_push),
        );
        passes::barrier(cmd, &self.device);

        // 3. Dispatch compact_finalize for thermal (wg_per_chunk = 512)
        let finalize_thermal_push = CaCompactPush {
            total_chunks: total_loaded,
            workgroups_per_chunk: THERMAL_WG_PER_CHUNK,
            _pad: [0; 2],
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.compact_finalize_pass,
            1,
            1,
            1,
            bytemuck::bytes_of(&finalize_thermal_push),
        );

        // Barrier: compute -> indirect command read + compute
        let indirect_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ,
            );
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[indirect_barrier],
                &[],
                &[],
            );
        }

        // 4. Dispatch thermal pass (INDIRECT)
        let thermal_push = CaThermalPush {
            frame_number: self.frame_number,
            _pad: [0; 3],
        };
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.thermal_pass.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.thermal_pass.pipeline_layout,
                0,
                &[self.thermal_pass.descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.thermal_pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&thermal_push),
            );
        }
        // Indirect dispatch: VkDispatchIndirectCommand at dirty_list byte offset 4
        pipeline::cmd_dispatch_indirect(
            &self.device,
            cmd,
            self.chunk_pool.dirty_list_buffer(),
            4,
        );
        passes::barrier(cmd, &self.device);

        // 5. Re-finalize for margolus (wg_per_chunk = 64)
        let finalize_margolus_push = CaCompactPush {
            total_chunks: total_loaded,
            workgroups_per_chunk: MARGOLUS_WG_PER_CHUNK,
            _pad: [0; 2],
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.compact_finalize_pass,
            1,
            1,
            1,
            bytemuck::bytes_of(&finalize_margolus_push),
        );

        // Barrier: compute -> indirect + compute
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[indirect_barrier],
                &[],
                &[],
            );
        }

        // 6. Dispatch margolus even pass (offset = 0,0,0)
        let margolus_even_push = CaMargolusPush {
            frame_number: self.frame_number,
            offset_x: 0,
            offset_y: 0,
            offset_z: 0,
            num_reactions: self.num_reactions,
            _pad: [0; 3],
        };
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.margolus_pass.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.margolus_pass.pipeline_layout,
                0,
                &[self.margolus_pass.descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.margolus_pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&margolus_even_push),
            );
        }
        pipeline::cmd_dispatch_indirect(
            &self.device,
            cmd,
            self.chunk_pool.dirty_list_buffer(),
            4,
        );
        passes::barrier(cmd, &self.device);

        // 7. Dispatch margolus odd pass (offset = 1,1,1)
        let margolus_odd_push = CaMargolusPush {
            frame_number: self.frame_number,
            offset_x: 1,
            offset_y: 1,
            offset_z: 1,
            num_reactions: self.num_reactions,
            _pad: [0; 3],
        };
        unsafe {
            // Pipeline and descriptor set already bound from even pass
            self.device.cmd_push_constants(
                cmd,
                self.margolus_pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&margolus_odd_push),
            );
        }
        pipeline::cmd_dispatch_indirect(
            &self.device,
            cmd,
            self.chunk_pool.dirty_list_buffer(),
            4,
        );
        passes::barrier(cmd, &self.device);

        // After CA: dispatch PB-MPM for active zones
        if let Some(ref pbmpm) = self.pbmpm {
            pbmpm.step(cmd, _ctx, 0.016); // ~60fps
        }
        if let Some(ref mut pbmpm) = self.pbmpm {
            pbmpm.check_sleep();
        }

        self.frame_number += 1;
    }

    /// Convert CA chunks to flat render voxels, then dispatch the ray-march render.
    ///
    /// This is the "bridge" that reuses the existing render pipeline with CA data.
    /// Steps:
    /// 1. Clear flat voxel buffer
    /// 2. Dispatch ca_to_render shader (CA chunks -> flat voxels)
    /// 3. Fill brick/super-brick occupancy with 1 (all occupied, POC)
    /// 4. Mark all dirty tiles
    /// 5. Dispatch existing render shader
    pub fn render(
        &self,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        eye: [f32; 3],
        target: [f32; 3],
    ) {
        let num_chunks = self.loaded_chunks.len() as u32;

        // 1. Clear flat voxel buffer to zero
        let voxel_buf_size = (CA_RENDER_GRID_SIZE as u64).pow(3) * 4 * 4;
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                self.render_voxel_buffer.buffer,
                0,
                voxel_buf_size,
                0,
            );
        }

        // Transfer -> compute barrier
        let transfer_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[transfer_barrier],
                &[],
                &[],
            );
        }

        // 2. Dispatch ca_to_render: convert CA chunks to flat voxel buffer
        if num_chunks > 0 {
            let push = CaToRenderPush {
                num_chunks,
                grid_size: CA_RENDER_GRID_SIZE,
                grid_origin_x: 0,
                grid_origin_y: 0,
                grid_origin_z: 0,
                _pad: [0; 3],
            };
            // Each chunk is 32x32x32, workgroup is 4x4x4, so we need 8x8x8 WGs per chunk
            // X dimension covers all chunks * 32 / 4 = num_chunks * 8
            let wg_x = num_chunks * 8;
            let wg_y = 8; // 32 / 4
            let wg_z = 8; // 32 / 4
            passes::dispatch(
                &self.device,
                cmd,
                &self.ca_to_render_pass,
                wg_x,
                wg_y,
                wg_z,
                bytemuck::bytes_of(&push),
            );
            passes::barrier(cmd, &self.device);
        }

        // 3. Fill brick occupancy with 1 (all occupied for POC)
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                self.brick_occupancy_buffer.buffer,
                0,
                CA_TOTAL_BRICKS as u64 * 4,
                1,
            );
            self.device.cmd_fill_buffer(
                cmd,
                self.super_brick_occupancy_buffer.buffer,
                0,
                CA_TOTAL_SUPER_BRICKS.max(1) as u64 * 4,
                1,
            );
        }

        // 4. Mark all dirty tiles (fill with 1)
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                self.dirty_tile_buffer.buffer,
                0,
                shared::DIRTY_TILE_COUNT as u64 * 4,
                1,
            );
        }

        // Barrier: transfer -> compute
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[transfer_barrier],
                &[],
                &[],
            );
        }

        // 5. Dispatch render shader
        let render_push = RenderPushConstants {
            width,
            height,
            grid_size: CA_RENDER_GRID_SIZE,
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
            bytemuck::bytes_of(&render_push),
        );
    }

    /// Returns the Vulkan buffer handle for the render output.
    ///
    /// The buffer contains packed BGRA u32 pixels, sized `width * height`.
    pub fn render_output_buffer(&self) -> vk::Buffer {
        self.render_output_buffer.buffer
    }

    /// Insert a final barrier and copy current render output to the previous-frame buffer.
    ///
    /// Must be called after [`render`].
    pub fn finalize_render(&self, cmd: vk::CommandBuffer) {
        // Barrier: SHADER_WRITE -> TRANSFER_READ | TRANSFER_WRITE
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
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

        // Copy current render output to previous frame buffer
        let copy_size = self.render_output_buffer.size;
        let region = vk::BufferCopy::default().size(copy_size);
        unsafe {
            self.device.cmd_copy_buffer(
                cmd,
                self.render_output_buffer.buffer,
                self.prev_render_output_buffer.buffer,
                &[region],
            );
        }

        // Barrier: ensure the copy finishes before next frame's shader reads
        let copy_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[copy_barrier],
                &[],
                &[],
            );
        }
    }

    /// Activate a PB-MPM physics zone at the given trigger location.
    ///
    /// Spawns particles from the chunk at the trigger position and starts
    /// PB-MPM simulation for the zone. Returns the zone index if successful.
    pub fn activate_physics_zone(
        &mut self,
        ctx: &VulkanContext,
        trigger: ActivationTrigger,
    ) -> anyhow::Result<Option<usize>> {
        let pbmpm = self
            .pbmpm
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("PB-MPM zone manager not available"))?;

        let zone_idx = match pbmpm.activate_zone(&trigger, 32) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        // Get voxel data from the chunk at the trigger location
        let chunk_coord = [
            (trigger.center[0] / 32.0).floor() as i32,
            (trigger.center[1] / 32.0).floor() as i32,
            (trigger.center[2] / 32.0).floor() as i32,
        ];

        if let Some(&slot_id) = self.loaded_chunks.get(&chunk_coord) {
            let mut voxel_data = self.chunk_pool.download_chunk_voxels(ctx, slot_id)?;
            pbmpm.spawn_particles_from_voxels(
                ctx,
                zone_idx,
                &voxel_data,
                chunk_coord,
                &trigger,
            )?;

            // Clear voxels in the explosion radius to create a visible crater
            let r = trigger.radius as i32;
            let cx_local = (trigger.center[0] - chunk_coord[0] as f32 * 32.0) as i32;
            let cy_local = (trigger.center[1] - chunk_coord[1] as f32 * 32.0) as i32;
            let cz_local = (trigger.center[2] - chunk_coord[2] as f32 * 32.0) as i32;
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy + dz * dz > r * r {
                            continue;
                        }
                        let lx = cx_local + dx;
                        let ly = cy_local + dy;
                        let lz = cz_local + dz;
                        if lx >= 0 && lx < 32 && ly >= 0 && ly < 32 && lz >= 0 && lz < 32 {
                            let idx = lz as usize * 32 * 32 + ly as usize * 32 + lx as usize;
                            voxel_data[idx] = 0; // air
                        }
                    }
                }
            }
            // Re-upload the modified chunk with the crater
            self.chunk_pool
                .upload_chunk_voxels(ctx, vk::CommandBuffer::null(), slot_id, &voxel_data)?;
        }

        Ok(Some(zone_idx))
    }

    /// Get a reference to the PB-MPM zone manager, if available.
    pub fn pbmpm(&self) -> Option<&PbmpmZoneManager> {
        self.pbmpm.as_ref()
    }

    /// Load a chunk into the simulation.
    ///
    /// `voxel_data` must contain exactly `CA_CHUNK_VOXELS` (32768) u32 values.
    /// `neighbors` contains slot IDs of the 6 face-neighbors (+X, -X, +Y, -Y, +Z, -Z),
    /// or -1 if the neighbor is not loaded.
    pub fn load_chunk(
        &mut self,
        ctx: &VulkanContext,
        coord: [i32; 3],
        voxel_data: &[u32],
        neighbors: [i32; 6],
    ) -> anyhow::Result<u32> {
        let slot_id = self
            .chunk_pool
            .allocate_slot()
            .ok_or_else(|| anyhow::anyhow!("chunk pool full"))?;

        // Upload voxel data
        self.chunk_pool
            .upload_chunk_voxels(ctx, vk::CommandBuffer::null(), slot_id, voxel_data)?;

        // Upload metadata
        let meta = ChunkGpuMeta {
            world_pos: glam::IVec4::new(coord[0], coord[1], coord[2], slot_id as i32),
            neighbor_ids: neighbors,
            activity: 1, // CAOnly
            flags: 1,    // dirty
            _pad: [0; 4],
        };
        let meta_bytes = bytemuck::bytes_of(&meta);
        self.chunk_pool
            .upload_metadata(ctx, vk::CommandBuffer::null(), slot_id, meta_bytes)?;

        self.loaded_chunks.insert(coord, slot_id);
        tracing::debug!(
            "Loaded chunk {:?} into slot {}",
            coord,
            slot_id,
        );
        Ok(slot_id)
    }

    /// Unload a chunk, optionally downloading its voxel data first.
    pub fn unload_chunk(
        &mut self,
        ctx: &VulkanContext,
        coord: [i32; 3],
        download: bool,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let slot_id = self
            .loaded_chunks
            .remove(&coord)
            .ok_or_else(|| anyhow::anyhow!("chunk not loaded: {:?}", coord))?;

        let data = if download {
            Some(self.chunk_pool.download_chunk_voxels(ctx, slot_id)?)
        } else {
            None
        };

        self.chunk_pool.free_slot(slot_id);
        tracing::debug!("Unloaded chunk {:?} from slot {}", coord, slot_id);
        Ok(data)
    }

    /// Download voxel data for a loaded chunk (for testing/verification).
    pub fn download_chunk(
        &self,
        ctx: &VulkanContext,
        coord: [i32; 3],
    ) -> anyhow::Result<Vec<u32>> {
        let slot_id = *self
            .loaded_chunks
            .get(&coord)
            .ok_or_else(|| anyhow::anyhow!("chunk not loaded: {:?}", coord))?;
        Ok(self.chunk_pool.download_chunk_voxels(ctx, slot_id)?)
    }

    /// Re-upload modified voxel data for a loaded chunk.
    ///
    /// `voxel_data` must contain exactly `CA_CHUNK_VOXELS` u32 values.
    pub fn upload_chunk_voxels(
        &self,
        ctx: &VulkanContext,
        coord: [i32; 3],
        voxel_data: &[u32],
    ) -> anyhow::Result<()> {
        let slot_id = *self
            .loaded_chunks
            .get(&coord)
            .ok_or_else(|| anyhow::anyhow!("chunk not loaded at {:?}", coord))?;
        self.chunk_pool
            .upload_chunk_voxels(ctx, vk::CommandBuffer::null(), slot_id, voxel_data)?;
        Ok(())
    }

    /// Debug: readback the first N u32 values from the dirty list buffer.
    ///
    /// Useful for verifying that compact/finalize wrote the expected dirty count
    /// and indirect dispatch args. Returns up to `count` u32 values.
    pub fn debug_readback_dirty_list(
        &self,
        ctx: &VulkanContext,
        count: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let byte_size = (count * 4) as vk::DeviceSize;
        let staging =
            buffer::create_readback_staging_buffer(ctx, byte_size, "dirty-list-readback")?;

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default().size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    self.chunk_pool.dirty_list_buffer().buffer,
                    staging.buffer,
                    &[region],
                );
            }
        })?;

        let result = {
            let mapped = staging
                .mapped_slice()
                .ok_or_else(|| anyhow::anyhow!("failed to map dirty list staging buffer"))?;
            let typed: &[u32] = bytemuck::cast_slice(&mapped[..byte_size as usize]);
            typed.to_vec()
        };
        buffer::destroy_buffer(ctx, staging);
        Ok(result)
    }

    /// Debug: readback metadata for a slot (returns 16 u32 values = 64 bytes).
    pub fn debug_readback_metadata(
        &self,
        ctx: &VulkanContext,
        slot_id: u32,
    ) -> anyhow::Result<Vec<u32>> {
        let byte_size = 64u64; // ChunkGpuMeta is 64 bytes
        let staging =
            buffer::create_readback_staging_buffer(ctx, byte_size, "metadata-readback")?;
        let src_offset = u64::from(slot_id) * 64;

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    self.chunk_pool.metadata_buffer().buffer,
                    staging.buffer,
                    &[region],
                );
            }
        })?;

        let result = {
            let mapped = staging
                .mapped_slice()
                .ok_or_else(|| anyhow::anyhow!("failed to map metadata staging buffer"))?;
            let typed: &[u32] = bytemuck::cast_slice(&mapped[..byte_size as usize]);
            typed.to_vec()
        };
        buffer::destroy_buffer(ctx, staging);
        Ok(result)
    }

    /// Number of loaded chunks.
    pub fn loaded_count(&self) -> usize {
        self.loaded_chunks.len()
    }

    /// Current frame number.
    pub fn frame_number(&self) -> u32 {
        self.frame_number
    }

    /// Free all GPU resources.
    ///
    /// Must be called before dropping the [`VulkanContext`].
    pub fn destroy(self, ctx: &VulkanContext) {
        // Destroy pipelines and layouts
        for pass in [
            &self.compact_pass,
            &self.compact_finalize_pass,
            &self.thermal_pass,
            &self.margolus_pass,
            &self.ca_to_render_pass,
            &self.render_pass,
            &self.compute_dirty_tiles_pass,
        ] {
            pipeline::destroy_pipeline(ctx, pass.pipeline);
            pipeline::destroy_pipeline_layout(ctx, pass.pipeline_layout);
            pipeline::destroy_descriptor_set_layout(ctx, pass.descriptor_set_layout);
        }

        pipeline::destroy_descriptor_pool(ctx, self.descriptor_pool);
        pipeline::destroy_shader_module(ctx, self.shader_module);

        if let Some(pbmpm) = self.pbmpm {
            pbmpm.destroy(ctx);
        }

        buffer::destroy_buffer(ctx, self.material_buffer);
        buffer::destroy_buffer(ctx, self.reaction_buffer);
        buffer::destroy_buffer(ctx, self.render_voxel_buffer);
        buffer::destroy_buffer(ctx, self.render_output_buffer);
        buffer::destroy_buffer(ctx, self.prev_render_output_buffer);
        buffer::destroy_buffer(ctx, self.brick_occupancy_buffer);
        buffer::destroy_buffer(ctx, self.super_brick_occupancy_buffer);
        buffer::destroy_buffer(ctx, self.dirty_tile_buffer);
        buffer::destroy_buffer(ctx, self.material_render_buffer);
        buffer::destroy_buffer(ctx, self.sleep_state_buffer);
        self.chunk_pool.destroy(ctx);

        tracing::info!("CaSimulation destroyed");
    }
}

// ---------------------------------------------------------------------------
// Default material and reaction tables
// ---------------------------------------------------------------------------

/// Create default material table for the CA simulation.
pub fn default_ca_materials() -> Vec<MaterialPropertiesCA> {
    let mut mats = vec![MaterialPropertiesCA::zeroed(); 20];

    // 0 = Air (all zeros — already zeroed)

    // 1 = Stone
    mats[1].phase = 0;
    mats[1].density = 200;
    mats[1].conductivity = 2;
    mats[1].melt_temp = 200;
    mats[1].melt_into = 4;
    mats[1].strength = 100;

    // 2 = Sand
    mats[2].phase = 1;
    mats[2].density = 150;
    mats[2].conductivity = 1;

    // 3 = Water
    mats[3].phase = 2;
    mats[3].density = 100;
    mats[3].conductivity = 4;
    mats[3].freeze_temp = 50;
    mats[3].freeze_into = 6;
    mats[3].boil_temp = 200;
    mats[3].boil_into = 5;

    // 4 = Lava
    mats[4].phase = 2;
    mats[4].density = 200;
    mats[4].conductivity = 8;
    mats[4].freeze_temp = 150;
    mats[4].freeze_into = 1;

    // 5 = Steam
    mats[5].phase = 3;
    mats[5].density = 10;
    mats[5].conductivity = 1;
    mats[5].freeze_temp = 100;
    mats[5].freeze_into = 3;

    // 6 = Ice
    mats[6].phase = 0;
    mats[6].density = 90;
    mats[6].conductivity = 3;
    mats[6].melt_temp = 50;
    mats[6].melt_into = 3;

    // 7-9 = Stone variants (same physics, different render color)
    for i in 7..=9 {
        mats[i] = mats[1]; // copy stone properties
    }

    mats
}

/// Create default reaction table for the CA simulation.
pub fn default_ca_reactions() -> Vec<ReactionEntry> {
    vec![ReactionEntry {
        input_a: 3,
        input_b: 4,    // water + lava
        output_a: 1,
        output_b: 5,   // stone + steam
        heat_delta: -20,
        condition: 0,
        probability: 255,
        impulse: 0,
    }]
}

// ---------------------------------------------------------------------------
// Material bridge: CA materials -> render MaterialParams
// ---------------------------------------------------------------------------

/// Convert CA material properties to render-compatible [`MaterialParams`].
///
/// The render shader reads `MaterialParams` (4 x Vec4 = 64 bytes each).
/// This creates a table with appropriate colors for each CA material.
pub fn ca_materials_to_render_params(
    ca_mats: &[MaterialPropertiesCA],
) -> Vec<shared::material::MaterialParams> {
    use shared::material::MaterialParams;

    // Colors indexed by RENDER material ID (after remap in ca_to_render shader).
    // Remap: CA 4 (lava) → render 2, CA 2 (sand) → render 4.
    // So: render 0=air, 1=stone, 2=LAVA, 3=water, 4=SAND, 5=steam, 6=ice
    // 7-9 = stone variants (different shades for visual texture)
    let colors: &[(f32, f32, f32)] = &[
        (0.0, 0.0, 0.0),       // render 0 = Air
        (0.50, 0.48, 0.45),    // render 1 = Stone (warm gray)
        (1.0, 0.3, 0.1),       // render 2 = Lava (orange-red)
        (0.2, 0.4, 0.8),       // render 3 = Water (blue)
        (0.76, 0.70, 0.50),    // render 4 = Sand (tan)
        (0.9, 0.9, 0.95),      // render 5 = Steam
        (0.7, 0.85, 1.0),      // render 6 = Ice
        (0.42, 0.40, 0.38),    // render 7 = Dark stone
        (0.55, 0.50, 0.45),    // render 8 = Light stone
        (0.45, 0.47, 0.42),    // render 9 = Mossy stone (green tint)
    ];

    let mut result = Vec::with_capacity(ca_mats.len());
    for (i, ca) in ca_mats.iter().enumerate() {
        let (r, g, b) = if i < colors.len() {
            colors[i]
        } else {
            (0.6, 0.6, 0.6) // default gray for unknown materials
        };

        // Determine emissive temperature. Render index 2 = lava (after remap).
        let emissive_temp = if i == 2 { 100.0 } else { 10000.0 };

        // Opacity: air=0, everything else=1
        let opacity = if i == 0 { 0.0 } else { 1.0 };

        result.push(MaterialParams {
            elastic: glam::Vec4::new(1000.0, 0.3, 100.0, ca.viscosity as f32),
            thermal: glam::Vec4::new(
                ca.melt_temp as f32 * 10.0,
                ca.boil_temp as f32 * 10.0,
                ca.conductivity as f32,
                1.0,
            ),
            visual: glam::Vec4::new(ca.density as f32, emissive_temp, opacity, 0.0),
            color: glam::Vec4::new(r, g, b, 1.0),
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Test scene generator
// ---------------------------------------------------------------------------

/// Simple hash-based noise for terrain generation.
fn noise2d(x: usize, z: usize) -> f32 {
    let n = (x as u32).wrapping_mul(374761393)
        .wrapping_add((z as u32).wrapping_mul(668265263));
    let n = n ^ (n >> 13);
    let n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493)).wrapping_add(19990303));
    let n = n ^ (n >> 16);
    (n as f32) / (u32::MAX as f32)
}

/// Smoothed noise with interpolation for terrain.
fn smooth_noise(x: f32, z: f32) -> f32 {
    let ix = x.floor() as usize;
    let iz = z.floor() as usize;
    let fx = x - x.floor();
    let fz = z - z.floor();

    // Smoothstep
    let fx = fx * fx * (3.0 - 2.0 * fx);
    let fz = fz * fz * (3.0 - 2.0 * fz);

    let n00 = noise2d(ix, iz);
    let n10 = noise2d(ix + 1, iz);
    let n01 = noise2d(ix, iz + 1);
    let n11 = noise2d(ix + 1, iz + 1);

    let nx0 = n00 + (n10 - n00) * fx;
    let nx1 = n01 + (n11 - n01) * fx;
    nx0 + (nx1 - nx0) * fz
}

/// Multi-octave terrain height at world (x, z).
fn terrain_height(wx: usize, wz: usize) -> usize {
    let x = wx as f32;
    let z = wz as f32;

    // Base terrain: valleys and hills with wide height variation
    let mut h = 0.0f32;
    h += smooth_noise(x * 0.015, z * 0.015) * 30.0;  // large rolling terrain
    h += smooth_noise(x * 0.04, z * 0.04) * 12.0;    // medium hills
    h += smooth_noise(x * 0.1, z * 0.1) * 4.0;       // small bumps

    // Add a volcano/mountain in the center
    let cx = 64.0f32;
    let cz = 64.0f32;
    let dist_to_center = ((x - cx) * (x - cx) + (z - cz) * (z - cz)).sqrt();
    if dist_to_center < 25.0 {
        let volcano = (1.0 - dist_to_center / 25.0) * 45.0;
        h += volcano;
    }
    // Volcano crater at the very top
    if dist_to_center < 7.0 {
        h -= (1.0 - dist_to_center / 7.0) * 12.0;
    }

    // Lower base height so water_level creates visible lakes
    let base_height = 10;
    (h as usize).saturating_add(base_height).min(120)
}

/// Quick hash for surface roughness (deterministic, no allocation).
fn surface_hash(x: usize, y: usize, z: usize) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393)
        ^ (y as u32).wrapping_mul(668265263)
        ^ (z as u32).wrapping_mul(1440670441);
    h ^= h >> 13;
    h = h.wrapping_mul(1103515245).wrapping_add(12345);
    h ^= h >> 16;
    h
}

/// Pick a stone material variant (1, 7, 8, or 9) based on position.
/// Creates visible patches of different stone shades across the terrain.
fn pick_stone_variant(wx: usize, wy: usize, wz: usize) -> u16 {
    let _h = surface_hash(wx, wy, wz);
    // Use large-scale noise for patches (not per-voxel)
    let patch = surface_hash(wx / 4, wy / 4, wz / 4) % 4;
    match patch {
        0 => 1,  // normal stone
        1 => 7,  // dark stone
        2 => 8,  // light stone
        _ => 9,  // mossy stone
    }
}

/// Generate a voxel at world position (wx, wy, wz) given terrain height.
fn generate_voxel(wx: usize, wy: usize, wz: usize, height: usize) -> Voxel {
    let water_level = 35;

    let cx = 64.0f32;
    let cz = 64.0f32;
    let dist_to_center = (((wx as f32) - cx).powi(2) + ((wz as f32) - cz).powi(2)).sqrt();

    // Surface roughness: randomly remove 1-2 voxels at the surface to
    // create crevices that AO can shade, giving stone visible texture.
    let roughness = (surface_hash(wx, 0, wz) % 3) as usize; // 0, 1, or 2
    let effective_height = if height > roughness {
        height - roughness
    } else {
        height
    };

    if wy > effective_height && wy > water_level {
        return Voxel::air();
    }

    // Lava in volcano crater
    if dist_to_center < 6.0 && wy > height.saturating_sub(3) && wy <= height {
        return Voxel::new(4, 250, 0, 0, 0);
    }

    // Water fills below water level
    if wy > effective_height && wy <= water_level {
        return Voxel::new(3, 128, 0, 10, 0);
    }

    // Surface layer: top 3 voxels use varied materials for visual interest
    let depth_from_surface = effective_height.saturating_sub(wy);

    if wy <= effective_height && depth_from_surface == 0 {
        // Top surface
        if wy < water_level + 2 && wy >= water_level.saturating_sub(1) {
            return Voxel::new(2, 128, 0, 15, 0); // Sand near water
        }
        // Pick stone variant based on position hash for visible texture
        let stone_id = pick_stone_variant(wx, wy, wz);
        return Voxel::new(stone_id, 128, 0b111111, 15, 0);
    }

    // Below surface: also use stone variants for exposed cliff faces
    if wy < effective_height {
        let stone_id = pick_stone_variant(wx, wy, wz);
        return Voxel::new(stone_id, 128, 0b111111, 15, 0);
    }

    Voxel::air()
}

/// Generate a terrain-based test scene filling the full 128x128x128 render grid.
///
/// Features: rolling hills, a volcano with lava crater in the center,
/// water filling low areas, sand beaches near the waterline.
/// Returns chunks as `(coord, voxel_data)` pairs for a 4x4x4 grid
/// (128x128x128 voxels).
pub fn generate_test_scene() -> Vec<([i32; 3], Vec<u32>)> {
    let mut chunks = Vec::new();

    // Generate 4x4x4 = 64 chunks covering 128x128x128 voxels
    for cx in 0..4i32 {
        for cy in 0..4i32 {
            for cz in 0..4i32 {
                let mut data = vec![0u32; CA_CHUNK_VOXELS];

                for lz in 0..32usize {
                    for lx in 0..32usize {
                        let wx = cx as usize * 32 + lx;
                        let wz = cz as usize * 32 + lz;

                        let height = terrain_height(wx, wz);

                        for ly in 0..32usize {
                            let wy = cy as usize * 32 + ly;
                            let idx = lz * 32 * 32 + ly * 32 + lx;

                            let voxel = generate_voxel(wx, wy, wz, height);
                            data[idx] = voxel.0;
                        }
                    }
                }

                chunks.push(([cx, cy, cz], data));
            }
        }
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::VulkanContext;

    fn init_tracing() {
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    }

    #[test]
    fn push_constant_sizes() {
        assert_eq!(mem::size_of::<CaCompactPush>(), 16);
        assert_eq!(mem::size_of::<CaThermalPush>(), 16);
        assert_eq!(mem::size_of::<CaMargolusPush>(), 32);
        assert_eq!(mem::size_of::<CaToRenderPush>(), 32);
    }

    #[test]
    fn test_default_materials() {
        let mats = default_ca_materials();
        assert!(mats.len() >= 7);
        // Stone
        assert_eq!(mats[1].phase, 0);
        assert_eq!(mats[1].density, 200);
        // Water
        assert_eq!(mats[3].phase, 2);
        assert_eq!(mats[3].boil_temp, 200);
    }

    #[test]
    fn test_ca_materials_to_render_params() {
        let ca_mats = default_ca_materials();
        let render_mats = ca_materials_to_render_params(&ca_mats);
        assert_eq!(render_mats.len(), ca_mats.len());
        // Air should have zero opacity
        assert_eq!(render_mats[0].visual.z, 0.0);
        // Stone should have full opacity and gray color
        assert_eq!(render_mats[1].visual.z, 1.0);
        assert!(render_mats[1].color.x > 0.4 && render_mats[1].color.x < 0.6);
        // Water should be blue
        assert!(render_mats[3].color.z > 0.7);
    }

    #[test]
    fn test_default_reactions() {
        let rxns = default_ca_reactions();
        assert_eq!(rxns.len(), 1);
        assert_eq!(rxns[0].input_a, 3); // water
        assert_eq!(rxns[0].input_b, 4); // lava
    }

    #[test]
    fn test_generate_test_scene() {
        let scene = generate_test_scene();
        // 4x4x4 = 64 chunks
        assert_eq!(scene.len(), 64);
        // Each chunk has CA_CHUNK_VOXELS voxels
        for (_, data) in &scene {
            assert_eq!(data.len(), CA_CHUNK_VOXELS);
        }
        // First chunk (0,0,0) should have stone underground
        let (_, data) = &scene[0]; // (0,0,0)
        // Voxel at (0, 0, 0) -> index 0*32*32 + 0*32 + 0 = 0
        let v = Voxel(data[0]);
        // Stone variants: 1=stone, 7=dark stone, 8=light stone, 9=mossy stone
        assert!(
            matches!(v.material_id(), 1 | 7 | 8 | 9),
            "underground voxel should be a stone variant, got {}",
            v.material_id()
        );
    }

    // --- GPU tests (require VulkanContext, run with --test-threads=1) ---

    #[test]
    fn test_create_and_destroy() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("Failed to create CaSimulation");
        assert_eq!(sim.loaded_count(), 0);
        sim.destroy(&ctx);
    }

    #[test]
    fn test_load_and_download_chunk() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("Failed to create CaSimulation");

        let data: Vec<u32> = vec![Voxel::new(1, 128, 0, 0, 0).0; CA_CHUNK_VOXELS];
        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6])
            .expect("Failed to load chunk");
        assert_eq!(sim.loaded_count(), 1);

        let downloaded = sim
            .download_chunk(&ctx, [0, 0, 0])
            .expect("Failed to download chunk");
        assert_eq!(downloaded, data);

        sim.destroy(&ctx);
    }

    #[test]
    fn test_step_no_crash() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("Failed to create CaSimulation");

        // Load a simple chunk with some non-air voxels
        let mut data = vec![0u32; CA_CHUNK_VOXELS];
        data[0] = Voxel::new(1, 200, 0, 0, 0).0; // hot stone
        data[1] = Voxel::new(3, 100, 0, 0, 0).0; // water
        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6])
            .expect("Failed to load chunk");

        // Execute one step via a one-shot command buffer
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd, &ctx);
        })
        .expect("Simulation step failed");

        // Download and verify no crash
        let result = sim
            .download_chunk(&ctx, [0, 0, 0])
            .expect("Failed to download after step");
        assert_eq!(result.len(), CA_CHUNK_VOXELS);

        sim.destroy(&ctx);
    }

    #[test]
    fn test_unload_chunk() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("Failed to create CaSimulation");

        let data: Vec<u32> = vec![Voxel::new(2, 128, 0, 0, 0).0; CA_CHUNK_VOXELS];
        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6])
            .expect("load");
        assert_eq!(sim.loaded_count(), 1);

        let downloaded = sim
            .unload_chunk(&ctx, [0, 0, 0], true)
            .expect("unload")
            .expect("should have data");
        assert_eq!(downloaded, data);
        assert_eq!(sim.loaded_count(), 0);

        sim.destroy(&ctx);
    }

    #[test]
    fn test_ca_metadata_dirty_flag() {
        init_tracing();
        let ctx = VulkanContext::new().expect("VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("CaSimulation");

        // Load a chunk with some non-air voxels (triggers dirty flag)
        let mut data = vec![0u32; CA_CHUNK_VOXELS];
        data[0] = Voxel::new(1, 128, 0, 0, 0).0; // stone
        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6]).expect("load");

        // Readback metadata for slot 0: flags at offset 11 should have bit 0 set
        let meta = sim.debug_readback_metadata(&ctx, 0).expect("readback meta");
        println!("Metadata for slot 0: {:?}", meta);
        println!("  world_pos: [{}, {}, {}, {}]", meta[0], meta[1], meta[2], meta[3]);
        println!("  neighbor_ids: [{}, {}, {}, {}, {}, {}]", meta[4], meta[5], meta[6], meta[7], meta[8], meta[9]);
        println!("  activity: {}", meta[10]);
        println!("  flags: {} (dirty={})", meta[11], meta[11] & 1);
        assert_eq!(meta[11] & 1, 1, "dirty flag should be set after load_chunk");

        sim.destroy(&ctx);
    }

    #[test]
    fn test_ca_compact_finds_dirty_chunks() {
        init_tracing();
        let ctx = VulkanContext::new().expect("VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("CaSimulation");

        // Load a chunk (should be dirty)
        let mut data = vec![0u32; CA_CHUNK_VOXELS];
        data[0] = Voxel::new(1, 128, 0, 0, 0).0;
        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6]).expect("load");

        // Run just the compact pass (step does compact + thermal + margolus)
        // We run a full step to exercise the whole pipeline, then readback dirty list
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd, &ctx);
        })
        .expect("step");

        // Readback dirty list: [dirty_count, dispatch_x, dispatch_y, dispatch_z, slot_id_0, ...]
        let dirty = sim.debug_readback_dirty_list(&ctx, 8).expect("readback dirty");
        println!("Dirty list after step: {:?}", dirty);
        println!("  dirty_count: {}", dirty[0]);
        println!("  dispatch_x: {}", dirty[1]);
        println!("  dispatch_y: {}", dirty[2]);
        println!("  dispatch_z: {}", dirty[3]);
        if dirty[0] > 0 {
            println!("  first dirty slot_id: {}", dirty[4]);
        }

        assert!(
            dirty[0] > 0,
            "compact pass should find at least 1 dirty chunk"
        );
        assert_eq!(dirty[4], 0, "first dirty slot should be slot 0");

        sim.destroy(&ctx);
    }

    #[test]
    fn test_ca_physics_water_falls() {
        init_tracing();
        let ctx = VulkanContext::new().expect("VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("CaSimulation");

        // Create chunk: stone floor at y=0..3, water at y=5, air at y=4
        let mut data = vec![0u32; CA_CHUNK_VOXELS];

        // Stone floor (y=0..3)
        for z in 0..32u32 {
            for x in 0..32u32 {
                for y in 0..4u32 {
                    let idx = (z * 32 * 32 + y * 32 + x) as usize;
                    data[idx] = Voxel::new(1, 128, 0, 0, 0).0; // stone
                }
            }
        }

        // Water at y=5 (center of chunk)
        for z in 10..20u32 {
            for x in 10..20u32 {
                let idx = (z * 32 * 32 + 5 * 32 + x) as usize;
                data[idx] = Voxel::new(3, 128, 0, 0, 0).0; // water
            }
        }

        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6]).expect("load");

        // Helper to count water at a given y-level in the 10..20 xz region
        fn count_water_at_y(data: &[u32], y: u32) -> usize {
            let mut count = 0;
            for z in 10..20u32 {
                for x in 10..20u32 {
                    let idx = (z * 32 * 32 + y * 32 + x) as usize;
                    if Voxel(data[idx]).material_id() == 3 {
                        count += 1;
                    }
                }
            }
            count
        }

        // Count water at y=4 and y=5 before step
        let pre = sim.download_chunk(&ctx, [0, 0, 0]).expect("download pre");
        let pre_y5 = count_water_at_y(&pre, 5);
        let pre_y4 = count_water_at_y(&pre, 4);
        println!("BEFORE step: water at y=5: {}, water at y=4: {}", pre_y5, pre_y4);

        // Run physics step
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd, &ctx);
        })
        .expect("step");

        // Download and check
        let result = sim.download_chunk(&ctx, [0, 0, 0]).expect("download post");
        let post_y5 = count_water_at_y(&result, 5);
        let post_y4 = count_water_at_y(&result, 4);
        println!("AFTER step: water at y=5: {}, water at y=4: {}", post_y5, post_y4);

        // Also check dirty list to verify compact pass worked
        let dirty = sim.debug_readback_dirty_list(&ctx, 8).expect("dirty list");
        println!("Dirty list: count={}, dispatch_x={}, slot={}", dirty[0], dirty[1], dirty[4]);

        assert!(
            post_y4 > 0,
            "Water should have fallen from y=5 to y=4 after one step. \
             Before: y5={} y4={}, After: y5={} y4={}",
            pre_y5, pre_y4, post_y5, post_y4,
        );

        sim.destroy(&ctx);
    }

    #[test]
    fn test_ca_physics_multi_step() {
        init_tracing();
        let ctx = VulkanContext::new().expect("VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("CaSimulation");

        // Create chunk: stone at y=0..3, air at y=4..6, water at y=10
        let mut data = vec![0u32; CA_CHUNK_VOXELS];
        for z in 0..32u32 {
            for x in 0..32u32 {
                for y in 0..4u32 {
                    let idx = (z * 32 * 32 + y * 32 + x) as usize;
                    data[idx] = Voxel::new(1, 128, 0, 0, 0).0;
                }
            }
        }
        // Water at y=10
        for z in 10..20u32 {
            for x in 10..20u32 {
                let idx = (z * 32 * 32 + 10 * 32 + x) as usize;
                data[idx] = Voxel::new(3, 128, 0, 0, 0).0;
            }
        }

        sim.load_chunk(&ctx, [0, 0, 0], &data, [-1; 6]).expect("load");

        // Run 5 steps
        for step_i in 0..5u32 {
            ctx.execute_one_shot(|cmd| {
                sim.step(cmd, &ctx);
            })
            .expect("step");

            let result = sim.download_chunk(&ctx, [0, 0, 0]).expect("download");
            let dirty = sim.debug_readback_dirty_list(&ctx, 8).expect("dirty");

            // Count total water voxels in the whole chunk (water spreads horizontally)
            let mut total_water = 0u32;
            let mut min_water_y = 32u32;
            for y in 0..32u32 {
                for z in 0..32u32 {
                    for x in 0..32u32 {
                        let idx = (z * 32 * 32 + y * 32 + x) as usize;
                        if Voxel(result[idx]).material_id() == 3 {
                            total_water += 1;
                            if y < min_water_y {
                                min_water_y = y;
                            }
                        }
                    }
                }
            }
            println!(
                "Step {}: min_water_y={}, total_water={}, dirty_count={}",
                step_i, min_water_y, total_water, dirty[0]
            );
        }

        // After 5 steps, water should have fallen AND spread
        let result = sim.download_chunk(&ctx, [0, 0, 0]).expect("download final");
        let mut water_at_y4 = 0;
        for z in 0..32u32 {
            for x in 0..32u32 {
                let idx = (z * 32 * 32 + 4 * 32 + x) as usize;
                if Voxel(result[idx]).material_id() == 3 {
                    water_at_y4 += 1;
                }
            }
        }
        println!("Water at y=4 (entire xz plane) after 5 steps: {}", water_at_y4);
        assert!(
            water_at_y4 > 0,
            "Water should reach y=4 (stone top) after falling from y=10"
        );

        sim.destroy(&ctx);
    }

    #[test]
    fn test_ca_physics_test_scene() {
        init_tracing();
        let ctx = VulkanContext::new().expect("VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("CaSimulation");

        // Load the actual test scene (same as used by --sim2 flag)
        let scene = generate_test_scene();
        for (coord, voxel_data) in &scene {
            sim.load_chunk(&ctx, *coord, voxel_data, [-1; 6]).expect("load");
        }
        println!("Loaded {} chunks", sim.loaded_count());

        // Snapshot: count all voxels by material across all chunks
        let coords: Vec<[i32; 3]> = scene.iter().map(|(c, _)| *c).collect();

        let count_materials = |sim: &CaSimulation| -> [usize; 10] {
            let mut counts = [0usize; 10];
            for coord in &coords {
                let data = sim.download_chunk(&ctx, *coord).unwrap();
                for &v in &data {
                    let mat = Voxel(v).material_id() as usize;
                    if mat < 10 {
                        counts[mat] += 1;
                    }
                }
            }
            counts
        };

        let before = count_materials(&sim);
        println!(
            "BEFORE: air={}, stone={}, sand={}, water={}, lava={}, steam={}, ice={}",
            before[0], before[1], before[2], before[3], before[4], before[5], before[6]
        );

        // Run 10 steps
        for _ in 0..10 {
            ctx.execute_one_shot(|cmd| {
                sim.step(cmd, &ctx);
            })
            .expect("step");
        }

        let after = count_materials(&sim);
        println!(
            "AFTER:  air={}, stone={}, sand={}, water={}, lava={}, steam={}, ice={}",
            after[0], after[1], after[2], after[3], after[4], after[5], after[6]
        );

        // Check if ANYTHING changed
        let any_changed = (0..10).any(|i| before[i] != after[i]);
        let total_changed_voxels: usize = {
            let mut changed = 0usize;
            for i in 0..10 {
                if before[i] != after[i] {
                    changed += (before[i] as i64 - after[i] as i64).unsigned_abs() as usize;
                }
            }
            changed
        };

        println!("Any material counts changed: {}", any_changed);
        println!("Total material count delta: {}", total_changed_voxels);

        // The scene must NOT be static: thermal diffusion should cause lava→stone transitions
        assert!(
            any_changed,
            "Test scene should not be static: thermal diffusion + spreading should cause changes"
        );

        sim.destroy(&ctx);
    }

    #[test]
    fn test_step_empty_no_crash() {
        init_tracing();
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mats = default_ca_materials();
        let reactions = default_ca_reactions();
        let mut sim =
            CaSimulation::new(&ctx, &mats, &reactions, 64).expect("Failed to create CaSimulation");

        // Step with no loaded chunks should be a no-op
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd, &ctx);
        })
        .expect("Empty step failed");

        assert_eq!(sim.frame_number(), 1);
        sim.destroy(&ctx);
    }
}
