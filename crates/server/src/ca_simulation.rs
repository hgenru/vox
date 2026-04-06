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
        // compact: 2, compact_finalize: 1, thermal: 4, margolus: 5 = 12 total
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16, // some headroom
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 4, "ca-descriptor-pool")?;

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

        self.frame_number += 1;
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
        ] {
            pipeline::destroy_pipeline(ctx, pass.pipeline);
            pipeline::destroy_pipeline_layout(ctx, pass.pipeline_layout);
            pipeline::destroy_descriptor_set_layout(ctx, pass.descriptor_set_layout);
        }

        pipeline::destroy_descriptor_pool(ctx, self.descriptor_pool);
        pipeline::destroy_shader_module(ctx, self.shader_module);

        buffer::destroy_buffer(ctx, self.material_buffer);
        buffer::destroy_buffer(ctx, self.reaction_buffer);
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
// Test scene generator
// ---------------------------------------------------------------------------

/// Generate a simple test scene: stone floor + sand + water pool + lava.
///
/// Returns chunks as `(coord, voxel_data)` pairs for a 4x2x4 grid
/// (128x64x128 voxels).
pub fn generate_test_scene() -> Vec<([i32; 3], Vec<u32>)> {
    let mut chunks = Vec::new();

    for cx in 0..4i32 {
        for cy in 0..2i32 {
            for cz in 0..4i32 {
                let mut data = vec![0u32; CA_CHUNK_VOXELS];
                let base_y = cy as usize * 32;

                for lz in 0..32usize {
                    for ly in 0..32usize {
                        for lx in 0..32usize {
                            let wy = base_y + ly;
                            let idx = lz * 32 * 32 + ly * 32 + lx;

                            let voxel = if wy < 4 {
                                // Stone floor
                                Voxel::new(1, 128, 0b111111, 15, 0)
                            } else if wy < 8 && cx == 1 && cz == 1 {
                                // Sand pile in center
                                Voxel::new(2, 128, 0, 15, 0)
                            } else if wy >= 4 && wy < 10 && cx == 2 && cz == 2 {
                                // Water pool
                                Voxel::new(3, 128, 0, 10, 0)
                            } else if wy == 8
                                && cx == 0
                                && cz == 0
                                && lx < 4
                                && lz < 4
                            {
                                // Small lava source
                                Voxel::new(4, 250, 0, 0, 0)
                            } else {
                                Voxel::air()
                            };

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
    fn test_default_reactions() {
        let rxns = default_ca_reactions();
        assert_eq!(rxns.len(), 1);
        assert_eq!(rxns[0].input_a, 3); // water
        assert_eq!(rxns[0].input_b, 4); // lava
    }

    #[test]
    fn test_generate_test_scene() {
        let scene = generate_test_scene();
        // 4x2x4 = 32 chunks
        assert_eq!(scene.len(), 32);
        // Each chunk has CA_CHUNK_VOXELS voxels
        for (_, data) in &scene {
            assert_eq!(data.len(), CA_CHUNK_VOXELS);
        }
        // First chunk (0,0,0) should have stone floor at y<4
        let (_, data) = &scene[0]; // (0,0,0)
        // Voxel at (0, 0, 0) -> index 0*32*32 + 0*32 + 0 = 0
        let v = Voxel(data[0]);
        assert_eq!(v.material_id(), 1); // stone
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
