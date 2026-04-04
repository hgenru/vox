//! [`GpuSimulation`] implementation — construction, stepping, rendering, destruction.

use std::mem;
use std::time::Instant;

use bytemuck::Zeroable;

use ash::vk;
use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
    pipeline,
};
use shared::{
    ChunkTableEntry, DIRTY_TILE_COUNT, FAR_FIELD_VOXEL_BUFFER_SIZE, GRID_CELL_COUNT,
    GRID_SIZE, GridCell, HASH_GRID_DEFAULT_CAPACITY, MAX_ACTIVE_CELLS, MAX_FAR_CHUNKS,
    MAX_PARTICLES, Particle, RENDER_HEIGHT, RENDER_WIDTH, SLEEP_THRESHOLD, TOTAL_BRICKS,
    TOTAL_SUPER_BRICKS,
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

    // Activity tracking
    pub(crate) activity_map_buffer: GpuBuffer,

    // Brick sleep
    sleep_counter_buffer: GpuBuffer,
    sleep_state_buffer: GpuBuffer,

    // Brick occupancy (render optimization)
    brick_occupancy_buffer: GpuBuffer,
    super_brick_occupancy_buffer: GpuBuffer,

    // Far-field rendering buffers
    far_field_voxel_buffer: GpuBuffer,
    far_field_chunk_table: GpuBuffer,

    // Any-active flag (1 u32, set by update_sleep if any brick is active)
    pub(crate) any_active_buffer: GpuBuffer,

    // Out-of-bounds flag (1 u32, set by G2P if any particle escapes domain)
    pub(crate) oob_flag_buffer: GpuBuffer,

    // Counting sort buffers (particle-by-brick sorting)
    brick_count_buffer: GpuBuffer,
    brick_offset_buffer: GpuBuffer,
    write_offset_buffer: GpuBuffer,
    sorted_particle_buffer: GpuBuffer,
    active_brick_list_buffer: GpuBuffer,
    active_brick_count_buffer: GpuBuffer,

    // Dirty-tile rendering
    prev_render_output_buffer: GpuBuffer,
    dirty_tile_buffer: GpuBuffer,

    // Sparse dispatch buffers
    mark_buffer: GpuBuffer,
    active_cells_buffer: GpuBuffer,
    active_count_buffer: GpuBuffer,
    indirect_dispatch_buffer: GpuBuffer,

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

    // Activity tracking pass
    compute_activity_pass: ComputePass,

    // Brick occupancy pass (render optimization)
    compute_occupancy_pass: ComputePass,

    // Far-field render pass
    render_far_field_pass: ComputePass,

    // Dirty-tile pass
    compute_dirty_tiles_pass: ComputePass,

    // Brick sleep pass
    update_sleep_pass: ComputePass,

    // Counting sort passes
    count_per_brick_pass: ComputePass,
    prefix_sum_pass: ComputePass,
    scatter_particles_pass: ComputePass,
    compact_active_bricks_pass: ComputePass,

    // Sparse compute passes
    mark_active_pass: ComputePass,
    prepare_indirect_pass: ComputePass,
    clear_grid_sparse_pass: ComputePass,
    grid_update_sparse_pass: ComputePass,

    // Hash grid buffers (sparse grid alternative)
    hash_grid_keys_buffer: GpuBuffer,
    hash_grid_values_buffer: GpuBuffer,

    // Hash grid compute passes
    clear_hash_grid_pass: ComputePass,
    p2g_hash_pass: ComputePass,
    g2p_hash_pass: ComputePass,
    grid_update_hash_pass: ComputePass,

    /// When true, use sparse hash grid P2G/G2P instead of dense grid.
    pub use_sparse_grid: bool,

    // Shader module (kept alive for pipeline lifetime)
    shader_module: vk::ShaderModule,

    // Descriptor pool (owns all descriptor sets)
    descriptor_pool: vk::DescriptorPool,

    // State
    pub(crate) num_particles: u32,
    num_phase_rules: u32,
    /// Number of far-field chunks currently uploaded to the GPU.
    far_field_chunk_count: u32,
    /// World-space origin of the simulation domain (grid [0,0,0] in world coords).
    sim_origin: [f32; 3],
    /// Monotonically increasing frame counter for graduated sleep scheduling.
    frame_number: core::cell::Cell<u32>,
    /// Previous frame camera eye (for dirty-tile camera-moved detection).
    prev_eye: core::cell::Cell<[f32; 3]>,
    /// Previous frame camera target (for dirty-tile camera-moved detection).
    prev_target: core::cell::Cell<[f32; 3]>,

    // Multi-rate scheduling timestamps
    /// Last time `compute_activity` + `update_sleep` were dispatched.
    last_sleep_update_time: core::cell::Cell<Instant>,
    /// Last time `compute_occupancy` was dispatched.
    last_occupancy_time: core::cell::Cell<Instant>,
    /// Last time `sort_particles` was dispatched.
    last_sort_time: core::cell::Cell<Instant>,

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

        // Grid buffer: used by the dense physics path.
        // When sparse hash grid is active, we allocate a small placeholder
        // since the dense passes won't run and this saves ~768MB at 256³.
        // GridCell = 48 bytes = 12 * f32.
        let use_sparse = true; // matches use_sparse_grid default
        let grid_buf_cells = if use_sparse {
            // Small placeholder — dense passes (clear_grid_sparse, mark_active,
            // prepare_indirect, grid_update_sparse, P2G, G2P) won't run.
            64 * 64 * 64_usize
        } else {
            GRID_CELL_COUNT as usize
        };
        let grid_buf_size =
            (grid_buf_cells * mem::size_of::<GridCell>()) as vk::DeviceSize;
        let grid_buffer = buffer::create_device_local_buffer(
            ctx,
            grid_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "grid-buffer",
        )?;
        tracing::info!(
            "Grid buffer: {} cells ({:.1}MB) [sparse={}]",
            grid_buf_cells,
            grid_buf_size as f64 / (1024.0 * 1024.0),
            use_sparse,
        );

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

        // -- Activity map buffer (1 u32 per brick, 128 KB) --
        let activity_map_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "activity-map-buffer",
        )?;

        // -- Brick sleep buffers (1 u32 per brick each, 128 KB) --
        // All bricks start awake (zeroed). Without explicit init, P2G would read
        // garbage from sleep_state and skip particles on the first frame.
        let sleep_counter_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "sleep-counter-buffer",
        )?;
        let sleep_state_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "sleep-state-buffer",
        )?;
        let zeros = vec![0u32; TOTAL_BRICKS as usize];
        buffer::upload(ctx, &zeros, &sleep_counter_buffer)?;
        buffer::upload(ctx, &zeros, &sleep_state_buffer)?;

        // -- Brick occupancy buffer (1 u32 per brick, 128 KB) --
        let brick_occupancy_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "brick-occupancy-buffer",
        )?;

        // -- Dirty-tile rendering buffers --
        // Previous frame render output (same size as render_output_buffer)
        let prev_render_output_buffer = buffer::create_device_local_buffer(
            ctx,
            render_output_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "prev-render-output-buffer",
        )?;

        // Dirty tile buffer: 1 u32 per tile
        let dirty_tile_buffer = buffer::create_device_local_buffer(
            ctx,
            (DIRTY_TILE_COUNT as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "dirty-tile-buffer",
        )?;

        // -- Super-brick occupancy buffer (1 u32 per super-brick) --
        // For 256³ grid: 4³ = 64 super-bricks, 256 bytes
        let super_brick_occupancy_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_SUPER_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "super-brick-occupancy-buffer",
        )?;

        // -- Far-field rendering buffers --
        // Voxel buffer: u8 per voxel packed as u32s → total bytes / 4
        let far_field_voxel_buf_size =
            (FAR_FIELD_VOXEL_BUFFER_SIZE as usize) as vk::DeviceSize;
        let far_field_voxel_buffer = buffer::create_device_local_buffer(
            ctx,
            far_field_voxel_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "far-field-voxel-buffer",
        )?;
        tracing::info!(
            "Far-field voxel buffer: {:.1}MB ({} chunks * {}³ voxels)",
            far_field_voxel_buf_size as f64 / (1024.0 * 1024.0),
            MAX_FAR_CHUNKS,
            64,
        );

        // Chunk table: ChunkTableEntry (32 bytes) * MAX_FAR_CHUNKS
        let far_field_table_size =
            (MAX_FAR_CHUNKS as usize * mem::size_of::<ChunkTableEntry>()) as vk::DeviceSize;
        let far_field_chunk_table = buffer::create_device_local_buffer(
            ctx,
            far_field_table_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "far-field-chunk-table",
        )?;

        // -- Any-active flag buffer (1 u32, set by update_sleep via atomicMax) --
        let any_active_buffer = buffer::create_device_local_buffer(
            ctx,
            4 as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            "any-active-buffer",
        )?;

        // -- OOB flag buffer (1 u32, set by G2P if any particle escapes domain) --
        let oob_flag_buffer = buffer::create_device_local_buffer(
            ctx,
            16 as vk::DeviceSize, // min 16 for alignment
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            "oob-flag-buffer",
        )?;

        // -- Sparse dispatch buffers --
        // When hash grid is active, mark_buffer and active_cells are unused (dense path only).
        // Allocate small placeholders to save memory.
        let sparse_dispatch_cells = if use_sparse {
            64_usize // minimal placeholder
        } else {
            GRID_CELL_COUNT as usize
        };
        let sparse_active_cells = if use_sparse {
            64_usize
        } else {
            MAX_ACTIVE_CELLS as usize
        };
        let mark_buffer = buffer::create_device_local_buffer(
            ctx,
            (sparse_dispatch_cells * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "mark-buffer",
        )?;
        let active_cells_buffer = buffer::create_device_local_buffer(
            ctx,
            (sparse_active_cells * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "active-cells-buffer",
        )?;
        let active_count_buffer = buffer::create_device_local_buffer(
            ctx,
            16 as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "active-count-buffer",
        )?;
        let indirect_dispatch_buffer =
            buffer::create_indirect_buffer(ctx, "indirect-dispatch-buffer")?;

        // -- Hash grid buffers --
        let hash_capacity = HASH_GRID_DEFAULT_CAPACITY as usize;
        let hash_keys_buf_size = (hash_capacity * mem::size_of::<u32>()) as vk::DeviceSize;
        let hash_grid_keys_buffer = buffer::create_device_local_buffer(
            ctx,
            hash_keys_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "hash-grid-keys-buffer",
        )?;

        let hash_values_buf_size =
            (hash_capacity * mem::size_of::<GridCell>()) as vk::DeviceSize;
        let hash_grid_values_buffer = buffer::create_device_local_buffer(
            ctx,
            hash_values_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "hash-grid-values-buffer",
        )?;

        // Initialize hash grid keys to EMPTY sentinel
        let empty_keys = vec![shared::HASH_GRID_EMPTY_KEY; hash_capacity];
        buffer::upload(ctx, &empty_keys, &hash_grid_keys_buffer)?;
        tracing::info!(
            "Created hash grid buffers: keys={}MB, values={}MB (capacity={})",
            hash_keys_buf_size / (1024 * 1024),
            hash_values_buf_size / (1024 * 1024),
            hash_capacity
        );

        // -- Counting sort buffers --
        let brick_count_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "brick-count-buffer",
        )?;
        let brick_offset_buffer = buffer::create_device_local_buffer(
            ctx,
            ((TOTAL_BRICKS as usize + 1) * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "brick-offset-buffer",
        )?;
        let write_offset_buffer = buffer::create_device_local_buffer(
            ctx,
            ((TOTAL_BRICKS as usize + 1) * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "write-offset-buffer",
        )?;
        let sorted_particle_buffer = buffer::create_device_local_buffer(
            ctx,
            particle_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "sorted-particle-buffer",
        )?;
        let active_brick_list_buffer = buffer::create_device_local_buffer(
            ctx,
            (TOTAL_BRICKS as usize * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "active-brick-list-buffer",
        )?;
        let active_brick_count_buffer = buffer::create_device_local_buffer(
            ctx,
            16 as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "active-brick-count-buffer",
        )?;

        // -- Descriptor pool --
        // All passes combined: dirty tiles + hash grid + counting sort + super bricks + far-field
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 184,
        }];
        let descriptor_pool =
            pipeline::create_descriptor_pool(ctx, &pool_sizes, 31, "sim-descriptor-pool")?;

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
            &[&particle_buffer, &grid_buffer, &material_buffer, &sleep_state_buffer],
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
            &[&particle_buffer, &grid_buffer, &voxel_buffer, &reaction_buffer, &sleep_state_buffer, &oob_flag_buffer],
            mem::size_of::<G2pPushConstants>() as u32,
            descriptor_pool,
            "g2p",
        )?;

        let clear_voxels_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::clear_voxels",
            &[&voxel_buffer, &sleep_state_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "clear-voxels",
        )?;

        let voxelize_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::voxelize::voxelize",
            &[&particle_buffer, &voxel_buffer, &sleep_state_buffer],
            mem::size_of::<VoxelizePushConstants>() as u32,
            descriptor_pool,
            "voxelize",
        )?;

        let render_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::render::render_voxels",
            &[
                &voxel_buffer,
                &render_output_buffer,
                &material_buffer,
                &brick_occupancy_buffer,
                &super_brick_occupancy_buffer,
                &dirty_tile_buffer,
                &prev_render_output_buffer,
            ],
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

        // -- Activity tracking pass --
        // compute_activity: particles(r), activity_map(rw)
        let compute_activity_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::compute_activity::compute_activity",
            &[&particle_buffer, &activity_map_buffer],
            mem::size_of::<ComputeActivityPushConstants>() as u32,
            descriptor_pool,
            "compute-activity",
        )?;

        // -- Brick occupancy pass --
        // compute_occupancy: voxels(r), brick_occupied(w), super_brick_occupied(w)
        let compute_occupancy_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::compute_occupancy::compute_occupancy",
            &[&voxel_buffer, &brick_occupancy_buffer, &super_brick_occupancy_buffer],
            mem::size_of::<ComputeOccupancyPushConstants>() as u32,
            descriptor_pool,
            "compute-occupancy",
        )?;

        // -- Far-field render pass --
        // render_far_field: render_output(rw), far_field_voxels(r), far_field_table(r), materials(r)
        let render_far_field_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::render_far_field::render_far_field",
            &[
                &render_output_buffer,
                &far_field_voxel_buffer,
                &far_field_chunk_table,
                &material_buffer,
            ],
            mem::size_of::<FarFieldPushConstants>() as u32,
            descriptor_pool,
            "render-far-field",
        )?;

        // -- Dirty-tile pass --
        // compute_dirty_tiles: sleep_state(r), dirty_tiles(w)
        let compute_dirty_tiles_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::compute_dirty_tiles::compute_dirty_tiles",
            &[&sleep_state_buffer, &dirty_tile_buffer],
            mem::size_of::<DirtyTilesPushConstants>() as u32,
            descriptor_pool,
            "compute-dirty-tiles",
        )?;

        // -- Brick sleep pass --
        // update_sleep: activity_map(r), sleep_counter(rw), sleep_state(w), any_active(w)
        let update_sleep_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::update_sleep::update_sleep",
            &[&activity_map_buffer, &sleep_counter_buffer, &sleep_state_buffer, &any_active_buffer],
            mem::size_of::<UpdateSleepPushConstants>() as u32,
            descriptor_pool,
            "update-sleep",
        )?;

        // -- Sparse dispatch passes --
        // mark_active: particles(r), mark(rw), active_cells(w), active_count(rw)
        let mark_active_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::mark_active::mark_active",
            &[&particle_buffer, &mark_buffer, &active_cells_buffer, &active_count_buffer],
            mem::size_of::<MarkActivePushConstants>() as u32,
            descriptor_pool,
            "mark-active",
        )?;

        // prepare_indirect: active_count(r), indirect_args(w)
        let prepare_indirect_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::prepare_indirect::prepare_indirect",
            &[&active_count_buffer, &indirect_dispatch_buffer],
            0, // no push constants
            descriptor_pool,
            "prepare-indirect",
        )?;

        // clear_grid_sparse: active_cells(r), grid(w), mark(w), active_count(r)
        let clear_grid_sparse_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::clear_grid_sparse::clear_grid_sparse",
            &[&active_cells_buffer, &grid_buffer, &mark_buffer, &active_count_buffer],
            0, // no push constants — reads count from buffer
            descriptor_pool,
            "clear-grid-sparse",
        )?;

        // grid_update_sparse: active_cells(r), grid(rw), active_count(r)
        let grid_update_sparse_pass = passes::create_pass(
            ctx,
            shader_module,
            c"compute::grid_update_sparse::grid_update_sparse",
            &[&active_cells_buffer, &grid_buffer, &active_count_buffer],
            mem::size_of::<GridUpdateSparsePushConstants>() as u32,
            descriptor_pool,
            "grid-update-sparse",
        )?;

        // -- Hash grid compute passes --
        let clear_hash_grid_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::clear_hash_grid::clear_hash_grid",
            &[&hash_grid_keys_buffer, &hash_grid_values_buffer],
            mem::size_of::<ClearHashGridPushConstants>() as u32,
            descriptor_pool, "clear-hash-grid",
        )?;
        let p2g_hash_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::p2g_sparse::p2g_sparse",
            &[&particle_buffer, &hash_grid_keys_buffer, &hash_grid_values_buffer, &material_buffer, &sleep_state_buffer],
            mem::size_of::<P2gSparsePushConstants>() as u32,
            descriptor_pool, "p2g-hash",
        )?;
        let g2p_hash_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::g2p_sparse::g2p_sparse",
            &[&particle_buffer, &hash_grid_keys_buffer, &hash_grid_values_buffer, &voxel_buffer, &reaction_buffer, &sleep_state_buffer, &oob_flag_buffer],
            mem::size_of::<G2pSparsePushConstants>() as u32,
            descriptor_pool, "g2p-hash",
        )?;
        let grid_update_hash_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::grid_update_hash::grid_update_hash",
            &[&hash_grid_keys_buffer, &hash_grid_values_buffer],
            mem::size_of::<GridUpdateHashPushConstants>() as u32,
            descriptor_pool, "grid-update-hash",
        )?;

        // -- Counting sort passes --
        let count_per_brick_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::count_per_brick::count_per_brick",
            &[&particle_buffer, &brick_count_buffer],
            mem::size_of::<CountPerBrickPushConstants>() as u32,
            descriptor_pool, "count-per-brick",
        )?;
        let prefix_sum_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::prefix_sum::prefix_sum",
            &[&brick_count_buffer, &brick_offset_buffer],
            mem::size_of::<PrefixSumPushConstants>() as u32,
            descriptor_pool, "prefix-sum",
        )?;
        let scatter_particles_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::scatter_particles::scatter_particles",
            &[&particle_buffer, &write_offset_buffer, &sorted_particle_buffer],
            mem::size_of::<ScatterParticlesPushConstants>() as u32,
            descriptor_pool, "scatter-particles",
        )?;
        let compact_active_bricks_pass = passes::create_pass(
            ctx, shader_module,
            c"compute::compact_active_bricks::compact_active_bricks",
            &[&brick_count_buffer, &sleep_state_buffer, &active_brick_list_buffer, &active_brick_count_buffer],
            mem::size_of::<CompactActiveBricksPushConstants>() as u32,
            descriptor_pool, "compact-active-bricks",
        )?;

        tracing::info!("GpuSimulation created successfully");

        Ok(Self {
            particle_buffer,
            grid_buffer,
            voxel_buffer,
            render_output_buffer,
            material_buffer,
            reaction_buffer,
            activity_map_buffer,
            sleep_counter_buffer,
            sleep_state_buffer,
            brick_occupancy_buffer,
            super_brick_occupancy_buffer,
            far_field_voxel_buffer,
            far_field_chunk_table,
            any_active_buffer,
            oob_flag_buffer,
            brick_count_buffer,
            brick_offset_buffer,
            write_offset_buffer,
            sorted_particle_buffer,
            active_brick_list_buffer,
            active_brick_count_buffer,
            prev_render_output_buffer,
            dirty_tile_buffer,
            mark_buffer,
            active_cells_buffer,
            active_count_buffer,
            indirect_dispatch_buffer,
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
            compute_activity_pass,
            compute_occupancy_pass,
            render_far_field_pass,
            compute_dirty_tiles_pass,
            update_sleep_pass,
            count_per_brick_pass,
            prefix_sum_pass,
            scatter_particles_pass,
            compact_active_bricks_pass,
            mark_active_pass,
            prepare_indirect_pass,
            clear_grid_sparse_pass,
            grid_update_sparse_pass,
            hash_grid_keys_buffer,
            hash_grid_values_buffer,
            clear_hash_grid_pass,
            p2g_hash_pass,
            g2p_hash_pass,
            grid_update_hash_pass,
            use_sparse_grid: true,
            shader_module,
            descriptor_pool,
            num_particles: 0,
            num_phase_rules: num_phase_rules as u32,
            far_field_chunk_count: 0,
            sim_origin: [0.0; 3],
            frame_number: core::cell::Cell::new(0),
            prev_eye: core::cell::Cell::new([f32::NAN; 3]),
            prev_target: core::cell::Cell::new([f32::NAN; 3]),
            last_sleep_update_time: core::cell::Cell::new(Instant::now()),
            last_occupancy_time: core::cell::Cell::new(Instant::now()),
            last_sort_time: core::cell::Cell::new(Instant::now()),
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

    /// Upload a new set of particles, replacing the current set.
    ///
    /// Unlike [`init_particles`](Self::init_particles), this preserves sleep state
    /// for bricks that haven't changed occupancy. Sleep state will naturally
    /// adapt via the activity tracking system.
    pub fn reload_particles(&mut self, ctx: &VulkanContext, particles: &[Particle]) -> Result<()> {
        if particles.len() > MAX_PARTICLES as usize {
            return Err(ServerError::TooManyParticles(particles.len()));
        }
        self.num_particles = particles.len() as u32;
        buffer::upload(ctx, particles, &self.particle_buffer)?;
        tracing::info!("Reloaded {} particles to GPU", self.num_particles);
        Ok(())
    }

    /// Record the full simulation dispatch chain into the given command buffer.
    ///
    /// The command buffer must already be in the recording state.
    /// Runs ALL passes every substep (no multi-rate scheduling).
    /// For production use, prefer [`step_physics_with_time`] which skips
    /// infrequent passes based on real-time intervals.
    ///
    /// Dispatch order:
    /// 1. `clear_grid_sparse` (indirect) — clears previous frame's active cells + marks
    /// 2. `vkCmdFillBuffer` — reset active_count and mark_buffer to 0
    /// 3. `P2G` (particle dispatch) — scatter to grid
    /// 4. `mark_active` (particle dispatch) — build new active cell list
    /// 5. `prepare_indirect` (1,1,1) — write indirect args from new active_count
    /// 6. `grid_update_sparse` (indirect) — physics on active cells only
    /// 7. `G2P` (particle dispatch) — gather from grid
    /// 8. `compute_activity` + `update_sleep` — brick sleep scheduling
    /// 9. `clear_voxels` + `voxelize` — rasterize particles to voxel grid
    /// 10. `compute_occupancy` — render optimization
    ///
    /// On the first frame, the indirect buffer contains (0,0,0) so
    /// `clear_grid_sparse` is a no-op. After `mark_active` + `prepare_indirect`,
    /// the sparse dispatch is populated for `grid_update_sparse` and will persist
    /// for next frame's `clear_grid_sparse`.
    ///
    /// Memory barriers are inserted between each dispatch to ensure
    /// correct ordering of shader reads and writes.
    /// Run physics only (no reactions). Call in substep loop.
    pub fn step_physics(&self, cmd: vk::CommandBuffer) {
        self.record_core_physics(cmd);
        self.record_activity_and_sleep(cmd);
        self.record_voxelize(cmd);
        self.record_occupancy(cmd);
        self.frame_number.set(self.frame_number.get().wrapping_add(1));
    }

    /// Record the simulation dispatch chain with multi-rate scheduling.
    ///
    /// Core physics (clear, P2G, mark_active, prepare_indirect, grid_update_sparse, G2P)
    /// runs every call. Less critical passes run at reduced rates:
    /// - `compute_activity` + `update_sleep`: every 100ms
    /// - `compute_occupancy`: every 200ms
    /// - Reactions (`step_react`): caller-controlled, recommended 100ms interval
    ///
    /// Voxelize always runs because the render pass reads it every frame.
    pub fn step_physics_with_time(&self, cmd: vk::CommandBuffer, now: Instant) {
        self.record_core_physics(cmd);

        // Activity tracking + sleep update: every 100ms
        let sleep_elapsed = now.duration_since(self.last_sleep_update_time.get());
        if sleep_elapsed.as_millis() >= 100 {
            self.record_activity_and_sleep(cmd);
            self.last_sleep_update_time.set(now);
        }

        self.record_voxelize(cmd);

        // Particle sorting (improves cache coherence + warp uniformity): every 500ms
        let sort_elapsed = now.duration_since(self.last_sort_time.get());
        if sort_elapsed.as_millis() >= 500 {
            tracing::debug!(
                num_particles = self.num_particles,
                "Sorting particles by brick (copy sorted->main)"
            );
            self.sort_particles(cmd);
            self.last_sort_time.set(now);
        }

        // Brick occupancy (render optimization): every 200ms
        let occupancy_elapsed = now.duration_since(self.last_occupancy_time.get());
        if occupancy_elapsed.as_millis() >= 200 {
            self.record_occupancy(cmd);
            self.last_occupancy_time.set(now);
        }

        self.frame_number.set(self.frame_number.get().wrapping_add(1));
    }

    /// Record core physics dispatches, choosing between sparse hash grid and
    /// dense flat grid paths based on `use_sparse_grid`.
    ///
    /// These MUST run every substep for correct simulation.
    fn record_core_physics(&self, cmd: vk::CommandBuffer) {
        // Clear OOB flag before physics (G2P will set it if any particle escapes)
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.oob_flag_buffer.buffer, 0, 4, 0);
        }
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        if self.use_sparse_grid {
            self.record_core_physics_sparse(cmd);
        } else {
            self.record_core_physics_dense(cmd);
        }
    }

    /// Dense flat-grid physics path: clear_grid_sparse (indirect on active list),
    /// mark_active, prepare_indirect, grid_update_sparse (indirect), dense P2G/G2P.
    ///
    /// This is the original dispatch chain using the flat `grid_buffer`.
    fn record_core_physics_dense(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let dt = shared::DT;
        let gravity = shared::GRAVITY;
        let particle_wg = (num_particles + 63) / 64;

        // 1. Clear grid (sparse) — clears only cells that were active last frame.
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.clear_grid_sparse_pass.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.clear_grid_sparse_pass.pipeline_layout,
                0,
                &[self.clear_grid_sparse_pass.descriptor_set],
                &[],
            );
            pipeline::cmd_dispatch_indirect(
                &self.device,
                cmd,
                &self.indirect_dispatch_buffer,
                0,
            );
        }
        passes::barrier(cmd, &self.device);

        // 2. Reset active_count to 0 AND clear mark_buffer for this frame.
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.active_count_buffer.buffer, 0, 4, 0);
            self.device
                .cmd_fill_buffer(cmd, self.mark_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // 3. P2G (particle dispatch)
        let frame = self.frame_number.get();
        let p2g_pc = P2gPushConstants {
            grid_size,
            dt,
            num_particles,
            frame_number: frame,
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

        // 4. mark_active (particle dispatch) — build new active cell list
        let mark_pc = MarkActivePushConstants {
            grid_size,
            num_particles,
            _pad0: 0,
            _pad1: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.mark_active_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&mark_pc),
        );
        passes::barrier(cmd, &self.device);

        // 5. prepare_indirect (1,1,1) — write dispatch args from new active_count
        passes::dispatch(
            &self.device,
            cmd,
            &self.prepare_indirect_pass,
            1,
            1,
            1,
            &[],
        );
        passes::barrier(cmd, &self.device);

        // 6. grid_update_sparse (indirect) — physics on active cells only
        let grid_sparse_pc = GridUpdateSparsePushConstants {
            grid_size,
            dt,
            gravity,
            _pad: 0,
        };
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.grid_update_sparse_pass.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.grid_update_sparse_pass.pipeline_layout,
                0,
                &[self.grid_update_sparse_pass.descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.grid_update_sparse_pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&grid_sparse_pc),
            );
            pipeline::cmd_dispatch_indirect(
                &self.device,
                cmd,
                &self.indirect_dispatch_buffer,
                0,
            );
        }
        passes::barrier(cmd, &self.device);

        // 7. G2P (particle dispatch)
        let g2p_pc = G2pPushConstants {
            grid_size,
            dt,
            num_particles,
            num_rules: self.num_phase_rules,
            frame_number: frame,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
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
    }

    /// Sparse hash-grid physics path: clear_hash_grid, p2g_sparse, grid_update_hash,
    /// g2p_sparse.
    ///
    /// Uses a hash table instead of a flat 3D grid buffer. Only occupied cells are
    /// allocated in the hash table, so memory usage is proportional to particle count
    /// rather than grid volume. This enables much larger grids (512³+) without
    /// allocating terabytes of flat buffer.
    ///
    /// Dispatch order:
    /// 1. `clear_hash_grid` — reset all keys to EMPTY sentinel, zero all values
    /// 2. `p2g_sparse` — scatter particle data into hash grid via atomic insert
    /// 3. `grid_update_hash` — update physics on all occupied hash grid slots
    /// 4. `g2p_sparse` — gather grid data back to particles via hash grid lookup
    fn record_core_physics_sparse(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let dt = shared::DT;
        let gravity = shared::GRAVITY;
        let particle_wg = (num_particles + 63) / 64;
        let hash_capacity = HASH_GRID_DEFAULT_CAPACITY;
        let hash_wg = (hash_capacity + 63) / 64;
        let frame = self.frame_number.get();

        // 1. Clear hash grid — reset keys to EMPTY, zero values
        let clear_pc = ClearHashGridPushConstants {
            hash_capacity,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.clear_hash_grid_pass,
            hash_wg,
            1,
            1,
            bytemuck::bytes_of(&clear_pc),
        );
        passes::barrier(cmd, &self.device);

        // 2. P2G sparse (particle dispatch) — scatter into hash grid
        let p2g_pc = P2gSparsePushConstants {
            grid_size,
            dt,
            num_particles,
            frame_number: frame,
            hash_capacity,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.p2g_hash_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&p2g_pc),
        );
        passes::barrier(cmd, &self.device);

        // 3. Grid update hash — physics on occupied hash grid slots only
        let grid_pc = GridUpdateHashPushConstants {
            grid_size,
            dt,
            gravity,
            hash_capacity,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.grid_update_hash_pass,
            hash_wg,
            1,
            1,
            bytemuck::bytes_of(&grid_pc),
        );
        passes::barrier(cmd, &self.device);

        // 4. G2P sparse (particle dispatch) — gather from hash grid
        let g2p_pc = G2pSparsePushConstants {
            grid_size,
            dt,
            num_particles,
            num_rules: self.num_phase_rules,
            frame_number: frame,
            hash_capacity,
            _pad0: 0,
            _pad1: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.g2p_hash_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&g2p_pc),
        );
        passes::barrier(cmd, &self.device);
    }

    /// Record activity tracking and brick sleep update dispatches.
    ///
    /// Clears the activity map, computes per-brick activity from particle
    /// positions, then updates the graduated sleep state for each brick.
    fn record_activity_and_sleep(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let particle_wg = (num_particles + 63) / 64;

        // Clear activity map
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.activity_map_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        // Barrier: TRANSFER → COMPUTE
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }
        let activity_pc = ComputeActivityPushConstants {
            grid_size,
            num_particles,
            brick_size: shared::BRICK_SIZE,
            _pad: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.compute_activity_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&activity_pc),
        );
        passes::barrier(cmd, &self.device);

        // 9. Clear any_active flag before update_sleep
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.any_active_buffer.buffer, 0, 4, 0);
        }
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // Update brick sleep state from activity map
        let sleep_pc = UpdateSleepPushConstants {
            total_bricks: TOTAL_BRICKS,
            sleep_threshold: SLEEP_THRESHOLD,
            bricks_per_axis: shared::BRICKS_PER_AXIS,
            _pad: 0,
        };
        let brick_wg = (TOTAL_BRICKS + 63) / 64;
        passes::dispatch(
            &self.device,
            cmd,
            &self.update_sleep_pass,
            brick_wg,
            1,
            1,
            bytemuck::bytes_of(&sleep_pc),
        );
        passes::barrier(cmd, &self.device);
    }

    /// Record voxelization dispatches: clear_voxels + voxelize.
    ///
    /// Must run every frame since the render pass reads the voxel buffer.
    fn record_voxelize(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let grid_wg = grid_size / 4;
        let particle_wg = (num_particles + 63) / 64;

        // 10. Clear voxels (dense) — skips frozen bricks via sleep_state
        let vox_pc = VoxelizePushConstants {
            grid_size,
            num_particles,
            frame_number: self.frame_number.get(),
            _pad0: 0,
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

    /// Record brick occupancy compute dispatch.
    ///
    /// Populates the brick_occupancy_buffer used by the render pass to skip
    /// empty bricks during ray marching. Can run at reduced rate (e.g., every 200ms).
    fn record_occupancy(&self, cmd: vk::CommandBuffer) {
        // Clear super-brick buffer before occupancy pass (uses atomicMax)
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.super_brick_occupancy_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }
        let occupancy_pc = ComputeOccupancyPushConstants {
            grid_size: GRID_SIZE,
            brick_size: shared::BRICK_SIZE,
            bricks_per_axis: shared::BRICKS_PER_AXIS,
            _pad: 0,
        };
        let brick_wg = (TOTAL_BRICKS + 63) / 64;
        passes::dispatch(
            &self.device,
            cmd,
            &self.compute_occupancy_pass,
            brick_wg,
            1,
            1,
            bytemuck::bytes_of(&occupancy_pc),
        );
        passes::barrier(cmd, &self.device);
    }

    /// Dispatch a dense grid clear (zeros ALL cells).
    ///
    /// Use this for initialization or manual reset. Normal physics uses
    /// sparse clear via the active cell list, so this is not called every frame.
    pub fn clear_grid_dense(&self, cmd: vk::CommandBuffer) {
        let grid_wg = GRID_SIZE / 4;
        passes::dispatch(&self.device, cmd, &self.clear_grid_pass, grid_wg, grid_wg, grid_wg, &[]);
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

    /// Record the counting sort dispatch chain into the given command buffer.
    ///
    /// Sorts particles by brick_id so that all particles in the same brick are
    /// contiguous in the sorted_particle_buffer, then copies sorted particles
    /// back to the main particle_buffer. This improves cache coherence for
    /// P2G/G2P and ensures sleeping particles form complete warps that can
    /// early-exit without divergence.
    ///
    /// Also builds a compact list of active (non-sleeping, non-empty) bricks.
    ///
    /// The command buffer must already be in the recording state.
    ///
    /// Dispatch order:
    /// 1. `vkCmdFillBuffer` -- clear brick_count to 0
    /// 2. Barrier (TRANSFER -> COMPUTE)
    /// 3. `count_per_brick` -- count particles per brick
    /// 4. Barrier
    /// 5. `prefix_sum` -- exclusive prefix sum over brick_count
    /// 6. Barrier
    /// 7. `vkCmdCopyBuffer` -- copy brick_offset to write_offset
    /// 8. Barrier (TRANSFER -> COMPUTE)
    /// 9. `scatter_particles` -- scatter to sorted positions
    /// 10. Barrier
    /// 11. `vkCmdFillBuffer` -- clear active_brick_count to 0
    /// 12. Barrier (TRANSFER -> COMPUTE)
    /// 13. `compact_active_bricks` -- build active brick list
    /// 14. Barrier
    pub fn sort_particles(&self, cmd: vk::CommandBuffer) {
        let num_particles = self.num_particles;
        let grid_size = GRID_SIZE;
        let particle_wg = (num_particles + 63) / 64;
        let brick_wg = (TOTAL_BRICKS + 63) / 64;

        // 1. Clear brick_count to 0
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.brick_count_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }

        // 2. Barrier: TRANSFER_WRITE -> SHADER_READ | SHADER_WRITE
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // 3. count_per_brick
        let count_pc = CountPerBrickPushConstants {
            num_particles,
            grid_size,
            brick_size: shared::BRICK_SIZE,
            _pad: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.count_per_brick_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&count_pc),
        );
        passes::barrier(cmd, &self.device);

        // 5. prefix_sum (single workgroup)
        let prefix_pc = PrefixSumPushConstants {
            total_bricks: TOTAL_BRICKS,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.prefix_sum_pass,
            1,
            1,
            1,
            bytemuck::bytes_of(&prefix_pc),
        );
        passes::barrier(cmd, &self.device);

        // 7. Copy brick_offset -> write_offset (scatter needs a mutable copy)
        {
            let copy_size = ((TOTAL_BRICKS as usize + 1) * 4) as vk::DeviceSize;
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: copy_size,
            };
            unsafe {
                self.device.cmd_copy_buffer(
                    cmd,
                    self.brick_offset_buffer.buffer,
                    self.write_offset_buffer.buffer,
                    &[region],
                );
            }
        }
        // 8. Barrier: TRANSFER -> COMPUTE
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // 9. scatter_particles
        let scatter_pc = ScatterParticlesPushConstants {
            num_particles,
            grid_size,
            brick_size: shared::BRICK_SIZE,
            _pad: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.scatter_particles_pass,
            particle_wg,
            1,
            1,
            bytemuck::bytes_of(&scatter_pc),
        );
        passes::barrier(cmd, &self.device);

        // 10. Copy sorted particles back to main particle buffer.
        // This gives P2G/G2P brick-sorted particles, improving cache coherence
        // and warp uniformity: sleeping particles form complete warps that skip
        // immediately instead of causing divergence within mixed warps.
        {
            let copy_size = (num_particles as u64) * (mem::size_of::<Particle>() as u64);
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: copy_size,
            };
            unsafe {
                self.device.cmd_copy_buffer(
                    cmd,
                    self.sorted_particle_buffer.buffer,
                    self.particle_buffer.buffer,
                    &[region],
                );
            }
        }
        // Barrier: TRANSFER_WRITE -> COMPUTE_SHADER (P2G/G2P will read particle_buffer)
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // 11. Clear active_brick_count to 0
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.active_brick_count_buffer.buffer, 0, 4, 0);
        }
        // 12. Barrier: TRANSFER -> COMPUTE
        {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        // 13. compact_active_bricks
        let compact_pc = CompactActiveBricksPushConstants {
            total_bricks: TOTAL_BRICKS,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        passes::dispatch(
            &self.device,
            cmd,
            &self.compact_active_bricks_pass,
            brick_wg,
            1,
            1,
            bytemuck::bytes_of(&compact_pc),
        );
        passes::barrier(cmd, &self.device);
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
    /// Uses dirty-tile optimization: first computes which 16x16 pixel tiles
    /// need re-rendering (based on camera movement and brick sleep state),
    /// then only ray-marches dirty tiles. Clean tiles are copied from the
    /// previous frame buffer.
    ///
    /// The command buffer must be in recording state.
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

        // Detect camera movement
        let prev_eye = self.prev_eye.get();
        let camera_moved = Self::has_camera_moved(eye, target, prev_eye, self.prev_target.get());

        // 1. Compute dirty tiles
        let dirty_tiles_pc = DirtyTilesPushConstants {
            width,
            height,
            grid_size: GRID_SIZE,
            bricks_per_axis: shared::BRICKS_PER_AXIS,
            eye: glam::Vec4::new(
                eye[0],
                eye[1],
                eye[2],
                if camera_moved { 1.0 } else { 0.0 },
            ),
            target: glam::Vec4::new(target[0], target[1], target[2], 0.0),
        };

        let tile_wg = (DIRTY_TILE_COUNT + 63) / 64;
        passes::dispatch(
            &self.device,
            cmd,
            &self.compute_dirty_tiles_pass,
            tile_wg,
            1,
            1,
            bytemuck::bytes_of(&dirty_tiles_pc),
        );
        passes::barrier(cmd, &self.device);

        // 2. Render with dirty-tile check
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

        // Store current camera for next frame comparison
        self.prev_eye.set(eye);
        self.prev_target.set(target);
    }

    /// Check if the camera has moved between frames.
    ///
    /// Returns `true` on the first frame (prev_eye is NaN) or if the eye/target
    /// position changed by more than a tiny epsilon.
    fn has_camera_moved(
        eye: [f32; 3],
        target: [f32; 3],
        prev_eye: [f32; 3],
        prev_target: [f32; 3],
    ) -> bool {
        // First frame: NaN comparison always fails, so camera is "moved"
        if prev_eye[0].is_nan() {
            return true;
        }
        Self::vec3_diff_exceeds(eye, prev_eye)
            || Self::vec3_diff_exceeds(target, prev_target)
    }

    /// Check if two 3-component vectors differ by more than epsilon.
    fn vec3_diff_exceeds(a: [f32; 3], b: [f32; 3]) -> bool {
        let eps = 0.001;
        let dx = (a[0] - b[0]).abs();
        let dy = (a[1] - b[1]).abs();
        let dz = (a[2] - b[2]).abs();
        dx > eps || dy > eps || dz > eps
    }

    /// Set the world-space origin of the simulation domain.
    ///
    /// This defines where grid voxel [0,0,0] sits in world coordinates.
    /// Used by the far-field render pass to convert between grid-local and world space.
    pub fn set_sim_origin(&mut self, origin: [f32; 3]) {
        self.sim_origin = origin;
    }

    /// Upload far-field voxel data for a set of chunks.
    ///
    /// Each chunk provides its world-space origin and a 64^3 u8 material palette.
    /// The `data` slice for each chunk must be exactly `FAR_FIELD_VOXELS_PER_CHUNK` bytes.
    /// Chunks are appended sequentially starting from offset 0.
    ///
    /// All voxel data is concatenated into a single upload to avoid per-chunk
    /// staging buffer overhead.
    ///
    /// Returns the number of chunks actually uploaded (may be less than provided
    /// if `MAX_FAR_CHUNKS` is exceeded).
    pub fn update_far_field(
        &mut self,
        ctx: &VulkanContext,
        chunks: &[(ChunkTableEntry, &[u8])],
    ) -> Result<u32> {
        let max = MAX_FAR_CHUNKS as usize;
        let count = chunks.len().min(max);

        // Build table entries with sequential offsets
        let mut table = vec![ChunkTableEntry::zeroed(); max];
        let voxels_per_chunk = shared::FAR_FIELD_VOXELS_PER_CHUNK as usize;

        // Concatenate all voxel data into a single buffer for one upload
        let mut all_voxels = vec![0u8; count * voxels_per_chunk];

        for (i, (entry, data)) in chunks.iter().take(count).enumerate() {
            table[i] = ChunkTableEntry {
                origin: entry.origin,
                voxel_offset: (i * voxels_per_chunk) as u32,
                _pad: [0; 3],
            };
            // Ensure w=1 for valid entries
            table[i].origin[3] = 1.0;

            // Copy voxel data into concatenated buffer
            let dst_start = i * voxels_per_chunk;
            let copy_len = data.len().min(voxels_per_chunk);
            all_voxels[dst_start..dst_start + copy_len].copy_from_slice(&data[..copy_len]);
        }

        // Upload chunk table
        buffer::upload(ctx, &table, &self.far_field_chunk_table)?;

        // Upload all voxel data in a single transfer
        if count > 0 {
            buffer::upload(ctx, &all_voxels, &self.far_field_voxel_buffer)?;
        }

        self.far_field_chunk_count = count as u32;
        tracing::info!(
            "Uploaded {} far-field chunks ({:.1}MB voxel data)",
            count,
            (count * voxels_per_chunk) as f64 / (1024.0 * 1024.0),
        );

        Ok(count as u32)
    }

    /// Record a far-field render dispatch into the given command buffer.
    ///
    /// Runs after the main render pass. For each pixel showing sky, casts rays
    /// into far-field chunks and writes hit colors.
    /// The command buffer must be in recording state.
    pub fn render_far_field(
        &self,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        eye: [f32; 3],
        target: [f32; 3],
    ) {
        if self.far_field_chunk_count == 0 {
            return;
        }

        passes::barrier(cmd, &self.device);

        let push = FarFieldPushConstants {
            width,
            height,
            chunk_count: self.far_field_chunk_count,
            grid_size: GRID_SIZE,
            eye: glam::Vec4::new(eye[0], eye[1], eye[2], 0.0),
            target: glam::Vec4::new(target[0], target[1], target[2], 0.0),
            sim_origin: glam::Vec4::new(
                self.sim_origin[0],
                self.sim_origin[1],
                self.sim_origin[2],
                0.0,
            ),
        };

        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        passes::dispatch(
            &self.device,
            cmd,
            &self.render_far_field_pass,
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
    /// before the buffer is copied to the swapchain, then copy current render
    /// output to the previous-frame buffer for dirty-tile reuse next frame.
    ///
    /// Must be called after [`render`] and [`render_toolbar`] (if used).
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

        // Copy current render output to previous frame buffer for next frame
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
            &self.compute_activity_pass,
            &self.compute_occupancy_pass,
            &self.render_far_field_pass,
            &self.compute_dirty_tiles_pass,
            &self.update_sleep_pass,
            &self.count_per_brick_pass,
            &self.prefix_sum_pass,
            &self.scatter_particles_pass,
            &self.compact_active_bricks_pass,
            &self.mark_active_pass,
            &self.prepare_indirect_pass,
            &self.clear_grid_sparse_pass,
            &self.grid_update_sparse_pass,
            &self.clear_hash_grid_pass,
            &self.p2g_hash_pass,
            &self.g2p_hash_pass,
            &self.grid_update_hash_pass,
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
        let activity_map_buf = std::mem::replace(
            &mut self.activity_map_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let sleep_counter_buf = std::mem::replace(
            &mut self.sleep_counter_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let sleep_state_buf = std::mem::replace(
            &mut self.sleep_state_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let brick_occupancy_buf = std::mem::replace(
            &mut self.brick_occupancy_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let super_brick_occupancy_buf = std::mem::replace(
            &mut self.super_brick_occupancy_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let far_field_voxel_buf = std::mem::replace(
            &mut self.far_field_voxel_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let far_field_table_buf = std::mem::replace(
            &mut self.far_field_chunk_table,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let any_active_buf = std::mem::replace(
            &mut self.any_active_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let oob_flag_buf = std::mem::replace(
            &mut self.oob_flag_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let prev_render_output_buf = std::mem::replace(
            &mut self.prev_render_output_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let dirty_tile_buf = std::mem::replace(
            &mut self.dirty_tile_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let mark_buf = std::mem::replace(
            &mut self.mark_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let active_cells_buf = std::mem::replace(
            &mut self.active_cells_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let active_count_buf = std::mem::replace(
            &mut self.active_count_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let indirect_buf = std::mem::replace(
            &mut self.indirect_dispatch_buffer,
            GpuBuffer {
                buffer: vk::Buffer::null(),
                allocation: None,
                size: 0,
            },
        );
        let hash_keys_buf = std::mem::replace(
            &mut self.hash_grid_keys_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let hash_values_buf = std::mem::replace(
            &mut self.hash_grid_values_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let brick_count_buf = std::mem::replace(
            &mut self.brick_count_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let brick_offset_buf = std::mem::replace(
            &mut self.brick_offset_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let write_offset_buf = std::mem::replace(
            &mut self.write_offset_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let sorted_particle_buf = std::mem::replace(
            &mut self.sorted_particle_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let active_brick_list_buf = std::mem::replace(
            &mut self.active_brick_list_buffer,
            GpuBuffer { buffer: vk::Buffer::null(), allocation: None, size: 0 },
        );
        let active_brick_count_buf = std::mem::replace(
            &mut self.active_brick_count_buffer,
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
        buffer::destroy_buffer(ctx, activity_map_buf);
        buffer::destroy_buffer(ctx, sleep_counter_buf);
        buffer::destroy_buffer(ctx, sleep_state_buf);
        buffer::destroy_buffer(ctx, brick_occupancy_buf);
        buffer::destroy_buffer(ctx, super_brick_occupancy_buf);
        buffer::destroy_buffer(ctx, far_field_voxel_buf);
        buffer::destroy_buffer(ctx, far_field_table_buf);
        buffer::destroy_buffer(ctx, any_active_buf);
        buffer::destroy_buffer(ctx, oob_flag_buf);
        buffer::destroy_buffer(ctx, prev_render_output_buf);
        buffer::destroy_buffer(ctx, dirty_tile_buf);
        buffer::destroy_buffer(ctx, brick_count_buf);
        buffer::destroy_buffer(ctx, brick_offset_buf);
        buffer::destroy_buffer(ctx, write_offset_buf);
        buffer::destroy_buffer(ctx, sorted_particle_buf);
        buffer::destroy_buffer(ctx, active_brick_list_buf);
        buffer::destroy_buffer(ctx, active_brick_count_buf);
        buffer::destroy_buffer(ctx, mark_buf);
        buffer::destroy_buffer(ctx, active_cells_buf);
        buffer::destroy_buffer(ctx, active_count_buf);
        buffer::destroy_buffer(ctx, indirect_buf);
        buffer::destroy_buffer(ctx, hash_keys_buf);
        buffer::destroy_buffer(ctx, hash_values_buf);
    }
}
