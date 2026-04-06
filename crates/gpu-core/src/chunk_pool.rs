//! Gigabuffer chunk pool allocator for Simulation 2.0.
//!
//! Instead of allocating separate GPU buffers per chunk, [`ChunkPool`] manages
//! a single large "gigabuffer" with fixed-size slots for 32x32x32 voxel chunks.
//! Each slot contains voxel data (128 KB as `u32`) plus a dirty bitmask (4 KB).
//!
//! Slot allocation/freeing is CPU-side via a LIFO free stack.
//! Upload/download uses staging buffers at the correct byte offsets.

use ash::vk;

use crate::buffer::{
    create_device_local_buffer, create_readback_staging_buffer, create_upload_staging_buffer,
    destroy_buffer, GpuBuffer,
};
use crate::context::VulkanContext;
use crate::error::GpuError;

// ---------------------------------------------------------------------------
// Local constants (will be replaced by shared:: imports after sim2 merge)
// ---------------------------------------------------------------------------

/// Number of voxels per chunk dimension.
const CHUNK_DIM: u32 = 32;

/// Total voxels in a chunk: 32^3 = 32768.
const VOXELS_PER_CHUNK: u32 = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;

/// Voxel data size per slot: 32768 voxels * 4 bytes = 131072 bytes (128 KB).
const SLOT_VOXEL_BYTES: u32 = VOXELS_PER_CHUNK * 4;

/// Dirty bitmask size per slot: one bit per voxel, stored as u32 words.
/// 32768 / 32 = 1024 u32s = 4096 bytes (4 KB).
const SLOT_BITMASK_BYTES: u32 = VOXELS_PER_CHUNK / 8;

/// Total bytes per slot in the gigabuffer: 128 KB + 4 KB = 135168 bytes.
const SLOT_TOTAL_BYTES: u32 = SLOT_VOXEL_BYTES + SLOT_BITMASK_BYTES;

/// Size of ChunkGpuMeta in the metadata buffer (padded to 64 bytes).
const METADATA_SIZE_BYTES: u32 = 64;

/// Size of a dirty list entry (one u32 slot ID).
const DIRTY_LIST_ENTRY_BYTES: u32 = 4;

/// Size of VkDispatchIndirectCommand (3 x u32 = 12 bytes).
const DISPATCH_ARGS_BYTES: u32 = 12;

/// Fixed-size pool of chunk slots in a single GPU buffer.
///
/// Each slot holds 32x32x32 voxels (128 KB as `u32`) plus a dirty bitmask (4 KB).
/// Allocation and freeing are CPU-side operations using a LIFO free stack.
pub struct ChunkPool {
    /// Single large buffer holding all chunk voxel data + bitmasks.
    voxel_buffer: GpuBuffer,
    /// Chunk metadata array (ChunkGpuMeta per slot).
    metadata_buffer: GpuBuffer,
    /// Compacted list of dirty chunk slot IDs + indirect dispatch args.
    dirty_list_buffer: GpuBuffer,
    /// CPU-side free slot stack (LIFO for cache-friendly reuse).
    free_stack: Vec<u32>,
    /// Total number of slots in the pool.
    slot_count: u32,
    /// Size of each slot in bytes.
    slot_size_bytes: u32,
}

impl ChunkPool {
    /// Create a new chunk pool with the given number of slots.
    ///
    /// Allocates three GPU buffers:
    /// - `voxel_buffer`: `slot_count * 135168` bytes (voxel data + bitmasks)
    /// - `metadata_buffer`: `slot_count * 64` bytes (per-slot metadata)
    /// - `dirty_list_buffer`: `slot_count * 4 + 12` bytes (slot IDs + dispatch args)
    ///
    /// All slots start as free. Use [`allocate_slot`] to claim one.
    pub fn new(ctx: &VulkanContext, slot_count: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(slot_count > 0, "ChunkPool slot_count must be > 0");

        let voxel_buf_size = u64::from(slot_count) * u64::from(SLOT_TOTAL_BYTES);
        let metadata_buf_size = u64::from(slot_count) * u64::from(METADATA_SIZE_BYTES);
        let dirty_list_buf_size =
            u64::from(slot_count) * u64::from(DIRTY_LIST_ENTRY_BYTES) + u64::from(DISPATCH_ARGS_BYTES);

        let voxel_buffer = create_device_local_buffer(
            ctx,
            voxel_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "chunk-pool-voxels",
        )?;

        let metadata_buffer = create_device_local_buffer(
            ctx,
            metadata_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "chunk-pool-metadata",
        )?;

        let dirty_list_buffer = create_device_local_buffer(
            ctx,
            dirty_list_buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            "chunk-pool-dirty-list",
        )?;

        // Initialize free stack with all slots (highest ID on top so pop yields 0 first)
        let free_stack: Vec<u32> = (0..slot_count).rev().collect();

        tracing::info!(
            "ChunkPool created: {} slots, voxel_buffer={} bytes, metadata={} bytes, dirty_list={} bytes",
            slot_count,
            voxel_buf_size,
            metadata_buf_size,
            dirty_list_buf_size,
        );

        Ok(Self {
            voxel_buffer,
            metadata_buffer,
            dirty_list_buffer,
            free_stack,
            slot_count,
            slot_size_bytes: SLOT_TOTAL_BYTES,
        })
    }

    /// Allocate a chunk slot. Returns `None` if the pool is full.
    ///
    /// Uses a LIFO stack so recently freed slots are reused first.
    pub fn allocate_slot(&mut self) -> Option<u32> {
        self.free_stack.pop()
    }

    /// Free a chunk slot back to the pool.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `slot_id >= slot_count`.
    pub fn free_slot(&mut self, slot_id: u32) {
        debug_assert!(
            slot_id < self.slot_count,
            "free_slot: slot_id {} >= slot_count {}",
            slot_id,
            self.slot_count,
        );
        self.free_stack.push(slot_id);
    }

    /// Number of currently allocated (in-use) slots.
    pub fn allocated_count(&self) -> u32 {
        self.slot_count - self.free_stack.len() as u32
    }

    /// Total number of slots in the pool.
    pub fn slot_count(&self) -> u32 {
        self.slot_count
    }

    /// Size of each slot in bytes (voxel data + dirty bitmask).
    pub fn slot_size_bytes(&self) -> u32 {
        self.slot_size_bytes
    }

    /// Byte offset of a slot's voxel data in the gigabuffer.
    ///
    /// Each slot occupies [`SLOT_TOTAL_BYTES`] contiguous bytes; voxel data
    /// starts at the beginning of that region.
    pub fn voxel_offset(slot_id: u32) -> u64 {
        u64::from(slot_id) * u64::from(SLOT_TOTAL_BYTES)
    }

    /// Byte offset of a slot's dirty bitmask in the gigabuffer.
    ///
    /// The bitmask immediately follows the voxel data within the slot.
    pub fn bitmask_offset(slot_id: u32) -> u64 {
        u64::from(slot_id) * u64::from(SLOT_TOTAL_BYTES) + u64::from(SLOT_VOXEL_BYTES)
    }

    /// Get the voxel/bitmask gigabuffer handle for descriptor binding.
    pub fn voxel_buffer(&self) -> &GpuBuffer {
        &self.voxel_buffer
    }

    /// Get the metadata buffer handle.
    pub fn metadata_buffer(&self) -> &GpuBuffer {
        &self.metadata_buffer
    }

    /// Get the dirty list buffer handle.
    pub fn dirty_list_buffer(&self) -> &GpuBuffer {
        &self.dirty_list_buffer
    }

    /// Upload voxel data for a slot via a staging buffer.
    ///
    /// `voxel_data` must contain exactly 32768 `u32` values (one per voxel).
    /// The data is copied to the correct offset within the gigabuffer.
    ///
    /// The `cmd` parameter is reserved for future batched recording; currently
    /// the upload is performed via an internal one-shot command submission.
    pub fn upload_chunk_voxels(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        slot_id: u32,
        voxel_data: &[u32],
    ) -> anyhow::Result<()> {
        let _ = cmd; // Reserved for future batched recording

        anyhow::ensure!(
            voxel_data.len() == VOXELS_PER_CHUNK as usize,
            "voxel_data must be {} u32s, got {}",
            VOXELS_PER_CHUNK,
            voxel_data.len(),
        );

        let src_bytes = bytemuck::cast_slice::<u32, u8>(voxel_data);
        Self::upload_bytes_one_shot(
            ctx,
            self.voxel_buffer.buffer,
            Self::voxel_offset(slot_id),
            src_bytes,
        )
    }

    /// Upload metadata for a slot.
    ///
    /// `metadata` must be exactly 64 bytes (a `ChunkGpuMeta` cast via bytemuck).
    ///
    /// The `cmd` parameter is reserved for future batched recording; currently
    /// the upload is performed via an internal one-shot command submission.
    pub fn upload_metadata(
        &self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        slot_id: u32,
        metadata: &[u8],
    ) -> anyhow::Result<()> {
        let _ = cmd; // Reserved for future batched recording

        anyhow::ensure!(
            metadata.len() == METADATA_SIZE_BYTES as usize,
            "metadata must be {} bytes, got {}",
            METADATA_SIZE_BYTES,
            metadata.len(),
        );

        let dst_offset = u64::from(slot_id) * u64::from(METADATA_SIZE_BYTES);
        Self::upload_bytes_one_shot(ctx, self.metadata_buffer.buffer, dst_offset, metadata)
    }

    /// Download voxel data from a slot.
    ///
    /// Creates a readback staging buffer, copies from the gigabuffer at the
    /// slot's offset, submits and waits, then maps and returns the data.
    /// Returns exactly 32768 `u32` values.
    pub fn download_chunk_voxels(
        &self,
        ctx: &VulkanContext,
        slot_id: u32,
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(
            slot_id < self.slot_count,
            "slot_id {} >= slot_count {}",
            slot_id,
            self.slot_count,
        );

        let byte_size = u64::from(SLOT_VOXEL_BYTES);
        let src_offset = Self::voxel_offset(slot_id);
        let staging = create_readback_staging_buffer(ctx, byte_size, "chunk-readback-staging")?;

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(0)
                .size(byte_size);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cmd,
                    self.voxel_buffer.buffer,
                    staging.buffer,
                    &[region],
                );
            }
        })?;

        let result = {
            let mapped = staging.mapped_slice().ok_or(GpuError::MappingFailed)?;
            let typed: &[u32] = bytemuck::cast_slice(&mapped[..byte_size as usize]);
            typed.to_vec()
        };

        destroy_buffer(ctx, staging);
        Ok(result)
    }

    /// Free all GPU resources owned by the pool.
    ///
    /// Must be called before dropping the [`VulkanContext`].
    pub fn destroy(self, ctx: &VulkanContext) {
        destroy_buffer(ctx, self.voxel_buffer);
        destroy_buffer(ctx, self.metadata_buffer);
        destroy_buffer(ctx, self.dirty_list_buffer);
        tracing::info!("ChunkPool destroyed");
    }

    /// Upload raw bytes to a target buffer at an offset using execute_one_shot.
    fn upload_bytes_one_shot(
        ctx: &VulkanContext,
        dst_buffer: vk::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) -> anyhow::Result<()> {
        let byte_size = data.len() as u64;
        let mut staging = create_upload_staging_buffer(ctx, byte_size, "chunk-upload-staging")?;

        {
            let mapped = staging
                .mapped_slice_mut()
                .ok_or(GpuError::MappingFailed)?;
            mapped[..data.len()].copy_from_slice(data);
        }

        ctx.execute_one_shot(|cmd| {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(dst_offset)
                .size(byte_size);
            unsafe {
                ctx.device
                    .cmd_copy_buffer(cmd, staging.buffer, dst_buffer, &[region]);
            }
        })?;

        destroy_buffer(ctx, staging);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Pure CPU tests (no GPU needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_slot_count_and_offsets() {
        // Verify offsets are non-overlapping and correctly aligned
        let slot0_voxel = ChunkPool::voxel_offset(0);
        let slot0_bitmask = ChunkPool::bitmask_offset(0);
        let slot1_voxel = ChunkPool::voxel_offset(1);
        let slot1_bitmask = ChunkPool::bitmask_offset(1);
        let slot7_voxel = ChunkPool::voxel_offset(7);
        let slot7_bitmask = ChunkPool::bitmask_offset(7);

        // Slot 0: voxels at 0, bitmask at 128 KB
        assert_eq!(slot0_voxel, 0);
        assert_eq!(slot0_bitmask, 131_072); // SLOT_VOXEL_BYTES

        // Slot 1 starts right after slot 0
        assert_eq!(slot1_voxel, u64::from(SLOT_TOTAL_BYTES));
        assert_eq!(
            slot1_bitmask,
            u64::from(SLOT_TOTAL_BYTES) + u64::from(SLOT_VOXEL_BYTES)
        );

        // Slot 7
        assert_eq!(slot7_voxel, 7 * u64::from(SLOT_TOTAL_BYTES));
        assert_eq!(
            slot7_bitmask,
            7 * u64::from(SLOT_TOTAL_BYTES) + u64::from(SLOT_VOXEL_BYTES)
        );

        // Non-overlapping: slot N bitmask end == slot N+1 voxel start
        let slot0_end = slot0_bitmask + u64::from(SLOT_BITMASK_BYTES);
        assert_eq!(slot0_end, slot1_voxel);

        let slot1_end = slot1_bitmask + u64::from(SLOT_BITMASK_BYTES);
        let slot2_voxel = ChunkPool::voxel_offset(2);
        assert_eq!(slot1_end, slot2_voxel);

        // All offsets are 4-byte aligned (u32 access)
        assert_eq!(slot0_voxel % 4, 0);
        assert_eq!(slot0_bitmask % 4, 0);
        assert_eq!(slot1_voxel % 4, 0);
        assert_eq!(slot7_bitmask % 4, 0);
    }

    #[test]
    fn test_slot_size_calculation() {
        assert_eq!(VOXELS_PER_CHUNK, 32_768);
        assert_eq!(SLOT_VOXEL_BYTES, 131_072); // 128 KB
        assert_eq!(SLOT_BITMASK_BYTES, 4_096); // 4 KB
        assert_eq!(SLOT_TOTAL_BYTES, 135_168); // 132 KB

        // Total buffer size for N slots
        let n: u32 = 1024;
        let expected = u64::from(n) * u64::from(SLOT_TOTAL_BYTES);
        assert_eq!(expected, 1024 * 135_168);
    }

    /// Helper to test allocation logic without GPU buffer creation.
    struct MockPoolState {
        free_stack: Vec<u32>,
        slot_count: u32,
    }

    impl MockPoolState {
        fn new(slot_count: u32) -> Self {
            let free_stack: Vec<u32> = (0..slot_count).rev().collect();
            Self {
                free_stack,
                slot_count,
            }
        }

        fn allocate_slot(&mut self) -> Option<u32> {
            self.free_stack.pop()
        }

        fn free_slot(&mut self, slot_id: u32) {
            self.free_stack.push(slot_id);
        }

        fn allocated_count(&self) -> u32 {
            self.slot_count - self.free_stack.len() as u32
        }
    }

    #[test]
    fn test_allocate_and_free() {
        let mut pool = MockPoolState::new(8);
        assert_eq!(pool.allocated_count(), 0);

        // Allocate 5 slots
        let mut allocated = Vec::new();
        for _ in 0..5 {
            let slot = pool.allocate_slot().expect("should have free slots");
            allocated.push(slot);
        }
        assert_eq!(pool.allocated_count(), 5);

        // All allocated slots should be unique
        allocated.sort();
        allocated.dedup();
        assert_eq!(allocated.len(), 5);

        // Free one slot
        pool.free_slot(allocated[2]);
        assert_eq!(pool.allocated_count(), 4);

        // Re-allocate -> get back a slot
        let reused = pool.allocate_slot().expect("should have a free slot");
        assert_eq!(pool.allocated_count(), 5);
        assert!(reused < 8); // valid slot ID
    }

    #[test]
    fn test_pool_full() {
        let mut pool = MockPoolState::new(4);

        // Allocate all slots
        for _ in 0..4 {
            assert!(pool.allocate_slot().is_some());
        }
        assert_eq!(pool.allocated_count(), 4);

        // Next allocation should fail
        assert!(pool.allocate_slot().is_none());
    }

    #[test]
    fn test_free_and_reuse() {
        let mut pool = MockPoolState::new(4);

        // LIFO: first allocation gives slot 0 (bottom of reversed stack)
        let slot0 = pool.allocate_slot().unwrap();
        assert_eq!(slot0, 0);

        // Free it
        pool.free_slot(slot0);

        // Re-allocate should give slot 0 back (LIFO: it's on top now)
        let reused = pool.allocate_slot().unwrap();
        assert_eq!(reused, 0);
    }

    // -----------------------------------------------------------------------
    // GPU tests (require VulkanContext — run with --test-threads=1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_and_destroy() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let pool = ChunkPool::new(&ctx, 16).expect("Failed to create ChunkPool");

        assert_eq!(pool.slot_count(), 16);
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.slot_size_bytes, SLOT_TOTAL_BYTES);

        // Verify buffer handles are valid
        assert_ne!(pool.voxel_buffer().buffer, vk::Buffer::null());
        assert_ne!(pool.metadata_buffer().buffer, vk::Buffer::null());
        assert_ne!(pool.dirty_list_buffer().buffer, vk::Buffer::null());

        // Verify buffer sizes
        assert_eq!(pool.voxel_buffer().size, 16 * u64::from(SLOT_TOTAL_BYTES));
        assert_eq!(
            pool.metadata_buffer().size,
            16 * u64::from(METADATA_SIZE_BYTES)
        );
        assert_eq!(
            pool.dirty_list_buffer().size,
            16 * u64::from(DIRTY_LIST_ENTRY_BYTES) + u64::from(DISPATCH_ARGS_BYTES)
        );

        pool.destroy(&ctx);
    }

    #[test]
    fn test_upload_and_download_round_trip() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let mut pool = ChunkPool::new(&ctx, 4).expect("Failed to create ChunkPool");

        let slot_id = pool.allocate_slot().expect("pool should have free slots");

        // Create test data: 32768 u32 values
        let voxel_data: Vec<u32> = (0..VOXELS_PER_CHUNK).collect();

        // Upload via the public API (cmd is unused internally)
        pool.upload_chunk_voxels(&ctx, vk::CommandBuffer::null(), slot_id, &voxel_data)
            .expect("upload failed");

        // Download and verify round-trip
        let downloaded = pool
            .download_chunk_voxels(&ctx, slot_id)
            .expect("download failed");
        assert_eq!(downloaded.len(), VOXELS_PER_CHUNK as usize);
        assert_eq!(downloaded, voxel_data);

        pool.destroy(&ctx);
    }
}
