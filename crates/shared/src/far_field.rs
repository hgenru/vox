//! Far-field voxel rendering types.
//!
//! Types for rendering chunks beyond the active simulation zone using
//! pre-baked voxel data. Each chunk is a 64x64x64 block of u8 material palette
//! indices stored in a compact buffer on the GPU.

use bytemuck::{Pod, Zeroable};

/// Maximum number of far-field chunks in the atlas.
pub const MAX_FAR_CHUNKS: u32 = 128;

/// Voxels per chunk axis (64).
pub const FAR_CHUNK_SIZE: u32 = 64;

/// Total voxels per chunk (64^3 = 262144).
pub const FAR_FIELD_VOXELS_PER_CHUNK: u32 = FAR_CHUNK_SIZE * FAR_CHUNK_SIZE * FAR_CHUNK_SIZE;

/// Total bytes of voxel data in the far-field atlas.
///
/// Each voxel is a single u8 material palette index.
/// 128 chunks * 262144 voxels/chunk = 33554432 bytes (32 MB).
pub const FAR_FIELD_VOXEL_BUFFER_SIZE: u32 = MAX_FAR_CHUNKS * FAR_FIELD_VOXELS_PER_CHUNK;

/// Entry in the far-field chunk table mapping a chunk to its voxel data.
///
/// 32 bytes, 16-byte aligned. Stored in a GPU storage buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ChunkTableEntry {
    /// World-space origin of this chunk (chunk_coord * FAR_CHUNK_SIZE).
    /// `w` = 1.0 if this entry is valid, 0.0 otherwise.
    pub origin: [f32; 4],
    /// Offset into the far_field_voxel_buffer in voxels (not bytes).
    pub voxel_offset: u32,
    /// Padding to 32 bytes (16-byte aligned).
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn chunk_table_entry_size_and_alignment() {
        assert_eq!(mem::size_of::<ChunkTableEntry>(), 32);
        assert_eq!(mem::align_of::<ChunkTableEntry>(), 4);
    }

    #[test]
    fn far_field_constants() {
        assert_eq!(FAR_FIELD_VOXELS_PER_CHUNK, 262144);
        assert_eq!(MAX_FAR_CHUNKS, 128);
        // 128 * 262144 = 33554432 bytes = 32 MB
        assert_eq!(FAR_FIELD_VOXEL_BUFFER_SIZE, 33_554_432);
    }

    #[test]
    fn chunk_table_entry_is_pod() {
        // Verify bytemuck traits work
        let entry = ChunkTableEntry::zeroed();
        let bytes: &[u8] = bytemuck::bytes_of(&entry);
        assert_eq!(bytes.len(), 32);
    }
}
