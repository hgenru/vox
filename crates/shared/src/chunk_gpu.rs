//! GPU-side chunk metadata for the CA substrate.
//!
//! Each loaded chunk has a [`ChunkGpuMeta`] entry in the metadata buffer,
//! indexed by its slot ID in the chunk pool.

use bytemuck::{Pod, Zeroable};
use glam::IVec4;

/// GPU-side chunk metadata. Stored in metadata buffer, indexed by slot_id.
///
/// Size: 48 bytes, padded to 64 bytes for 16-byte alignment.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ChunkGpuMeta {
    /// xyz = chunk world coordinates, w = slot_id in chunk pool.
    pub world_pos: IVec4,
    /// Indices of neighbor chunks in pool: +X, -X, +Y, -Y, +Z, -Z.
    /// -1 = not loaded.
    pub neighbor_ids: [i32; 6],
    /// Activity level: 0=Sleeping, 1=CAOnly, 2=Physics.
    pub activity: u32,
    /// Bit flags: bit 0 = dirty, bit 1 = has_physics_zone.
    pub flags: u32,
    /// Padding to reach 64 bytes for 16-byte alignment.
    pub _pad: [u32; 4],
}

// Compile-time size check
const _: () = assert!(core::mem::size_of::<ChunkGpuMeta>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn test_size_align() {
        assert_eq!(size_of::<ChunkGpuMeta>(), 64);
        // With VK_EXT_scalar_block_layout, 4-byte alignment is sufficient.
        // Size is a multiple of 16 for GPU buffer array indexing.
        assert!(size_of::<ChunkGpuMeta>() % 16 == 0);
        assert!(align_of::<ChunkGpuMeta>() >= 4);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let meta: ChunkGpuMeta = ChunkGpuMeta::zeroed();
        assert_eq!(meta.world_pos, IVec4::ZERO);
        assert_eq!(meta.neighbor_ids, [0; 6]);
        assert_eq!(meta.activity, 0);
        assert_eq!(meta.flags, 0);
    }
}
