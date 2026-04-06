//! Stream compaction compute shader for the CA substrate.
//!
//! Builds a list of dirty chunk IDs for indirect dispatch of subsequent CA passes.
//! One thread per chunk checks the dirty flag in metadata and atomically appends
//! the slot ID to the dirty list.
//!
//! Dirty list buffer layout (in u32s):
//! - `[0]`: dirty_count (atomic counter)
//! - `[1]`: dispatch_x (filled by thread 0 of a finalize pass, or same pass)
//! - `[2]`: dispatch_y = 1
//! - `[3]`: dispatch_z = 1
//! - `[4..4+N]`: dirty chunk slot IDs

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::ca_types;

/// Push constants for the compact pass.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CaCompactPush {
    /// Total number of loaded chunks.
    pub total_chunks: u32,
    /// Workgroups-per-chunk for subsequent indirect dispatch (e.g., 512 for thermal).
    pub workgroups_per_chunk: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 2],
}

/// Header offset in the dirty list: slot IDs start at index 4.
const DIRTY_LIST_HEADER: u32 = 4;

/// Checks one chunk's metadata and appends it to the dirty list if flagged dirty.
///
/// Separated into a helper to satisfy the rust-gpu entry-point-must-call-helper rule
/// (see CLAUDE.md trap #4a).
fn compact_impl(
    chunk_id: u32,
    metadata: &[u32],
    dirty_list: &mut [u32],
    total_chunks: u32,
    workgroups_per_chunk: u32,
) {
    if chunk_id >= total_chunks {
        return;
    }

    // Read flags from metadata: bit 0 = dirty
    let flags = ca_types::meta_flags(metadata, chunk_id);
    let is_dirty = (flags & 1) != 0;

    if is_dirty {
        // Atomic increment of dirty_count at dirty_list[0]
        let old_count = unsafe {
            spirv_std::arch::atomic_i_add::<
                u32,
                { spirv_std::memory::Scope::Device as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(&mut dirty_list[0], 1)
        };
        // Write slot ID at the next available position
        dirty_list[(DIRTY_LIST_HEADER + old_count) as usize] = chunk_id;
    }
}

/// Compute shader entry point: stream compaction of dirty chunks.
///
/// Descriptor set 0:
/// - binding 0: metadata buffer (ChunkGpuMeta array as `&[u32]`)
/// - binding 1: dirty_list buffer (`&mut [u32]`)
///
/// Dispatch: `(ceil(total_chunks / 256), 1, 1)` workgroups.
#[spirv(compute(threads(256)))]
pub fn ca_compact(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] metadata: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dirty_list: &mut [u32],
    #[spirv(push_constant)] push: &CaCompactPush,
) {
    compact_impl(
        global_id.x,
        metadata,
        dirty_list,
        push.total_chunks,
        push.workgroups_per_chunk,
    );
}

/// Finalize pass: writes VkDispatchIndirectCommand to dirty_list header.
///
/// Must be dispatched as a single workgroup (1,1,1) AFTER ca_compact and a barrier.
/// Reads dirty_count from `dirty_list[0]` and writes dispatch args.
#[spirv(compute(threads(1)))]
pub fn ca_compact_finalize(
    #[spirv(global_invocation_id)] _global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dirty_list: &mut [u32],
    #[spirv(push_constant)] push: &CaCompactPush,
) {
    finalize_impl(dirty_list, push.workgroups_per_chunk);
}

/// Writes the indirect dispatch command based on the dirty count.
fn finalize_impl(dirty_list: &mut [u32], workgroups_per_chunk: u32) {
    let dirty_count = dirty_list[0];
    dirty_list[1] = dirty_count * workgroups_per_chunk; // dispatch_x
    dirty_list[2] = 1; // dispatch_y
    dirty_list[3] = 1; // dispatch_z
}
