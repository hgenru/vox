//! Exclusive prefix sum compute shader.
//!
//! Computes an exclusive prefix sum over the brick_count buffer to produce
//! brick_offset. This is the second step of the counting sort pipeline:
//! count_per_brick -> prefix_sum -> scatter_particles -> compact_active_bricks
//!
//! For 32K bricks (TOTAL_BRICKS = 32768), we use a single workgroup with
//! 256 threads, each handling 128 elements. The scan is done in shared memory
//! using a work-efficient Blelloch scan (up-sweep + down-sweep).
//!
//! brick_offset[i] = sum of brick_count[0..i] (exclusive)
//! brick_offset[TOTAL_BRICKS] = total particle count (for validation)

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Push constants for the prefix_sum shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PrefixSumPushConstants {
    /// Total number of bricks.
    pub total_bricks: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    /// Padding for 16-byte alignment.
    pub _pad1: u32,
    /// Padding for 16-byte alignment.
    pub _pad2: u32,
}

/// Number of threads per workgroup for the prefix sum.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum number of elements the prefix sum can handle.
/// Must be >= TOTAL_BRICKS. 256 threads * 128 elements each = 32768.
const MAX_ELEMENTS: u32 = 32768;

/// Perform the sequential scan of a chunk assigned to this thread.
///
/// Each thread sums its chunk of the brick_count buffer into shared memory.
/// Extracted as a helper to satisfy trap #4a (entry points must call a helper).
pub fn load_and_sum_chunk(
    local_id: u32,
    total_bricks: u32,
    brick_count: &[u32],
    chunk_size: u32,
) -> u32 {
    let start = local_id * chunk_size;
    let mut sum = 0u32;
    let mut i = 0u32;
    while i < chunk_size {
        let idx = start + i;
        if idx < total_bricks {
            sum += brick_count[idx as usize];
        }
        i += 1;
    }
    sum
}

/// Write the prefix sum results back to the brick_offset buffer.
///
/// Each thread writes the offsets for its chunk of elements.
/// The exclusive offset for each element is computed from the thread's
/// base offset (from the shared memory scan) plus a running sum within the chunk.
pub fn write_offsets(
    local_id: u32,
    total_bricks: u32,
    brick_count: &[u32],
    brick_offset: &mut [u32],
    base_offset: u32,
    chunk_size: u32,
) {
    let start = local_id * chunk_size;
    let mut running = base_offset;
    let mut i = 0u32;
    while i < chunk_size {
        let idx = start + i;
        if idx < total_bricks {
            brick_offset[idx as usize] = running;
            running += brick_count[idx as usize];
        }
        i += 1;
    }
    // Last thread writes the total count at brick_offset[total_bricks]
    if local_id == WORKGROUP_SIZE - 1 {
        brick_offset[total_bricks as usize] = running;
    }
}

/// Compute shader entry point: exclusive prefix sum over brick_count.
///
/// Uses a two-phase approach:
/// 1. Each thread sums its chunk of brick_count into shared memory
/// 2. Hillis-Steele inclusive scan in shared memory (O(n log n) work, simple)
/// 3. Each thread writes back per-element exclusive offsets
///
/// Descriptor set 0, binding 0: storage buffer of `u32` (brick_count, read).
/// Descriptor set 0, binding 1: storage buffer of `u32` (brick_offset, write, length = total_bricks + 1).
/// Push constants: `PrefixSumPushConstants`.
/// Dispatch with `(1, 1, 1)` — single workgroup.
#[spirv(compute(threads(256)))]
pub fn prefix_sum(
    #[spirv(global_invocation_id)] _id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(push_constant)] push: &PrefixSumPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] brick_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] brick_offset: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 512],
) {
    let lid = local_id.x;
    let total_bricks = push.total_bricks;
    let chunk_size = (total_bricks + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // Phase 1: each thread sums its chunk
    let my_sum = load_and_sum_chunk(lid, total_bricks, brick_count, chunk_size);

    // Store in shared memory for scan
    // We use double-buffering: shared[0..256] and shared[256..512]
    shared[lid as usize] = my_sum;

    // Phase 2: Hillis-Steele inclusive prefix sum in shared memory
    // After this, shared[lid] = sum of all thread sums from thread 0..=lid
    unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync(); }

    // Hillis-Steele: log2(256) = 8 iterations
    // Unrolled to avoid dynamic loop variable issues with shared memory
    // Each step: shared[i] += shared[i - stride] if i >= stride
    let mut stride = 1u32;
    let mut read_bank = 0u32;
    let mut write_bank = 256u32;

    // We need 8 iterations for 256 elements. Unrolling to avoid >3 branches per fn.
    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    stride = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    write_bank = 256 - write_bank;

    let _ = do_scan_step(lid, shared, stride, read_bank, write_bank);
    read_bank = 256 - read_bank;
    // After 8 iterations, the inclusive scan result is in the read_bank

    unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync(); }

    // Convert inclusive scan to exclusive: base_offset = inclusive[lid - 1], or 0 for lid==0
    let inclusive_sum = shared[(read_bank + lid) as usize];
    let base_offset = if lid == 0 { 0 } else { shared[(read_bank + lid - 1) as usize] };

    // Phase 3: write per-element offsets back to brick_offset
    write_offsets(lid, total_bricks, brick_count, brick_offset, base_offset, chunk_size);

    // Ensure the last thread also has the correct total — it's already handled
    // in write_offsets for the last thread. But we need to also handle the case
    // where total_bricks is not evenly divisible. The last thread's running sum
    // after its chunk is the total count. This is written by write_offsets.
    let _ = inclusive_sum; // suppress unused warning
}

/// One step of the Hillis-Steele scan with double-buffered shared memory.
///
/// Reads from `shared[read_bank + lid]`, writes to `shared[write_bank + lid]`.
/// Returns `stride * 2` for the next iteration.
pub fn do_scan_step(
    lid: u32,
    shared: &mut [u32; 512],
    stride: u32,
    read_bank: u32,
    write_bank: u32,
) -> u32 {
    unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync(); }

    let val = shared[(read_bank + lid) as usize];
    let sum = if lid >= stride {
        val + shared[(read_bank + lid - stride) as usize]
    } else {
        val
    };
    shared[(write_bank + lid) as usize] = sum;

    stride * 2
}
