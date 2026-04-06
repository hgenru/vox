//! VRAM budget system: query available GPU memory and compute resource limits.
//!
//! After creating a [`VulkanContext`], call [`query_vram_mb`] to get the total
//! device-local VRAM in megabytes, then [`VramBudget::from_available_vram`] to
//! compute recommended resource limits (particles, grid slots, render resolution).

use ash::vk;

/// Resource budget derived from available VRAM.
///
/// Use [`VramBudget::from_available_vram`] to compute recommended limits for
/// particle count, hash grid capacity, and render resolution based on the GPU's
/// available device-local memory.
#[derive(Debug, Clone, Copy)]
pub struct VramBudget {
    /// Total device-local VRAM in megabytes.
    pub total_vram_mb: u32,
    /// Maximum number of particles to allocate.
    pub max_particles: u32,
    /// Hash grid capacity (number of grid slots for sparse grid).
    pub hash_grid_capacity: u32,
    /// Render target width in pixels.
    pub render_width: u32,
    /// Render target height in pixels.
    pub render_height: u32,
    /// Number of chunk pool slots for the gigabuffer allocator.
    pub chunk_pool_slots: u32,
}

impl VramBudget {
    /// Compute a resource budget from the available VRAM in megabytes.
    ///
    /// Tiers:
    /// - 6 GB:  200K particles, 512K grid slots, 960x540
    /// - 8 GB:  500K particles, 1M grid slots,   1280x720
    /// - 12 GB: 1M particles,   2M grid slots,   1280x720
    /// - 24 GB: 2M particles,   4M grid slots,   1920x1080
    pub fn from_available_vram(vram_mb: u32) -> Self {
        let (max_particles, hash_grid_capacity, render_width, render_height, chunk_pool_slots) =
            budget_for_tier(vram_mb);

        Self {
            total_vram_mb: vram_mb,
            max_particles,
            hash_grid_capacity,
            render_width,
            render_height,
            chunk_pool_slots,
        }
    }
}

/// Select budget parameters based on VRAM tier.
///
/// Separated from [`VramBudget::from_available_vram`] to keep branch count
/// manageable (trap #15) and satisfy trap #4a for testability.
fn budget_for_tier(vram_mb: u32) -> (u32, u32, u32, u32, u32) {
    if vram_mb >= 20_000 {
        // 24 GB+ tier (RTX 4090, A6000, etc.)
        // 2048 chunk slots ~= 264 MB gigabuffer
        return (2_000_000, 4_194_304, 1920, 1080, 2048);
    }
    budget_for_lower_tier(vram_mb)
}

/// Select budget for VRAM below 20 GB.
///
/// Split helper to avoid >3 if/else branches per function (trap #15).
fn budget_for_lower_tier(vram_mb: u32) -> (u32, u32, u32, u32, u32) {
    if vram_mb >= 10_000 {
        // 12 GB tier (RTX 4070 Ti, etc.)
        // 1536 chunk slots ~= 198 MB gigabuffer
        (1_000_000, 2_097_152, 1280, 720, 1536)
    } else if vram_mb >= 7_000 {
        // 8 GB tier (RTX 4060, etc.)
        // 1024 chunk slots ~= 132 MB gigabuffer
        (500_000, 1_048_576, 1280, 720, 1024)
    } else {
        // 6 GB or less
        // 512 chunk slots ~= 66 MB gigabuffer
        (200_000, 524_288, 960, 540, 512)
    }
}

/// Query total device-local VRAM in megabytes from the physical device memory properties.
///
/// Scans all memory heaps and returns the size of the largest heap with the
/// `DEVICE_LOCAL` flag set. Returns 0 if no device-local heap is found.
pub fn query_vram_mb(memory_properties: &vk::PhysicalDeviceMemoryProperties) -> u32 {
    let mut max_device_local_bytes: u64 = 0;

    let heap_count = memory_properties.memory_heap_count as usize;
    let mut i = 0;
    while i < heap_count {
        let heap = &memory_properties.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) && heap.size > max_device_local_bytes {
            max_device_local_bytes = heap.size;
        }
        i += 1;
    }

    (max_device_local_bytes / (1024 * 1024)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_6gb() {
        let budget = VramBudget::from_available_vram(6_000);
        assert_eq!(budget.max_particles, 200_000);
        assert_eq!(budget.hash_grid_capacity, 524_288);
        assert_eq!(budget.render_width, 960);
        assert_eq!(budget.render_height, 540);
        assert_eq!(budget.chunk_pool_slots, 512);
    }

    #[test]
    fn budget_8gb() {
        let budget = VramBudget::from_available_vram(8_000);
        assert_eq!(budget.max_particles, 500_000);
        assert_eq!(budget.hash_grid_capacity, 1_048_576);
        assert_eq!(budget.render_width, 1280);
        assert_eq!(budget.render_height, 720);
        assert_eq!(budget.chunk_pool_slots, 1024);
    }

    #[test]
    fn budget_12gb() {
        let budget = VramBudget::from_available_vram(12_000);
        assert_eq!(budget.max_particles, 1_000_000);
        assert_eq!(budget.hash_grid_capacity, 2_097_152);
        assert_eq!(budget.render_width, 1280);
        assert_eq!(budget.render_height, 720);
        assert_eq!(budget.chunk_pool_slots, 1536);
    }

    #[test]
    fn budget_24gb() {
        let budget = VramBudget::from_available_vram(24_000);
        assert_eq!(budget.max_particles, 2_000_000);
        assert_eq!(budget.hash_grid_capacity, 4_194_304);
        assert_eq!(budget.render_width, 1920);
        assert_eq!(budget.render_height, 1080);
        assert_eq!(budget.chunk_pool_slots, 2048);
    }

    #[test]
    fn budget_4gb_fallback() {
        let budget = VramBudget::from_available_vram(4_000);
        assert_eq!(budget.max_particles, 200_000);
        assert_eq!(budget.render_width, 960);
        assert_eq!(budget.chunk_pool_slots, 512);
    }

    #[test]
    fn query_vram_from_context() {
        let ctx = crate::VulkanContext::new().expect("Failed to create VulkanContext");
        let vram_mb = query_vram_mb(&ctx.memory_properties);
        // Any real GPU should have at least 1 GB of VRAM
        assert!(vram_mb >= 1_000, "Expected at least 1 GB VRAM, got {} MB", vram_mb);

        let budget = VramBudget::from_available_vram(vram_mb);
        assert!(budget.max_particles > 0);
        assert!(budget.render_width > 0);
        assert!(budget.render_height > 0);
    }
}
