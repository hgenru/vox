//! GPU buffer creation, upload, and readback helpers.
//!
//! Provides [`GpuBuffer`] wrapping a Vulkan buffer with its gpu-allocator allocation,
//! plus helpers for creating device-local and staging buffers, and for uploading/reading
//! data via staging buffers.

use ash::vk;
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::{
    context::VulkanContext,
    error::{GpuError, Result},
};

/// A GPU buffer with its associated memory allocation.
///
/// Owns both the Vulkan buffer handle and the gpu-allocator [`Allocation`].
/// Destroyed on drop via the provided device and allocator references.
pub struct GpuBuffer {
    /// Vulkan buffer handle.
    pub buffer: vk::Buffer,
    /// Memory allocation backing this buffer.
    pub allocation: Option<Allocation>,
    /// Size in bytes.
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Returns the mapped pointer if this buffer has CPU-visible memory.
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        self.allocation.as_ref().and_then(|a| a.mapped_ptr())
    }

    /// Returns the mapped memory as a byte slice.
    pub fn mapped_slice(&self) -> Option<&[u8]> {
        self.allocation.as_ref().and_then(|a| a.mapped_slice())
    }

    /// Returns the mapped memory as a mutable byte slice.
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.allocation.as_mut().and_then(|a| a.mapped_slice_mut())
    }
}

/// Create a device-local buffer (GPU-only, not CPU-accessible).
///
/// Suitable for storage buffers, vertex buffers, index buffers, etc.
/// Data must be uploaded via a staging buffer.
pub fn create_device_local_buffer(
    ctx: &VulkanContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer> {
    create_buffer_internal(
        ctx,
        size,
        usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuOnly,
        name,
    )
}

/// Create a staging buffer for uploading data to the GPU (CPU→GPU).
///
/// Uses `CpuToGpu` memory location for optimal upload performance.
pub fn create_upload_staging_buffer(
    ctx: &VulkanContext,
    size: vk::DeviceSize,
    name: &str,
) -> Result<GpuBuffer> {
    create_buffer_internal(
        ctx,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        name,
    )
}

/// Create a staging buffer for reading data back from the GPU (GPU→CPU).
///
/// Uses `GpuToCpu` memory location for optimal readback performance.
pub fn create_readback_staging_buffer(
    ctx: &VulkanContext,
    size: vk::DeviceSize,
    name: &str,
) -> Result<GpuBuffer> {
    create_buffer_internal(
        ctx,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        name,
    )
}

/// Internal helper to create a buffer with the given memory location.
fn create_buffer_internal(
    ctx: &VulkanContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &str,
) -> Result<GpuBuffer> {
    let buffer_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { ctx.device.create_buffer(&buffer_ci, None)? };
    let requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };

    let allocation = {
        let mut allocator = ctx.allocator.lock().unwrap_or_else(|e| e.into_inner());
        allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?
    };

    unsafe {
        ctx.device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    ctx.set_debug_name(buffer, name);

    Ok(GpuBuffer {
        buffer,
        allocation: Some(allocation),
        size,
    })
}

/// Upload data from CPU to a device-local GPU buffer via a staging buffer.
///
/// Creates a temporary staging buffer, copies `data` into it, then issues
/// a GPU copy command from staging to `dst`.
pub fn upload<T: Pod>(ctx: &VulkanContext, data: &[T], dst: &GpuBuffer) -> Result<()> {
    let byte_size = std::mem::size_of_val(data) as vk::DeviceSize;
    let mut staging = create_upload_staging_buffer(ctx, byte_size, "upload-staging")?;

    // Copy data into staging buffer
    {
        let mapped = staging.mapped_slice_mut().ok_or(GpuError::MappingFailed)?;
        let src_bytes = bytemuck::cast_slice::<T, u8>(data);
        mapped[..src_bytes.len()].copy_from_slice(src_bytes);
    }

    // GPU copy: staging -> device
    ctx.execute_one_shot(|cmd| {
        let region = vk::BufferCopy::default().size(byte_size);
        unsafe {
            ctx.device
                .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
        }
    })?;

    destroy_buffer(ctx, staging);
    Ok(())
}

/// Read data back from a device-local GPU buffer to CPU memory.
///
/// Creates a temporary staging buffer, copies from `src` on GPU, maps the
/// staging buffer, and returns the data as a `Vec<T>`.
pub fn readback<T: Pod>(ctx: &VulkanContext, src: &GpuBuffer, count: usize) -> Result<Vec<T>> {
    let byte_size = (count * std::mem::size_of::<T>()) as vk::DeviceSize;
    let staging = create_readback_staging_buffer(ctx, byte_size, "readback-staging")?;

    // GPU copy: device -> staging
    ctx.execute_one_shot(|cmd| {
        let region = vk::BufferCopy::default().size(byte_size);
        unsafe {
            ctx.device
                .cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[region]);
        }
    })?;

    // Read from mapped staging buffer
    let result = {
        let mapped = staging.mapped_slice().ok_or(GpuError::MappingFailed)?;
        let typed: &[T] = bytemuck::cast_slice(&mapped[..byte_size as usize]);
        typed.to_vec()
    };

    destroy_buffer(ctx, staging);
    Ok(result)
}

/// Size in bytes of a `VkDispatchIndirectCommand` (3 x `u32`), padded to 16 bytes
/// for safe alignment when used as a storage buffer.
pub const DISPATCH_INDIRECT_SIZE: vk::DeviceSize = 16;

/// Create a device-local indirect dispatch buffer.
///
/// The buffer is sized for a single `VkDispatchIndirectCommand` (3 x `u32`, 16 bytes
/// with padding) and has `STORAGE_BUFFER | INDIRECT_BUFFER` usage flags so it can be
/// written by a compute shader and consumed by `vkCmdDispatchIndirect`.
pub fn create_indirect_buffer(ctx: &VulkanContext, name: &str) -> Result<GpuBuffer> {
    create_device_local_buffer(
        ctx,
        DISPATCH_INDIRECT_SIZE,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
        name,
    )
}

/// Destroy a GPU buffer and free its memory allocation.
pub fn destroy_buffer(ctx: &VulkanContext, mut buf: GpuBuffer) {
    unsafe {
        ctx.device.destroy_buffer(buf.buffer, None);
    }
    if let Some(allocation) = buf.allocation.take() {
        let mut allocator = ctx.allocator.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = allocator.free(allocation) {
            tracing::error!("Failed to free buffer allocation: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_device_local_and_staging() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let device_buf = create_device_local_buffer(
            &ctx,
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "test-device-buf",
        )
        .expect("Failed to create device-local buffer");
        assert_ne!(device_buf.buffer, vk::Buffer::null());
        assert_eq!(device_buf.size, 1024);

        let staging_buf = create_upload_staging_buffer(&ctx, 1024, "test-staging-buf")
            .expect("Failed to create staging buffer");
        assert_ne!(staging_buf.buffer, vk::Buffer::null());
        assert!(staging_buf.mapped_ptr().is_some());

        destroy_buffer(&ctx, device_buf);
        destroy_buffer(&ctx, staging_buf);
    }

    #[test]
    fn create_indirect_buffer_works() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let buf = create_indirect_buffer(&ctx, "test-indirect-buf")
            .expect("Failed to create indirect buffer");
        assert_ne!(buf.buffer, vk::Buffer::null());
        assert_eq!(buf.size, DISPATCH_INDIRECT_SIZE);

        destroy_buffer(&ctx, buf);
    }

    #[test]
    fn upload_and_readback_round_trip() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let data: Vec<u32> = (0..64).collect();
        let buf = create_device_local_buffer(
            &ctx,
            (data.len() * std::mem::size_of::<u32>()) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "test-roundtrip",
        )
        .expect("Failed to create buffer");

        upload(&ctx, &data, &buf).expect("Upload failed");
        let result: Vec<u32> = readback(&ctx, &buf, data.len()).expect("Readback failed");

        assert_eq!(data, result);

        destroy_buffer(&ctx, buf);
    }
}
