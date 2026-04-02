//! 3D image creation for voxel grid rendering.
//!
//! Provides [`GpuImage`] wrapping a Vulkan image with its view and allocation,
//! plus helpers for creating 3D images, image views, layout transitions, and samplers.

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::{context::VulkanContext, error::Result};

/// A GPU image with its associated view and memory allocation.
pub struct GpuImage {
    /// Vulkan image handle.
    pub image: vk::Image,
    /// Image view.
    pub view: vk::ImageView,
    /// Memory allocation backing this image.
    pub allocation: Option<Allocation>,
    /// Image format.
    pub format: vk::Format,
    /// Image extent.
    pub extent: vk::Extent3D,
}

/// Create a 3D image suitable for voxel grid storage.
///
/// The image is allocated on GPU-only memory. An image view is created
/// automatically.
pub fn create_3d_image(
    ctx: &VulkanContext,
    width: u32,
    height: u32,
    depth: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    name: &str,
) -> Result<GpuImage> {
    let extent = vk::Extent3D {
        width,
        height,
        depth,
    };

    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { ctx.device.create_image(&image_ci, None)? };
    let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };

    let allocation = {
        let mut allocator = ctx.allocator.lock().unwrap_or_else(|e| e.into_inner());
        allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?
    };

    unsafe {
        ctx.device
            .bind_image_memory(image, allocation.memory(), allocation.offset())?;
    }

    let view = create_image_view_3d(ctx, image, format)?;

    ctx.set_debug_name(image, name);
    ctx.set_debug_name(view, &format!("{}-view", name));

    tracing::debug!(
        "Created 3D image '{}' ({}x{}x{}, {:?})",
        name,
        width,
        height,
        depth,
        format
    );

    Ok(GpuImage {
        image,
        view,
        allocation: Some(allocation),
        format,
        extent,
    })
}

/// Create a 3D image view for the given image.
pub fn create_image_view_3d(
    ctx: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView> {
    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let view = unsafe { ctx.device.create_image_view(&view_ci, None)? };
    Ok(view)
}

/// Transition an image layout using a pipeline barrier.
///
/// Records a barrier command into the given command buffer.
/// The caller must ensure the command buffer is in the recording state.
pub fn transition_image_layout(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_access, src_stage, dst_access, dst_stage) =
        layout_transition_masks(old_layout, new_layout);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Determine access masks and pipeline stages for a layout transition.
fn layout_transition_masks(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> (
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::AccessFlags,
    vk::PipelineStageFlags,
) {
    let (src_access, src_stage) = match old_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::GENERAL => (
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => (
            vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    };

    let (dst_access, dst_stage) = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        vk::ImageLayout::GENERAL => (
            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        _ => (
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    };

    (src_access, src_stage, dst_access, dst_stage)
}

/// Create a sampler for ray marching / voxel grid sampling.
pub fn create_sampler(
    ctx: &VulkanContext,
    filter: vk::Filter,
    address_mode: vk::SamplerAddressMode,
    name: &str,
) -> Result<vk::Sampler> {
    let sampler_ci = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .address_mode_u(address_mode)
        .address_mode_v(address_mode)
        .address_mode_w(address_mode)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .min_lod(0.0)
        .max_lod(0.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false);

    let sampler = unsafe { ctx.device.create_sampler(&sampler_ci, None)? };

    ctx.set_debug_name(sampler, name);
    Ok(sampler)
}

/// Destroy a sampler.
pub fn destroy_sampler(ctx: &VulkanContext, sampler: vk::Sampler) {
    unsafe {
        ctx.device.destroy_sampler(sampler, None);
    }
}

/// Destroy a GPU image and free its memory allocation.
pub fn destroy_image(ctx: &VulkanContext, mut img: GpuImage) {
    unsafe {
        ctx.device.destroy_image_view(img.view, None);
        ctx.device.destroy_image(img.image, None);
    }
    if let Some(allocation) = img.allocation.take() {
        let mut allocator = ctx.allocator.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = allocator.free(allocation) {
            tracing::error!("Failed to free image allocation: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_destroy_3d_image() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let img = create_3d_image(
            &ctx,
            32,
            32,
            32,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "test-3d-image",
        )
        .expect("Failed to create 3D image");

        assert_ne!(img.image, vk::Image::null());
        assert_ne!(img.view, vk::ImageView::null());
        assert_eq!(img.extent.width, 32);
        assert_eq!(img.extent.height, 32);
        assert_eq!(img.extent.depth, 32);

        destroy_image(&ctx, img);
    }

    #[test]
    fn create_sampler_nearest() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let sampler = create_sampler(
            &ctx,
            vk::Filter::NEAREST,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            "test-sampler",
        )
        .expect("Failed to create sampler");

        assert_ne!(sampler, vk::Sampler::null());

        destroy_sampler(&ctx, sampler);
    }

    #[test]
    fn transition_image_layout_in_one_shot() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let img = create_3d_image(
            &ctx,
            8,
            8,
            8,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "test-transition",
        )
        .expect("Failed to create 3D image");

        ctx.execute_one_shot(|cmd| {
            transition_image_layout(
                &ctx,
                cmd,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
        })
        .expect("Layout transition failed");

        destroy_image(&ctx, img);
    }
}
