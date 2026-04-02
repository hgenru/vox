//! Swapchain creation, recreation on resize, acquire/present.
//!
//! Uses `VK_KHR_dynamic_rendering` — no render pass objects needed.

use ash::vk;
use std::ffi::CStr;

use crate::context::VulkanContext;
use crate::error::Result;

/// Swapchain wrapper with image views and format info.
pub struct Swapchain {
    /// The swapchain loader.
    swapchain_loader: ash::khr::swapchain::Device,
    /// The swapchain handle.
    pub handle: vk::SwapchainKHR,
    /// Swapchain images (owned by the swapchain, not manually destroyed).
    pub images: Vec<vk::Image>,
    /// Image views for each swapchain image.
    pub image_views: Vec<vk::ImageView>,
    /// Swapchain image format.
    pub format: vk::SurfaceFormatKHR,
    /// Swapchain extent.
    pub extent: vk::Extent2D,
}

/// Required instance extensions for windowed/surface operation.
pub const SURFACE_INSTANCE_EXTENSIONS: &[&CStr] = &[
    ash::khr::surface::NAME,
    // On Windows:
    ash::khr::win32_surface::NAME,
];

/// Required device extension for swapchain.
pub const SWAPCHAIN_DEVICE_EXTENSION: &CStr = ash::khr::swapchain::NAME;

impl Swapchain {
    /// Create a new swapchain for the given surface.
    ///
    /// The surface must have been created by the caller (e.g., via ash-window).
    /// Uses FIFO present mode (vsync) as fallback, prefers MAILBOX.
    pub fn new(
        ctx: &VulkanContext,
        surface_loader: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let surface_caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(
                ctx.physical_device,
                surface,
            )?
        };

        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(
                ctx.physical_device,
                surface,
            )?
        };

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(
                ctx.physical_device,
                surface,
            )?
        };

        // Choose format: prefer B8G8R8A8_SRGB
        let format = surface_formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(surface_formats[0]);

        // Choose present mode: prefer MAILBOX, fallback FIFO
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        // Choose extent
        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        // Image count: at least min + 1, but not exceeding max (if max > 0)
        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0
            && image_count > surface_caps.max_image_count
        {
            image_count = surface_caps.max_image_count;
        }

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let handle = unsafe {
            swapchain_loader.create_swapchain(&swapchain_ci, None)?
        };

        let images = unsafe { swapchain_loader.get_swapchain_images(handle)? };

        let image_views: std::result::Result<Vec<_>, _> = images
            .iter()
            .enumerate()
            .map(|(i, &image)| {
                let view_ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                let view = unsafe { ctx.device.create_image_view(&view_ci, None) };
                if let Ok(ref v) = view {
                    ctx.set_debug_name(*v, &format!("swapchain-view-{}", i));
                }
                view
            })
            .collect();
        let image_views = image_views?;

        tracing::info!(
            "Swapchain created: {}x{}, {:?}, {} images, {:?}",
            extent.width,
            extent.height,
            format.format,
            images.len(),
            present_mode
        );

        Ok(Self {
            swapchain_loader,
            handle,
            images,
            image_views,
            format,
            extent,
        })
    }

    /// Acquire the next swapchain image.
    ///
    /// Returns the image index and whether the swapchain is suboptimal.
    pub fn acquire_next_image(
        &self,
        semaphore: vk::Semaphore,
        timeout_ns: u64,
    ) -> std::result::Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.handle,
                timeout_ns,
                semaphore,
                vk::Fence::null(),
            )
        }
    }

    /// Present the given image index.
    ///
    /// Returns whether the swapchain is suboptimal.
    pub fn present(
        &self,
        queue: vk::Queue,
        wait_semaphore: vk::Semaphore,
        image_index: u32,
    ) -> std::result::Result<bool, vk::Result> {
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&wait_semaphore))
            .swapchains(std::slice::from_ref(&self.handle))
            .image_indices(std::slice::from_ref(&image_index));

        unsafe {
            self.swapchain_loader
                .queue_present(queue, &present_info)
        }
    }

    /// Destroy swapchain resources. Must be called before dropping.
    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            for &view in &self.image_views {
                ctx.device.destroy_image_view(view, None);
            }
            self.image_views.clear();
            self.images.clear();

            self.swapchain_loader
                .destroy_swapchain(self.handle, None);
        }
    }
}
