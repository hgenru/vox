//! Renderer: owns Vulkan surface, swapchain, and frame manager.
//!
//! For the MVP, clears the swapchain image with a color that cycles
//! based on the frame number (proof of life). Actual voxel ray tracing
//! rendering will be added in a later iteration.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use gpu_core::context::VulkanContext;
use gpu_core::frame::FrameManager;
use gpu_core::swapchain::{Swapchain, SURFACE_INSTANCE_EXTENSIONS};

/// Errors that can occur in the renderer.
#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    /// GPU-core error.
    #[error("GPU error: {0}")]
    Gpu(#[from] gpu_core::error::GpuError),

    /// Window handle error.
    #[error("Window handle error: {0}")]
    HandleError(#[from] raw_window_handle::HandleError),

    /// Vulkan result error (from ash-window surface creation).
    #[error("Vulkan error: {0}")]
    Vulkan(vk::Result),
}

/// Result type alias for renderer operations.
pub type Result<T> = std::result::Result<T, RendererError>;

/// Returns the instance extensions required for windowed rendering.
///
/// Pass these to `VulkanContext::new_with_instance_extensions`.
pub fn required_instance_extensions() -> &'static [&'static std::ffi::CStr] {
    SURFACE_INSTANCE_EXTENSIONS
}

/// The renderer owns the Vulkan surface, swapchain, and frame manager.
///
/// Create via [`Renderer::new`] after creating a `VulkanContext` with
/// the extensions from [`required_instance_extensions`].
pub struct Renderer {
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    swapchain: Swapchain,
    frame_manager: FrameManager,
    frame_number: u64,
    needs_resize: bool,
    width: u32,
    height: u32,
}

impl Renderer {
    /// Create a new renderer attached to the given window.
    ///
    /// The `VulkanContext` must have been created with the surface instance
    /// extensions (see [`required_instance_extensions`]).
    pub fn new(ctx: &VulkanContext, window: &Window) -> Result<Self> {
        // Create surface via ash-window
        let surface = unsafe {
            ash_window::create_surface(
                &ctx.entry,
                &ctx.instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .map_err(RendererError::Vulkan)?
        };

        let surface_loader =
            ash::khr::surface::Instance::new(&ctx.entry, &ctx.instance);

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let swapchain =
            Swapchain::new(ctx, &surface_loader, surface, width, height)?;

        let frame_manager = FrameManager::new(ctx)?;

        tracing::info!(
            "Renderer initialized: {}x{}, {} swapchain images",
            width,
            height,
            swapchain.images.len()
        );

        Ok(Self {
            surface,
            surface_loader,
            swapchain,
            frame_manager,
            frame_number: 0,
            needs_resize: false,
            width,
            height,
        })
    }

    /// Mark the renderer for swapchain recreation on the next frame.
    ///
    /// Call this when the window is resized.
    pub fn notify_resized(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.needs_resize = true;
        tracing::debug!("Resize requested: {}x{}", self.width, self.height);
    }

    /// Render one frame: acquire image, clear with a cycling color, present.
    ///
    /// Returns `Ok(true)` on success, `Ok(false)` if the frame was skipped
    /// (e.g., minimized window), or an error.
    pub fn draw_frame(&mut self, ctx: &VulkanContext) -> Result<bool> {
        if self.needs_resize {
            self.recreate_swapchain(ctx)?;
            self.needs_resize = false;
        }

        // Skip drawing if window is minimized (zero extent)
        if self.width == 0 || self.height == 0 {
            return Ok(false);
        }

        // Begin frame (acquire swapchain image)
        let frame_ctx = match self.frame_manager.begin_frame(ctx, &self.swapchain)? {
            Some(fc) => fc,
            None => {
                // Swapchain out of date, recreate
                self.recreate_swapchain(ctx)?;
                return Ok(false);
            }
        };

        let cmd = frame_ctx.command_buffer;
        let image = self.swapchain.images[frame_ctx.image_index as usize];

        // Transition image: UNDEFINED -> TRANSFER_DST_OPTIMAL
        let barrier_to_clear = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(full_color_range());

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_clear],
            );
        }

        // Clear with a cycling color (proof of life)
        let t = self.frame_number as f32 * 0.01;
        let clear_color = vk::ClearColorValue {
            float32: [
                (t.sin() * 0.5 + 0.5) * 0.2,
                (t * 0.7).sin() * 0.5 + 0.5,
                ((t * 1.3).sin() * 0.5 + 0.5) * 0.8,
                1.0,
            ],
        };

        unsafe {
            ctx.device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[full_color_range()],
            );
        }

        // Transition image: TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR
        let barrier_to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(full_color_range());

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_present],
            );
        }

        // End frame (submit + present)
        let present_ok =
            self.frame_manager.end_frame(ctx, &self.swapchain, frame_ctx)?;

        if !present_ok {
            self.needs_resize = true;
        }

        self.frame_number += 1;
        Ok(true)
    }

    /// Recreate the swapchain (e.g., after a window resize).
    fn recreate_swapchain(&mut self, ctx: &VulkanContext) -> Result<()> {
        unsafe {
            ctx.device.device_wait_idle().map_err(gpu_core::GpuError::Vulkan)?;
        }

        self.swapchain.destroy(ctx);

        self.swapchain = Swapchain::new(
            ctx,
            &self.surface_loader,
            self.surface,
            self.width,
            self.height,
        )?;

        tracing::info!(
            "Swapchain recreated: {}x{}",
            self.swapchain.extent.width,
            self.swapchain.extent.height
        );

        Ok(())
    }

    /// Destroy all renderer resources. Must be called before the `VulkanContext`
    /// is dropped.
    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            let _ = ctx.device.device_wait_idle();
        }

        self.frame_manager.destroy(ctx);
        self.swapchain.destroy(ctx);

        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }

        tracing::info!("Renderer destroyed");
    }

    /// Get the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }
}

/// Helper: full color subresource range (mip 0, layer 0).
fn full_color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
