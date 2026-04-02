//! Frame management with double buffering (2 frames in flight).
//!
//! [`FrameManager`] manages per-frame synchronization primitives (fences,
//! semaphores) and command buffers. Uses `VK_KHR_dynamic_rendering` — no
//! render passes needed.

use ash::vk;
use shared::FRAMES_IN_FLIGHT;

use crate::{
    context::VulkanContext,
    error::{GpuError, Result},
    swapchain::Swapchain,
};

/// Per-frame synchronization and command recording state.
pub struct PerFrameData {
    /// Fence signaled when the GPU finishes this frame's work.
    pub fence: vk::Fence,
    /// Semaphore signaled when a swapchain image is available.
    pub image_available: vk::Semaphore,
    /// Semaphore signaled when rendering is finished (before present).
    pub render_finished: vk::Semaphore,
    /// Command buffer for this frame.
    pub command_buffer: vk::CommandBuffer,
}

/// Active frame context returned by [`FrameManager::begin_frame`].
///
/// Contains everything needed to record commands for this frame.
pub struct FrameContext {
    /// Index into the frames-in-flight ring (0 or 1).
    pub frame_index: usize,
    /// Swapchain image index acquired for this frame.
    pub image_index: u32,
    /// Command buffer (already in recording state).
    pub command_buffer: vk::CommandBuffer,
}

/// Manages per-frame resources for double-buffered rendering.
///
/// Owns fences, semaphores, and command buffers for each frame in flight.
pub struct FrameManager {
    /// Per-frame data (fences, semaphores, command buffers).
    pub frames: Vec<PerFrameData>,
    /// Command pool for frame command buffers (separate from context's pool).
    pub command_pool: vk::CommandPool,
    /// Current frame index (alternates 0, 1, 0, 1, ...).
    current_frame: usize,
}

impl FrameManager {
    /// Create a new frame manager with per-frame sync objects and command buffers.
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let n = FRAMES_IN_FLIGHT as usize;

        // Create a dedicated command pool for frame rendering
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_families.graphics)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_ci, None)? };
        ctx.set_debug_name(command_pool, "frame-cmd-pool");

        // Allocate command buffers
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(n as u32);
        let command_buffers = unsafe { ctx.device.allocate_command_buffers(&cmd_alloc)? };

        // Create per-frame sync objects
        let mut frames = Vec::with_capacity(n);
        for i in 0..n {
            let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = unsafe { ctx.device.create_fence(&fence_ci, None)? };

            let sem_ci = vk::SemaphoreCreateInfo::default();
            let image_available = unsafe { ctx.device.create_semaphore(&sem_ci, None)? };
            let render_finished = unsafe { ctx.device.create_semaphore(&sem_ci, None)? };

            ctx.set_debug_name(fence, &format!("frame-fence-{}", i));
            ctx.set_debug_name(image_available, &format!("image-available-{}", i));
            ctx.set_debug_name(render_finished, &format!("render-finished-{}", i));
            ctx.set_debug_name(command_buffers[i], &format!("frame-cmd-{}", i));

            frames.push(PerFrameData {
                fence,
                image_available,
                render_finished,
                command_buffer: command_buffers[i],
            });
        }

        tracing::info!("FrameManager created with {} frames in flight", n);

        Ok(Self {
            frames,
            command_pool,
            current_frame: 0,
        })
    }

    /// Begin a new frame: wait for the fence, acquire a swapchain image,
    /// reset and begin the command buffer.
    ///
    /// Returns `None` if the swapchain needs recreation (suboptimal or out-of-date).
    pub fn begin_frame(
        &mut self,
        ctx: &VulkanContext,
        swapchain: &Swapchain,
    ) -> Result<Option<FrameContext>> {
        let frame = &self.frames[self.current_frame];

        // Wait for this frame's previous work to finish
        unsafe {
            ctx.device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
        }

        // Acquire next swapchain image
        let (image_index, suboptimal) =
            match swapchain.acquire_next_image(frame.image_available, u64::MAX) {
                Ok((idx, sub)) => (idx, sub),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(None),
                Err(e) => return Err(GpuError::Vulkan(e)),
            };

        if suboptimal {
            return Ok(None);
        }

        // Reset fence only after we know we'll submit work
        unsafe {
            ctx.device.reset_fences(&[frame.fence])?;
        }

        // Reset and begin command buffer
        unsafe {
            ctx.device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        Ok(Some(FrameContext {
            frame_index: self.current_frame,
            image_index,
            command_buffer: frame.command_buffer,
        }))
    }

    /// End a frame: end the command buffer, submit with proper synchronization,
    /// and present.
    ///
    /// Returns `false` if the swapchain needs recreation.
    pub fn end_frame(
        &mut self,
        ctx: &VulkanContext,
        swapchain: &Swapchain,
        frame_ctx: FrameContext,
    ) -> Result<bool> {
        let frame = &self.frames[frame_ctx.frame_index];

        unsafe {
            ctx.device.end_command_buffer(frame.command_buffer)?;
        }

        // Submit
        let wait_semaphores = [frame.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished];
        let command_buffers = [frame.command_buffer];

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            ctx.device
                .queue_submit(ctx.graphics_queue, &[submit], frame.fence)?;
        }

        // Present
        let present_ok = match swapchain.present(
            ctx.graphics_queue,
            frame.render_finished,
            frame_ctx.image_index,
        ) {
            Ok(suboptimal) => !suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => false,
            Err(e) => return Err(GpuError::Vulkan(e)),
        };

        // Advance frame index
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT as usize;

        Ok(present_ok)
    }

    /// Destroy all frame resources.
    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            let _ = ctx.device.device_wait_idle();

            for frame in &self.frames {
                ctx.device.destroy_fence(frame.fence, None);
                ctx.device.destroy_semaphore(frame.image_available, None);
                ctx.device.destroy_semaphore(frame.render_finished, None);
            }

            ctx.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_frame_manager() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let fm = FrameManager::new(&ctx).expect("Failed to create FrameManager");

        assert_eq!(fm.frames.len(), FRAMES_IN_FLIGHT as usize);
        for frame in &fm.frames {
            assert_ne!(frame.fence, vk::Fence::null());
            assert_ne!(frame.image_available, vk::Semaphore::null());
            assert_ne!(frame.render_finished, vk::Semaphore::null());
            assert_ne!(frame.command_buffer, vk::CommandBuffer::null());
        }

        fm.destroy(&ctx);
    }
}
