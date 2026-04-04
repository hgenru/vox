//! GPU readback and buffer accessor methods for [`GpuSimulation`].

use ash::vk;
use gpu_core::{
    VulkanContext,
    buffer::{self, GpuBuffer},
};
use shared::Particle;

use crate::{Result, simulation::GpuSimulation};

impl GpuSimulation {
    /// Read particle data back from the GPU for debugging.
    ///
    /// This is a synchronous operation that stalls the GPU pipeline.
    /// Use only for debugging and testing.
    pub fn readback_particles(&self, ctx: &VulkanContext) -> Result<Vec<Particle>> {
        let particles =
            buffer::readback::<Particle>(ctx, &self.particle_buffer, self.num_particles as usize)?;
        Ok(particles)
    }

    /// Returns the current number of active particles.
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    /// Returns a reference to the particle buffer.
    pub fn particle_buffer(&self) -> &GpuBuffer {
        &self.particle_buffer
    }

    /// Returns a reference to the grid buffer.
    pub fn grid_buffer(&self) -> &GpuBuffer {
        &self.grid_buffer
    }

    /// Returns a reference to the voxel buffer.
    pub fn voxel_buffer(&self) -> &GpuBuffer {
        &self.voxel_buffer
    }

    /// Returns the Vulkan buffer handle for the render output.
    ///
    /// The buffer contains packed BGRA u32 pixels, sized `width * height`.
    pub fn render_output_buffer(&self) -> vk::Buffer {
        self.render_output_buffer.buffer
    }

    /// Returns a reference to the activity map buffer (for readback/debug).
    ///
    /// The buffer contains one `u32` per brick, where each entry holds the
    /// number of active particles in that brick after the last `step_physics`.
    pub fn activity_map_buffer(&self) -> &GpuBuffer {
        &self.activity_map_buffer
    }

    /// Returns a reference to the render output [`GpuBuffer`].
    ///
    /// Useful for readback operations (e.g., headless screenshot).
    /// The buffer contains packed BGRA u32 pixels, sized `width * height`.
    pub fn render_output_gpu_buffer(&self) -> &GpuBuffer {
        &self.render_output_buffer
    }

    /// Read the any_active flag from the GPU.
    ///
    /// Returns `true` if any brick has a non-zero tick_period (i.e., the
    /// simulation is not fully frozen). The CPU can use this to decide
    /// whether to skip rendering when nothing is changing.
    ///
    /// This is a synchronous readback operation; call after `step_physics()`
    /// completes and the GPU is idle.
    pub fn readback_any_active(&self, ctx: &VulkanContext) -> Result<bool> {
        let data = buffer::readback::<u32>(ctx, &self.any_active_buffer, 1)?;
        Ok(data[0] != 0)
    }

    /// Returns a reference to the any_active flag buffer.
    ///
    /// Contains a single `u32`: 0 = all bricks frozen, 1 = at least one active.
    /// Written by `update_sleep` via atomicMax, cleared to 0 before each dispatch.
    pub fn any_active_buffer(&self) -> &GpuBuffer {
        &self.any_active_buffer
    }

    /// Read the out-of-bounds flag from the GPU.
    ///
    /// Returns `true` if any particle was clamped back into the domain during
    /// the last `step_physics()`. The CPU can use this to trigger chunk
    /// streaming when particles approach the simulation boundary.
    ///
    /// This is a synchronous readback operation; call after `step_physics()`
    /// completes and the GPU is idle.
    pub fn readback_oob_flag(&self, ctx: &VulkanContext) -> Result<bool> {
        let data = buffer::readback::<u32>(ctx, &self.oob_flag_buffer, 1)?;
        Ok(data[0] != 0)
    }

    /// Returns a reference to the out-of-bounds flag buffer.
    ///
    /// Contains a single `u32`: 0 = all particles within margin, non-zero = at
    /// least one particle hit the domain boundary during G2P.
    pub fn oob_flag_buffer(&self) -> &GpuBuffer {
        &self.oob_flag_buffer
    }
}
