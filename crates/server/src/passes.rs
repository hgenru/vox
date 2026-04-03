//! Compute pass abstraction — pipeline + descriptor set + layout.
//!
//! [`ComputePass`] bundles the Vulkan objects needed to record a single
//! compute dispatch.  [`GpuSimulation::create_pass`] and the dispatch /
//! barrier helpers live here so the main simulation module stays focused on
//! orchestration logic.

use std::ffi::CStr;

use ash::vk;
use gpu_core::{
    VulkanContext,
    buffer::GpuBuffer,
    pipeline,
};

use crate::Result;

/// A single compute pipeline with its layout, descriptor set layout, and descriptor set.
pub(crate) struct ComputePass {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

/// Create a single compute pass: descriptor set layout, pipeline layout,
/// pipeline, and descriptor set with buffer bindings.
pub(crate) fn create_pass(
    ctx: &VulkanContext,
    shader_module: vk::ShaderModule,
    entry_point: &CStr,
    buffers: &[&GpuBuffer],
    push_constant_size: u32,
    descriptor_pool: vk::DescriptorPool,
    name: &str,
) -> Result<ComputePass> {
    // Descriptor bindings: one STORAGE_BUFFER per buffer
    let bindings: Vec<pipeline::DescriptorBinding> = (0..buffers.len() as u32)
        .map(|i| pipeline::DescriptorBinding {
            binding: i,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        })
        .collect();

    let ds_layout_name = format!("{name}-ds-layout");
    let descriptor_set_layout =
        pipeline::create_descriptor_set_layout(ctx, &bindings, &ds_layout_name)?;

    let pipe_layout_name = format!("{name}-pipe-layout");
    let pipeline_layout = pipeline::create_pipeline_layout(
        ctx,
        &[descriptor_set_layout],
        push_constant_size,
        &pipe_layout_name,
    )?;

    let pipe_name = format!("{name}-pipeline");
    let compute_pipeline = pipeline::create_compute_pipeline(
        ctx,
        shader_module,
        entry_point,
        pipeline_layout,
        &pipe_name,
    )?;

    // Allocate and update descriptor set
    let sets =
        pipeline::allocate_descriptor_sets(ctx, descriptor_pool, &[descriptor_set_layout])?;
    let descriptor_set = sets[0];

    let buffer_bindings: Vec<pipeline::BufferBinding<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| pipeline::BufferBinding {
            set: descriptor_set,
            binding: i as u32,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            buffer: buf,
        })
        .collect();
    pipeline::update_descriptor_sets(ctx, &buffer_bindings);

    Ok(ComputePass {
        pipeline: compute_pipeline,
        pipeline_layout,
        descriptor_set_layout,
        descriptor_set,
    })
}

/// Record a compute dispatch for a single pass.
pub(crate) fn dispatch(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pass: &ComputePass,
    group_x: u32,
    group_y: u32,
    group_z: u32,
    push_constants: &[u8],
) {
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pass.pipeline_layout,
            0,
            &[pass.descriptor_set],
            &[],
        );
        if !push_constants.is_empty() {
            device.cmd_push_constants(
                cmd,
                pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
        device.cmd_dispatch(cmd, group_x, group_y, group_z);
    }
}

/// Insert a compute-to-compute memory barrier.
pub(crate) fn barrier(cmd: vk::CommandBuffer, device: &ash::Device) {
    let memory_barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }
}
