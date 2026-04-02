//! Compute pipeline creation from SPIR-V, descriptor set management.
//!
//! Provides helpers for creating shader modules, descriptor set layouts,
//! pipeline layouts, compute pipelines, descriptor pools, and updating
//! descriptor sets with buffer bindings.

use std::ffi::CStr;

use ash::vk;

use crate::{
    context::VulkanContext,
    error::{GpuError, Result},
};

/// Description of a single descriptor binding.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorBinding {
    /// Binding index in the shader.
    pub binding: u32,
    /// Type of descriptor (storage buffer, uniform buffer, etc.).
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors at this binding (usually 1).
    pub count: u32,
    /// Shader stages that access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

/// Create a shader module from SPIR-V bytes.
///
/// The `spv_bytes` must be valid SPIR-V with length that is a multiple of 4.
pub fn create_shader_module(
    ctx: &VulkanContext,
    spv_bytes: &[u8],
    name: &str,
) -> Result<vk::ShaderModule> {
    if spv_bytes.len() % 4 != 0 {
        return Err(GpuError::InvalidSpirv(spv_bytes.len()));
    }

    let spv_words: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spv_words);
    let module = unsafe { ctx.device.create_shader_module(&shader_ci, None)? };

    ctx.set_debug_name(module, name);
    tracing::debug!(
        "Created shader module '{}' ({} bytes)",
        name,
        spv_bytes.len()
    );

    Ok(module)
}

/// Create a descriptor set layout from a list of bindings.
pub fn create_descriptor_set_layout(
    ctx: &VulkanContext,
    bindings: &[DescriptorBinding],
    name: &str,
) -> Result<vk::DescriptorSetLayout> {
    let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
        .iter()
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b.binding)
                .descriptor_type(b.descriptor_type)
                .descriptor_count(b.count)
                .stage_flags(b.stage_flags)
        })
        .collect();

    let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
    let layout = unsafe { ctx.device.create_descriptor_set_layout(&layout_ci, None)? };

    ctx.set_debug_name(layout, name);
    Ok(layout)
}

/// Create a pipeline layout with a single descriptor set layout and optional push constants.
pub fn create_pipeline_layout(
    ctx: &VulkanContext,
    desc_layouts: &[vk::DescriptorSetLayout],
    push_constant_size: u32,
    name: &str,
) -> Result<vk::PipelineLayout> {
    let mut push_constant_ranges = Vec::new();
    if push_constant_size > 0 {
        push_constant_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_size),
        );
    }

    let layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(desc_layouts)
        .push_constant_ranges(&push_constant_ranges);

    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_ci, None)? };

    ctx.set_debug_name(layout, name);
    Ok(layout)
}

/// Create a compute pipeline from a shader module and entry point.
pub fn create_compute_pipeline(
    ctx: &VulkanContext,
    shader_module: vk::ShaderModule,
    entry_point: &CStr,
    pipeline_layout: vk::PipelineLayout,
    name: &str,
) -> Result<vk::Pipeline> {
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_point);

    let pipeline_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let pipeline = unsafe {
        ctx.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            .map_err(|e| e.1)?[0]
    };

    ctx.set_debug_name(pipeline, name);
    tracing::debug!("Created compute pipeline '{}'", name);

    Ok(pipeline)
}

/// Create a descriptor pool that can allocate the specified descriptor types.
pub fn create_descriptor_pool(
    ctx: &VulkanContext,
    pool_sizes: &[vk::DescriptorPoolSize],
    max_sets: u32,
    name: &str,
) -> Result<vk::DescriptorPool> {
    let pool_ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(pool_sizes);

    let pool = unsafe { ctx.device.create_descriptor_pool(&pool_ci, None)? };

    ctx.set_debug_name(pool, name);
    Ok(pool)
}

/// Allocate descriptor sets from a pool using the given layouts.
pub fn allocate_descriptor_sets(
    ctx: &VulkanContext,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>> {
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);

    let sets = unsafe { ctx.device.allocate_descriptor_sets(&alloc_info)? };
    Ok(sets)
}

/// A buffer binding for updating a descriptor set.
pub struct BufferBinding<'a> {
    /// The descriptor set to update.
    pub set: vk::DescriptorSet,
    /// Binding index.
    pub binding: u32,
    /// Descriptor type.
    pub descriptor_type: vk::DescriptorType,
    /// Buffer to bind.
    pub buffer: &'a crate::buffer::GpuBuffer,
}

/// Update descriptor sets with buffer bindings.
pub fn update_descriptor_sets(ctx: &VulkanContext, bindings: &[BufferBinding<'_>]) {
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = bindings
        .iter()
        .map(|b| {
            vk::DescriptorBufferInfo::default()
                .buffer(b.buffer.buffer)
                .offset(0)
                .range(b.buffer.size)
        })
        .collect();

    let writes: Vec<vk::WriteDescriptorSet> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| {
            vk::WriteDescriptorSet::default()
                .dst_set(b.set)
                .dst_binding(b.binding)
                .descriptor_type(b.descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_infos[i]))
        })
        .collect();

    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }
}

/// Destroy a shader module.
pub fn destroy_shader_module(ctx: &VulkanContext, module: vk::ShaderModule) {
    unsafe {
        ctx.device.destroy_shader_module(module, None);
    }
}

/// Destroy a compute pipeline.
pub fn destroy_pipeline(ctx: &VulkanContext, pipeline: vk::Pipeline) {
    unsafe {
        ctx.device.destroy_pipeline(pipeline, None);
    }
}

/// Destroy a pipeline layout.
pub fn destroy_pipeline_layout(ctx: &VulkanContext, layout: vk::PipelineLayout) {
    unsafe {
        ctx.device.destroy_pipeline_layout(layout, None);
    }
}

/// Destroy a descriptor set layout.
pub fn destroy_descriptor_set_layout(ctx: &VulkanContext, layout: vk::DescriptorSetLayout) {
    unsafe {
        ctx.device.destroy_descriptor_set_layout(layout, None);
    }
}

/// Destroy a descriptor pool (and all sets allocated from it).
pub fn destroy_descriptor_pool(ctx: &VulkanContext, pool: vk::DescriptorPool) {
    unsafe {
        ctx.device.destroy_descriptor_pool(pool, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_descriptor_set_layout_works() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let bindings = [
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
        ];

        let layout = create_descriptor_set_layout(&ctx, &bindings, "test-ds-layout")
            .expect("Failed to create descriptor set layout");
        assert_ne!(layout, vk::DescriptorSetLayout::null());

        destroy_descriptor_set_layout(&ctx, layout);
    }

    #[test]
    fn create_pipeline_layout_with_push_constants() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let bindings = [DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        }];

        let ds_layout = create_descriptor_set_layout(&ctx, &bindings, "test-ds-layout")
            .expect("Failed to create descriptor set layout");

        let pipe_layout = create_pipeline_layout(&ctx, &[ds_layout], 16, "test-pipe-layout")
            .expect("Failed to create pipeline layout");
        assert_ne!(pipe_layout, vk::PipelineLayout::null());

        destroy_pipeline_layout(&ctx, pipe_layout);
        destroy_descriptor_set_layout(&ctx, ds_layout);
    }

    #[test]
    fn invalid_spirv_returns_error() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        // 3 bytes is not a multiple of 4
        let bad_spirv = [0u8, 1, 2];
        let result = create_shader_module(&ctx, &bad_spirv, "bad-spirv");
        assert!(result.is_err());

        match result.unwrap_err() {
            GpuError::InvalidSpirv(len) => assert_eq!(len, 3),
            other => panic!("Expected InvalidSpirv, got: {:?}", other),
        }
    }

    #[test]
    fn descriptor_pool_and_sets() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

        let bindings = [DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        }];

        let ds_layout = create_descriptor_set_layout(&ctx, &bindings, "test-ds-layout")
            .expect("Failed to create descriptor set layout");

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        }];

        let pool = create_descriptor_pool(&ctx, &pool_sizes, 4, "test-pool")
            .expect("Failed to create descriptor pool");

        let sets = allocate_descriptor_sets(&ctx, pool, &[ds_layout, ds_layout])
            .expect("Failed to allocate descriptor sets");
        assert_eq!(sets.len(), 2);

        destroy_descriptor_pool(&ctx, pool);
        destroy_descriptor_set_layout(&ctx, ds_layout);
    }
}
