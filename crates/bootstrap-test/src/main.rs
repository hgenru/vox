//! Bootstrap test: compile rust-gpu shader → load in ash → dispatch → readback → verify.
//!
//! Validates the entire toolchain: rust-gpu + spirv-builder + ash + Vulkan on this machine.

use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

const ELEMENT_COUNT: usize = 64;

fn main() -> Result<()> {
    println!("=== Bootstrap Test: rust-gpu + ash ===");

    // Load SPIR-V compiled by build.rs
    let spv_bytes = include_bytes!(env!("BOOTSTRAP_SPV_PATH"));
    println!("[OK] SPIR-V compiled: {} bytes", spv_bytes.len());

    // Create Vulkan instance
    let entry = unsafe { ash::Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance_ci = vk::InstanceCreateInfo::default()
        .application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_ci, None)? };
    println!("[OK] Vulkan instance created");

    // Pick physical device
    let phys_devices = unsafe { instance.enumerate_physical_devices()? };
    let phys_device = phys_devices
        .into_iter()
        .find(|&pd| {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .context("No discrete GPU found")?;

    let props = unsafe { instance.get_physical_device_properties(phys_device) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    println!("[OK] GPU: {}", name.to_string_lossy());

    // Find compute queue family
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys_device) };
    let compute_family = queue_families
        .iter()
        .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .context("No compute queue family")? as u32;

    // Create device
    let queue_priorities = [1.0f32];
    let queue_ci = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_family)
        .queue_priorities(&queue_priorities);
    let device_ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_ci));
    let device = unsafe { instance.create_device(phys_device, &device_ci, None)? };
    let queue = unsafe { device.get_device_queue(compute_family, 0) };
    println!("[OK] Device + compute queue created");

    // Memory helpers
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys_device) };
    let find_memory_type = |type_bits: u32, flags: vk::MemoryPropertyFlags| -> Result<u32> {
        for i in 0..mem_props.memory_type_count {
            if (type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(flags)
            {
                return Ok(i);
            }
        }
        anyhow::bail!("No suitable memory type found")
    };

    let buf_size = (ELEMENT_COUNT * std::mem::size_of::<u32>()) as vk::DeviceSize;

    // Create device-local buffer and staging buffer
    let create_buffer = |usage: vk::BufferUsageFlags| -> Result<vk::Buffer> {
        let ci = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        Ok(unsafe { device.create_buffer(&ci, None)? })
    };

    let device_buf = create_buffer(
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let staging_buf = create_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    )?;

    // Allocate and bind memory
    let alloc_and_bind = |buf: vk::Buffer, host_visible: bool| -> Result<vk::DeviceMemory> {
        let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
        let flags = if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mem_type = find_memory_type(reqs.memory_type_bits, flags)?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type);
        let mem = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buf, mem, 0)? };
        Ok(mem)
    };

    let device_mem = alloc_and_bind(device_buf, false)?;
    let staging_mem = alloc_and_bind(staging_buf, true)?;

    // Upload input data: [0, 1, 2, ..., 63]
    let input_data: Vec<u32> = (0..ELEMENT_COUNT as u32).collect();
    unsafe {
        let ptr = device.map_memory(staging_mem, 0, buf_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(input_data.as_ptr() as *const u8, ptr as *mut u8, buf_size as usize);
        device.unmap_memory(staging_mem);
    }

    // Command pool + buffer
    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(compute_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None)? };
    let cmd_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc)? }[0];

    // Copy staging → device
    unsafe {
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        let region = vk::BufferCopy::default().size(buf_size);
        device.cmd_copy_buffer(cmd, staging_buf, device_buf, &[region]);
        device.end_command_buffer(cmd)?;

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, &[submit], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }
    println!("[OK] Input data uploaded");

    // Create shader module (copy to aligned buffer since include_bytes may not be 4-byte aligned)
    let spv_words: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spv_words);
    let shader_module = unsafe { device.create_shader_module(&shader_ci, None)? };
    println!("[OK] Shader module created");

    // Descriptor set layout: 1 storage buffer
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);
    let ds_layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(std::slice::from_ref(&binding));
    let ds_layout = unsafe { device.create_descriptor_set_layout(&ds_layout_ci, None)? };

    // Pipeline layout + compute pipeline
    let pipe_layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&ds_layout));
    let pipe_layout = unsafe { device.create_pipeline_layout(&pipe_layout_ci, None)? };

    let entry_name = c"main_cs";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_name);
    let pipeline_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipe_layout);
    let pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            .map_err(|e| e.1)?[0]
    };
    println!("[OK] Compute pipeline created");

    // Descriptor pool + set
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
    };
    let dp_ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));
    let desc_pool = unsafe { device.create_descriptor_pool(&dp_ci, None)? };
    let ds_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(std::slice::from_ref(&ds_layout));
    let desc_set = unsafe { device.allocate_descriptor_sets(&ds_alloc)? }[0];

    let buf_info = vk::DescriptorBufferInfo::default()
        .buffer(device_buf)
        .range(buf_size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&buf_info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

    // Dispatch
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipe_layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_dispatch(cmd, 1, 1, 1); // 1 workgroup of 64 threads
        device.end_command_buffer(cmd)?;

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, &[submit], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }
    println!("[OK] Compute dispatch completed");

    // Readback: device → staging → map
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        let region = vk::BufferCopy::default().size(buf_size);
        device.cmd_copy_buffer(cmd, device_buf, staging_buf, &[region]);
        device.end_command_buffer(cmd)?;

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, &[submit], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    let result: Vec<u32> = unsafe {
        let ptr = device.map_memory(staging_mem, 0, buf_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(ptr as *const u32, ELEMENT_COUNT);
        let v = slice.to_vec();
        device.unmap_memory(staging_mem);
        v
    };

    // Verify: each element should be doubled
    let mut pass = true;
    for i in 0..ELEMENT_COUNT {
        let expected = (i as u32) * 2;
        if result[i] != expected {
            println!("[FAIL] result[{}] = {}, expected {}", i, result[i], expected);
            pass = false;
        }
    }

    if pass {
        println!("[OK] All {} values correct! data[i] = data[i] * 2", ELEMENT_COUNT);
        println!("\n=== BOOTSTRAP TEST PASSED ===");
        println!("rust-gpu + ash toolchain works on this machine.");
    } else {
        println!("\n=== BOOTSTRAP TEST FAILED ===");
        std::process::exit(1);
    }

    // Cleanup
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipe_layout, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_descriptor_set_layout(ds_layout, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_buffer(device_buf, None);
        device.destroy_buffer(staging_buf, None);
        device.free_memory(device_mem, None);
        device.free_memory(staging_mem, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
