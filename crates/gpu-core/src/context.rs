//! Vulkan context: instance, device, queues, and GPU memory allocator.
//!
//! [`VulkanContext`] owns the entire Vulkan lifetime: instance, debug messenger,
//! physical/logical device, queues, command pool, and a gpu-allocator [`Allocator`].
//! It supports headless mode (no surface) for testing.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::AllocationSizes;
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::sync::Mutex;

use crate::error::{GpuError, Result};

/// All required device extensions for the VOX engine.
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    // RT ray tracing: BLAS/TLAS for voxel bricks
    ash::khr::acceleration_structure::NAME,
    // RT ray tracing: ray queries from fragment shader
    ash::khr::ray_query::NAME,
    // Renderpass-less rendering (Vulkan 1.3 promoted but extension still needed for feature enable)
    ash::khr::dynamic_rendering::NAME,
    // Pipeline barriers v2, required by the engine's sync model
    ash::khr::synchronization2::NAME,
    // Bindless descriptors for material/texture arrays
    ash::ext::descriptor_indexing::NAME,
    // Avoid vec3 padding in storage buffers (see CLAUDE.md trap #1)
    ash::ext::scalar_block_layout::NAME,
    // GPU pointer access for acceleration structure builds
    ash::khr::buffer_device_address::NAME,
    // SPIR-V 1.4 features used by rust-gpu output
    ash::khr::spirv_1_4::NAME,
    // Required by VK_KHR_acceleration_structure
    ash::khr::deferred_host_operations::NAME,
    // Required by VK_KHR_spirv_1_4
    ash::khr::shader_float_controls::NAME,
    // P2G shader uses spirv_std::arch::atomic_f_add() for scatter accumulation
    ash::ext::shader_atomic_float::NAME,
    // Swapchain presentation to window surface
    ash::khr::swapchain::NAME,
];

/// Indices of the queue families used by the engine.
#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    /// Queue family supporting both graphics and compute.
    pub graphics: u32,
    /// Dedicated or shared compute queue family.
    pub compute: u32,
}

/// The core Vulkan context owning all GPU resources.
///
/// Create via [`VulkanContext::new`] (headless) or
/// [`VulkanContext::new_with_instance_extensions`] (windowed).
/// All Vulkan objects are destroyed in [`Drop`] in the correct order.
pub struct VulkanContext {
    // --- Fields are ordered for correct drop: allocator first, then device, then instance ---

    /// GPU memory allocator. Wrapped in ManuallyDrop so we can drop it before
    /// the device in our Drop impl.
    pub allocator: ManuallyDrop<Mutex<Allocator>>,

    /// General-purpose command pool (for the graphics queue family).
    pub command_pool: vk::CommandPool,

    /// Logical device.
    pub device: ash::Device,

    /// Graphics queue (also supports compute).
    pub graphics_queue: vk::Queue,

    /// Compute queue.
    pub compute_queue: vk::Queue,

    /// Queue family indices.
    pub queue_families: QueueFamilyIndices,

    /// Selected physical device.
    pub physical_device: vk::PhysicalDevice,

    /// Physical device properties.
    pub device_properties: vk::PhysicalDeviceProperties,

    /// Physical device memory properties.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,

    /// Debug messenger (only in debug builds).
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    /// Debug utils extension loader (only in debug builds).
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,

    /// Debug utils device loader (for naming objects, only in debug builds).
    #[cfg(debug_assertions)]
    debug_utils_device: Option<ash::ext::debug_utils::Device>,

    /// Vulkan instance.
    pub instance: ash::Instance,

    /// Vulkan entry point (library loader).
    pub entry: ash::Entry,
}

impl VulkanContext {
    /// Create a new headless Vulkan context (no surface/window).
    ///
    /// Selects a discrete GPU, enables all required extensions, creates queues,
    /// command pool, and gpu-allocator.
    pub fn new() -> Result<Self> {
        Self::create_internal(None)
    }

    /// Create a new Vulkan context with additional instance extensions
    /// (e.g., surface extensions for windowed mode).
    ///
    /// The caller must provide the required surface instance extension names.
    pub fn new_with_instance_extensions(extra_instance_extensions: &[&CStr]) -> Result<Self> {
        Self::create_internal(Some(extra_instance_extensions))
    }

    fn create_internal(extra_instance_extensions: Option<&[&CStr]>) -> Result<Self> {
        // Load Vulkan
        let entry = unsafe { ash::Entry::load()? };

        // Instance creation
        let app_name = c"VOX Engine";
        let engine_name = c"VOX";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Instance extensions
        let mut instance_extensions: Vec<*const i8> = Vec::new();

        if let Some(extras) = extra_instance_extensions {
            for ext in extras {
                instance_extensions.push(ext.as_ptr());
            }
        }

        // Validation layers + debug utils in debug builds (if available)
        let mut layer_names_raw: Vec<*const i8> = Vec::new();
        let mut _debug_utils_enabled = false;

        #[cfg(debug_assertions)]
        let _validation_layer_name = c"VK_LAYER_KHRONOS_validation";
        #[cfg(debug_assertions)]
        {
            let available_layers = unsafe {
                entry
                    .enumerate_instance_layer_properties()
                    .unwrap_or_default()
            };
            let has_validation = available_layers.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                name == _validation_layer_name.as_ref()
            });

            if has_validation {
                layer_names_raw.push(_validation_layer_name.as_ptr());
                instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
                _debug_utils_enabled = true;
                tracing::info!("Vulkan validation layers enabled");
            } else {
                tracing::warn!(
                    "Vulkan validation layer not available, running without validation"
                );
            }
        }

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_raw)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&instance_ci, None)? };
        tracing::info!("Vulkan 1.3 instance created");

        // Debug messenger (only if debug utils extension is enabled)
        let (debug_utils_loader, debug_messenger) = if _debug_utils_enabled {
            Self::setup_debug_messenger(&entry, &instance)
        } else {
            (None, None)
        };

        // Physical device selection
        let (physical_device, device_properties) = Self::pick_physical_device(&instance)?;

        let device_name =
            unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) };
        tracing::info!("Selected GPU: {}", device_name.to_string_lossy());

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Queue families
        let queue_families = Self::find_queue_families(&instance, physical_device)?;
        tracing::info!(
            "Queue families: graphics={}, compute={}",
            queue_families.graphics,
            queue_families.compute
        );

        // Create logical device
        let device =
            Self::create_device(&instance, physical_device, &queue_families)?;
        tracing::info!("Logical device created");

        // Debug utils device loader for naming objects
        #[cfg(debug_assertions)]
        let debug_utils_device = debug_utils_loader
            .as_ref()
            .map(|_| ash::ext::debug_utils::Device::new(&instance, &device));

        // Get queues
        let graphics_queue =
            unsafe { device.get_device_queue(queue_families.graphics, 0) };
        let compute_queue =
            unsafe { device.get_device_queue(queue_families.compute, 0) };

        // Command pool for the graphics family
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_families.graphics)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None)? };

        // GPU allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: {
                let mut s = gpu_allocator::AllocatorDebugSettings::default();
                s.log_memory_information = cfg!(debug_assertions);
                s.log_leaks_on_shutdown = true;
                s.log_allocations = cfg!(debug_assertions);
                s.log_frees = cfg!(debug_assertions);
                s
            },
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .map_err(GpuError::Allocator)?;
        tracing::info!("GPU allocator initialized");

        Ok(Self {
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            command_pool,
            device,
            graphics_queue,
            compute_queue,
            queue_families,
            physical_device,
            device_properties,
            memory_properties,
            debug_messenger,
            debug_utils_loader,
            #[cfg(debug_assertions)]
            debug_utils_device,
            instance,
            entry,
        })
    }

    /// Set up the debug messenger for validation layer output.
    /// Returns (None, None) in release builds.
    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (
        Option<ash::ext::debug_utils::Instance>,
        Option<vk::DebugUtilsMessengerEXT>,
    ) {
        #[cfg(not(debug_assertions))]
        {
            let _ = (entry, instance);
            return (None, None);
        }

        #[cfg(debug_assertions)]
        {
            let debug_utils_loader =
                ash::ext::debug_utils::Instance::new(entry, instance);

            let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));

            let messenger = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .ok()
            };

            if messenger.is_some() {
                tracing::info!("Debug messenger installed");
            }

            (Some(debug_utils_loader), messenger)
        }
    }

    /// Pick the best physical device (prefer discrete GPU).
    fn pick_physical_device(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
        let devices = unsafe { instance.enumerate_physical_devices()? };

        if devices.is_empty() {
            return Err(GpuError::NoSuitableDevice);
        }

        // Prefer discrete GPU, fall back to any
        let mut best = None;
        for pd in &devices {
            let props = unsafe { instance.get_physical_device_properties(*pd) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                best = Some((*pd, props));
                break;
            }
            if best.is_none() {
                best = Some((*pd, props));
            }
        }

        best.ok_or(GpuError::NoSuitableDevice)
    }

    /// Find queue families that support graphics+compute.
    fn find_queue_families(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<QueueFamilyIndices> {
        let families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics = None;
        let mut compute_only = None;

        for (i, family) in families.iter().enumerate() {
            let i = i as u32;
            let has_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let has_compute = family.queue_flags.contains(vk::QueueFlags::COMPUTE);

            if has_graphics && has_compute && graphics.is_none() {
                graphics = Some(i);
            }

            // Prefer a dedicated compute queue if available
            if has_compute && !has_graphics && compute_only.is_none() {
                compute_only = Some(i);
            }
        }

        let graphics_idx = graphics.ok_or_else(|| {
            GpuError::NoSuitableQueueFamily(
                "No graphics+compute queue family".into(),
            )
        })?;

        // Use dedicated compute queue if available, otherwise share with graphics
        let compute_idx = compute_only.unwrap_or(graphics_idx);

        Ok(QueueFamilyIndices {
            graphics: graphics_idx,
            compute: compute_idx,
        })
    }

    /// Create the logical device with all required extensions and features.
    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_families: &QueueFamilyIndices,
    ) -> Result<ash::Device> {
        // Check extension support
        let supported_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device)?
        };
        let supported_names: Vec<&CStr> = supported_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .collect();

        for required in REQUIRED_DEVICE_EXTENSIONS {
            if !supported_names.contains(required) {
                return Err(GpuError::ExtensionNotSupported(
                    required.to_string_lossy().into_owned(),
                ));
            }
        }

        // Extension names as raw pointers
        let extension_names: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect();

        // Queue create infos
        let queue_priorities = [1.0f32];
        let mut queue_cis = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_families.graphics)
            .queue_priorities(&queue_priorities)];
        if queue_families.compute != queue_families.graphics {
            queue_cis.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_families.compute)
                    .queue_priorities(&queue_priorities),
            );
        }

        // Enable required features via chained structs
        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .scalar_block_layout(true)
            .runtime_descriptor_array(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .timeline_semaphore(true);
        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut accel_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
        let mut ray_query_features =
            vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut atomic_float_features =
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
                .shader_buffer_float32_atomics(true)
                .shader_buffer_float32_atomic_add(true);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3)
            .push_next(&mut accel_features)
            .push_next(&mut ray_query_features)
            .push_next(&mut atomic_float_features);

        let device = unsafe {
            instance.create_device(physical_device, &device_ci, None)?
        };

        Ok(device)
    }

    /// Execute a one-shot command buffer on the graphics queue, then wait.
    ///
    /// The provided closure receives a command buffer that is already in the
    /// recording state. After the closure returns, the command buffer is
    /// ended, submitted, and waited on.
    pub fn execute_one_shot<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer),
    {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin_info)?;

            f(cmd);

            self.device.end_command_buffer(cmd)?;

            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(self.graphics_queue, &[submit], vk::Fence::null())?;
            self.device.queue_wait_idle(self.graphics_queue)?;

            self.device
                .free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(())
    }

    /// Set a debug name on a Vulkan object (no-op in release builds).
    pub fn set_debug_name<T: vk::Handle>(&self, handle: T, name: &str) {
        #[cfg(debug_assertions)]
        {
            if let Some(ref debug_device) = self.debug_utils_device {
                if let Ok(c_name) = CString::new(name) {
                    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(handle)
                        .object_name(&c_name);
                    let _ =
                        unsafe { debug_device.set_debug_utils_object_name(&name_info) };
                }
            }
        }

        #[cfg(not(debug_assertions))]
        {
            let _ = (handle, name);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        tracing::info!("Destroying VulkanContext");
        unsafe {
            // Wait for all GPU work to finish
            let _ = self.device.device_wait_idle();

            // Drop allocator first (before device)
            ManuallyDrop::drop(&mut self.allocator);

            // Destroy command pool
            self.device
                .destroy_command_pool(self.command_pool, None);

            // Destroy debug messenger
            if let (Some(loader), Some(messenger)) =
                (&self.debug_utils_loader, self.debug_messenger)
            {
                loader.destroy_debug_utils_messenger(messenger, None);
            }

            // Destroy device
            self.device.destroy_device(None);

            // Destroy instance
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan debug callback that routes messages to the `tracing` crate.
#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if callback_data.is_null() {
        return vk::FALSE;
    }

    let msg = unsafe {
        let data = &*callback_data;
        if data.p_message.is_null() {
            "<no message>"
        } else {
            CStr::from_ptr(data.p_message)
                .to_str()
                .unwrap_or("<invalid utf8>")
        }
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!("[Vulkan] {}", msg);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!("[Vulkan] {}", msg);
    }

    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_headless_context() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        assert_ne!(ctx.physical_device, vk::PhysicalDevice::null());
        assert_ne!(ctx.graphics_queue, vk::Queue::null());
        assert_ne!(ctx.compute_queue, vk::Queue::null());

        let device_name =
            unsafe { CStr::from_ptr(ctx.device_properties.device_name.as_ptr()) };
        println!("GPU: {}", device_name.to_string_lossy());
    }

    #[test]
    fn queue_families_valid() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        // Graphics queue must support both graphics and compute
        let families = unsafe {
            ctx.instance
                .get_physical_device_queue_family_properties(ctx.physical_device)
        };
        let gf = &families[ctx.queue_families.graphics as usize];
        assert!(gf.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        assert!(gf.queue_flags.contains(vk::QueueFlags::COMPUTE));
    }

    #[test]
    fn execute_one_shot_works() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        // Just record and submit an empty command buffer
        ctx.execute_one_shot(|_cmd| {
            // no-op
        })
        .expect("execute_one_shot failed");
    }
}
