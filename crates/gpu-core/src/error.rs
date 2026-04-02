//! Error types for the gpu-core crate.

/// Errors that can occur in gpu-core operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// Vulkan API error.
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    /// Failed to load the Vulkan library.
    #[error("Failed to load Vulkan library: {0}")]
    LoadingError(#[from] ash::LoadingError),

    /// GPU memory allocation error.
    #[error("GPU allocator error: {0}")]
    Allocator(#[from] gpu_allocator::AllocationError),

    /// No suitable physical device found.
    #[error("No suitable physical device found")]
    NoSuitableDevice,

    /// No suitable queue family found.
    #[error("No suitable queue family: {0}")]
    NoSuitableQueueFamily(String),

    /// Required extension not supported.
    #[error("Required extension not supported: {0}")]
    ExtensionNotSupported(String),

    /// Invalid SPIR-V data.
    #[error("Invalid SPIR-V data: byte length {0} is not a multiple of 4")]
    InvalidSpirv(usize),

    /// Buffer mapping failed.
    #[error("Buffer mapping failed: allocation has no mapped pointer")]
    MappingFailed,
}

/// Result type alias for gpu-core operations.
pub type Result<T> = std::result::Result<T, GpuError>;
