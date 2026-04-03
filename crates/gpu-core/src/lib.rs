//! # gpu-core
//!
//! Vulkan foundation layer built on `ash`.
//! Provides: VulkanContext (device, queues, allocator), buffer/image creation,
//! compute + graphics pipeline creation, BLAS/TLAS management,
//! synchronization primitives, and debug utilities.

pub mod buffer;
pub mod context;
pub mod error;
pub mod frame;
pub mod image;
pub mod pipeline;
pub mod swapchain;
pub mod vram_budget;

pub use context::VulkanContext;
pub use error::{GpuError, Result};
pub use vram_budget::VramBudget;
