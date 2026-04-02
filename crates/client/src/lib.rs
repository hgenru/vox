//! # client
//!
//! Rendering and input client for the VOX voxel engine.
//!
//! Manages the window (via winit 0.30), Vulkan surface/swapchain,
//! and frame loop. For the MVP, renders a proof-of-life clear color
//! that changes each frame. Voxel ray tracing comes later.

pub mod renderer;

pub use renderer::Renderer;
