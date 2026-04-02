//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Creates a window, initializes Vulkan with surface support,
//! sets up the renderer, and runs the main event loop.
//! For the MVP, renders a proof-of-life cycling clear color.
//! Simulation compute dispatches will be added once the server crate is ready.

use std::sync::Arc;

use anyhow::Result;
use client::{Renderer, renderer::required_instance_extensions};
use gpu_core::VulkanContext;
use shared::{
    MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle, RENDER_HEIGHT, RENDER_WIDTH,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

/// Create the initial set of particles for the MVP demo.
///
/// Stone floor at y=2..4 across the grid, and a water cube
/// floating at y=15..20 in the center.
fn create_initial_particles() -> Vec<Particle> {
    use glam::Vec3;

    let mut particles = Vec::new();

    // Stone floor
    for x in 2..30 {
        for z in 2..30 {
            for y in 2..4 {
                particles.push(Particle::new(
                    Vec3::new(x as f32, y as f32, z as f32),
                    1.0,
                    MAT_STONE,
                    PHASE_SOLID,
                ));
            }
        }
    }

    // Water cube
    for x in 12..20 {
        for z in 12..20 {
            for y in 15..20 {
                particles.push(Particle::new(
                    Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
                    1.0,
                    MAT_WATER,
                    PHASE_LIQUID,
                ));
            }
        }
    }

    particles
}

/// Application state for the winit event loop.
struct App {
    /// The winit window (created on first `resumed` call).
    window: Option<Arc<Window>>,
    /// Vulkan context (created alongside the window).
    ctx: Option<VulkanContext>,
    /// Renderer (surface + swapchain + frame management).
    renderer: Option<Renderer>,
    /// Initial particles (stored until GPU simulation is ready).
    _particles: Vec<Particle>,
}

impl App {
    fn new() -> Self {
        let particles = create_initial_particles();
        tracing::info!("Created {} initial particles", particles.len());

        Self {
            window: None,
            ctx: None,
            renderer: None,
            _particles: particles,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Only create the window once
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("VOX Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(RENDER_WIDTH, RENDER_HEIGHT));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("Failed to create window: {}", e);
                event_loop.exit();
                return;
            }
        };

        tracing::info!(
            "Window created: {}x{}",
            window.inner_size().width,
            window.inner_size().height
        );

        // Create Vulkan context with surface extensions
        let ctx = match VulkanContext::new_with_instance_extensions(required_instance_extensions())
        {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to create VulkanContext: {}", e);
                event_loop.exit();
                return;
            }
        };

        // Create renderer (surface + swapchain)
        let renderer = match Renderer::new(&ctx, &window) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to create Renderer: {}", e);
                event_loop.exit();
                return;
            }
        };

        self.window = Some(window);
        self.ctx = Some(ctx);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("Close requested, shutting down");
                // Clean up before exit
                if let (Some(renderer), Some(ctx)) = (self.renderer.as_mut(), self.ctx.as_ref()) {
                    renderer.destroy(ctx);
                }
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.notify_resized(new_size.width, new_size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(ctx)) = (self.renderer.as_mut(), self.ctx.as_ref()) {
                    match renderer.draw_frame(ctx) {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::error!("Frame error: {}", e);
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("VOX Engine starting");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    tracing::info!("VOX Engine shut down");
    Ok(())
}
