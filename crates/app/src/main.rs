//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Creates a window, initializes Vulkan with surface support,
//! sets up the renderer, and runs the main event loop.
//! For the MVP, renders a proof-of-life cycling clear color while
//! running the GPU MPM simulation each frame.
//!
//! Supports headless mode for automated visual testing:
//! ```bash
//! cargo run -p app -- --headless --frames 100 --output screenshot.png
//! ```

use std::sync::Arc;

use anyhow::Result;
use client::{Renderer, renderer::required_instance_extensions};
use gpu_core::VulkanContext;
use server::GpuSimulation;
use shared::{
    MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle, RENDER_HEIGHT, RENDER_WIDTH,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

/// Command-line arguments for the app.
struct Args {
    /// Run in headless mode (no window, render to PNG).
    headless: bool,
    /// Number of simulation frames to run in headless mode.
    frames: u32,
    /// Output file path for the screenshot (headless mode).
    output: Option<String>,
}

/// Parse command-line arguments from `std::env::args`.
fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let headless = args.contains(&"--headless".to_string());
    let frames = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1).cloned());
    Args {
        headless,
        frames,
        output,
    }
}

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

/// Run the engine in headless mode: simulate, render, and save a PNG.
fn run_headless(args: &Args) -> Result<()> {
    tracing::info!(
        "Headless mode: {} frames, output: {}",
        args.frames,
        args.output.as_deref().unwrap_or("screenshot.png")
    );

    let ctx = VulkanContext::new()?;
    let mut sim = GpuSimulation::new(&ctx)?;

    let particles = create_initial_particles();
    tracing::info!("Created {} initial particles", particles.len());
    sim.init_particles(&ctx, &particles)?;

    // Run simulation frames
    for i in 0..args.frames {
        ctx.execute_one_shot(|cmd| {
            sim.step(cmd);
        })?;
        if i % 10 == 0 {
            tracing::info!("Frame {}/{}", i, args.frames);
        }
    }

    // Render final frame
    ctx.execute_one_shot(|cmd| {
        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT);
    })?;

    // Readback render output buffer
    let pixel_count = (RENDER_WIDTH as usize) * (RENDER_HEIGHT as usize) * 4;
    let pixels: Vec<u8> =
        gpu_core::buffer::readback::<u8>(&ctx, sim.render_output_gpu_buffer(), pixel_count)?;

    // Convert BGRA to RGBA for PNG
    let mut rgba = pixels;
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2); // swap B and R
    }

    // Save PNG
    let output_path = args.output.as_deref().unwrap_or("screenshot.png");
    let img = image::RgbaImage::from_raw(RENDER_WIDTH, RENDER_HEIGHT, rgba)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from pixel data"))?;
    img.save(output_path)?;

    tracing::info!("Screenshot saved to {}", output_path);

    sim.destroy(&ctx);
    Ok(())
}

/// Application state for the winit event loop.
struct App {
    /// The winit window (created on first `resumed` call).
    window: Option<Arc<Window>>,
    /// Vulkan context (created alongside the window).
    ctx: Option<VulkanContext>,
    /// Renderer (surface + swapchain + frame management).
    renderer: Option<Renderer>,
    /// GPU simulation (created on resume, after particles are uploaded).
    sim: Option<GpuSimulation>,
    /// Initial particles (stored until GPU simulation is ready).
    particles: Vec<Particle>,
}

impl App {
    fn new() -> Self {
        let particles = create_initial_particles();
        tracing::info!("Created {} initial particles", particles.len());

        Self {
            window: None,
            ctx: None,
            renderer: None,
            sim: None,
            particles,
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
        let ctx =
            match VulkanContext::new_with_instance_extensions(required_instance_extensions()) {
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

        // Create GPU simulation and upload initial particles
        let mut sim = match GpuSimulation::new(&ctx) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to create GpuSimulation: {}", e);
                event_loop.exit();
                return;
            }
        };

        if let Err(e) = sim.init_particles(&ctx, &self.particles) {
            tracing::error!("Failed to upload particles: {}", e);
            event_loop.exit();
            return;
        }

        tracing::info!(
            "GpuSimulation initialized with {} particles",
            self.particles.len()
        );

        self.window = Some(window);
        self.ctx = Some(ctx);
        self.renderer = Some(renderer);
        self.sim = Some(sim);
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
                if let Some(ctx) = self.ctx.as_ref() {
                    if let Some(sim) = self.sim.as_mut() {
                        sim.destroy(ctx);
                    }
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.destroy(ctx);
                    }
                }
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.notify_resized(new_size.width, new_size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(ctx), Some(sim)) =
                    (self.renderer.as_mut(), self.ctx.as_ref(), self.sim.as_ref())
                {
                    // Run simulation step + render to buffer (separate submission)
                    if let Err(e) = ctx.execute_one_shot(|cmd| {
                        sim.step(cmd);
                        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT);
                    }) {
                        tracing::error!("Simulation/render error: {}", e);
                    }

                    // Copy render output buffer to swapchain and present
                    match renderer.draw_frame_with_buffer(
                        ctx,
                        sim.render_output_buffer(),
                        RENDER_WIDTH,
                        RENDER_HEIGHT,
                    ) {
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

    let args = parse_args();
    if args.headless {
        return run_headless(&args);
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    tracing::info!("VOX Engine shut down");
    Ok(())
}
