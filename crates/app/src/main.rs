//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Creates a window, initializes Vulkan with surface support,
//! sets up the renderer, and runs the main event loop.
//! For the MVP, renders a proof-of-life cycling clear color while
//! running the GPU MPM simulation each frame.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use client::{Camera, Renderer, renderer::required_instance_extensions};
use glam::Vec3;
use gpu_core::VulkanContext;
use server::GpuSimulation;
use shared::{
    MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle, RENDER_HEIGHT, RENDER_WIDTH,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

/// Create the initial set of particles for the MVP demo.
///
/// Stone floor at y=2..4 across the grid, and a water cube
/// floating at y=15..20 in the center.
fn create_initial_particles() -> Vec<Particle> {
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
    /// GPU simulation (created on resume, after particles are uploaded).
    sim: Option<GpuSimulation>,
    /// Initial particles (stored until GPU simulation is ready).
    particles: Vec<Particle>,
    /// FPS camera for WASD + mouse look.
    camera: Camera,
    /// Set of currently pressed physical keys for continuous movement.
    pressed_keys: HashSet<PhysicalKey>,
    /// Whether the mouse cursor is captured for camera look.
    cursor_captured: bool,
    /// Timestamp of the last frame for delta-time calculation.
    last_frame_time: Instant,
}

impl App {
    fn new() -> Self {
        let particles = create_initial_particles();
        tracing::info!("Created {} initial particles", particles.len());

        // Camera looking toward grid center from corner
        let camera = Camera::new(
            Vec3::new(48.0, 24.0, 48.0),
            -2.35, // yaw: looking toward origin
            -0.3,  // pitch: slightly downward
        );

        Self {
            window: None,
            ctx: None,
            renderer: None,
            sim: None,
            particles,
            camera,
            pressed_keys: HashSet::new(),
            cursor_captured: false,
            last_frame_time: Instant::now(),
        }
    }

    /// Capture the mouse cursor for camera look control.
    fn capture_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(false);
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            self.cursor_captured = true;
        }
    }

    /// Release the mouse cursor.
    fn release_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(true);
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            self.cursor_captured = false;
        }
    }

    /// Process continuous key movement based on currently pressed keys.
    fn update_movement(&mut self, dt: f32) {
        let keys: Vec<PhysicalKey> = self.pressed_keys.iter().copied().collect();
        for key in keys {
            if let PhysicalKey::Code(code) = key {
                self.camera.process_keyboard(code, dt);
            }
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

        // Update camera aspect from actual window size
        let size = window.inner_size();
        self.camera.update_aspect(size.width, size.height);

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
                self.camera.update_aspect(new_size.width, new_size.height);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => {
                        self.pressed_keys.insert(event.physical_key);
                    }
                    ElementState::Released => {
                        self.pressed_keys.remove(&event.physical_key);
                    }
                }

                // Escape releases cursor
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(winit::keyboard::KeyCode::Escape) =
                        event.physical_key
                    {
                        self.release_cursor();
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !self.cursor_captured {
                    self.capture_cursor();
                }
            }

            WindowEvent::RedrawRequested => {
                // Calculate delta time
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;

                // Process continuous movement from held keys
                self.update_movement(dt);

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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.cursor_captured {
                self.camera.process_mouse(dx as f32, dy as f32);
            }
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
