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

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use client::{Camera, Renderer, renderer::required_instance_extensions};
use glam::Vec3;
use gpu_core::VulkanContext;
use server::GpuSimulation;
use shared::{
    MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle, RENDER_HEIGHT,
    RENDER_WIDTH,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

/// Default number of simulation substeps per frame.
const DEFAULT_SUBSTEPS: u32 = 4;

/// Distance from camera eye at which particles are spawned.
const SPAWN_DISTANCE: f32 = 8.0;

/// Radius for particle removal (middle click).
const REMOVE_RADIUS: f32 = 2.0;

/// Command-line arguments for the app.
struct Args {
    /// Run in headless mode (no window, render to PNG).
    headless: bool,
    /// Number of simulation frames to run in headless mode.
    frames: u32,
    /// Output file path for the screenshot (headless mode).
    output: Option<String>,
    /// Number of simulation substeps per frame.
    substeps: u32,
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
    let substeps = args
        .iter()
        .position(|a| a == "--substeps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SUBSTEPS);
    Args {
        headless,
        frames,
        output,
        substeps,
    }
}

/// Spawn 8 particles (2x2x2 sub-grid) within a single cell.
///
/// Distributes particles at quarter-cell offsets for smoother fluid behavior.
/// This eliminates grid-seam artifacts caused by particles clustering at
/// grid nodes when using only 1 particle per cell.
fn spawn_cell(particles: &mut Vec<Particle>, x: f32, y: f32, z: f32, mass_per_particle: f32, mat: u32, phase: u32) {
    for dx in 0..2u32 {
        for dy in 0..2u32 {
            for dz in 0..2u32 {
                let pos = Vec3::new(
                    x + 0.25 + dx as f32 * 0.5,
                    y + 0.25 + dy as f32 * 0.5,
                    z + 0.25 + dz as f32 * 0.5,
                );
                particles.push(Particle::new(pos, mass_per_particle, mat, phase));
            }
        }
    }
}

/// Create the initial set of particles for the MVP demo.
///
/// Scene: hollow ellipsoidal cave carved from a stone block,
/// with a water lake at the bottom and a lava stream dripping
/// from a ceiling crack. Demonstrates water-lava reaction
/// (stone + steam), temperature, and phase transitions.
///
/// Stone uses 1 particle per cell (solid, pinned). Water and lava use
/// 8 particles per cell (2x2x2 sub-grid) for smoother fluid behavior.
fn create_initial_particles() -> Vec<Particle> {
    use shared::GRID_SIZE;

    let mut particles = Vec::new();
    let gs = GRID_SIZE as f32;
    let half = gs / 2.0;
    let center = Vec3::new(half, half * 0.75, half); // 64, 48, 64
    let radii = Vec3::new(gs * 0.3125, gs * 0.25, gs * 0.3125); // 40, 32, 40

    let margin = 2u32;
    let stone_top = GRID_SIZE - margin - 4; // leave some headroom

    // 1. Stone shell: fill solid block, skip interior of ellipsoid (= cave)
    for x in margin..(GRID_SIZE - margin) {
        for z in margin..(GRID_SIZE - margin) {
            for y in margin..stone_top {
                let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d = (pos - center) / radii;
                let dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;

                if dist_sq > 1.0 {
                    // Outside ellipsoid = stone wall
                    particles.push(Particle::new(pos, 1.0, MAT_STONE, PHASE_SOLID));
                }
            }
        }
    }

    // 2. Water lake at bottom of cave (8 PPC for smooth fluid)
    let water_min = (gs * 0.22) as u32; // ~28
    let water_max = (gs * 0.78) as u32; // ~100
    let water_y_min = (gs * 0.125) as u32; // ~16
    let water_y_max = (gs * 0.22) as u32; // ~28
    for x in water_min..water_max {
        for z in water_min..water_max {
            for y in water_y_min..water_y_max {
                let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d = (pos - center) / radii;
                let dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;

                if dist_sq < 1.0 {
                    // Inside cave = water, 8 particles per cell
                    spawn_cell(&mut particles, x as f32, y as f32, z as f32, 0.125, MAT_WATER, PHASE_LIQUID);
                }
            }
        }
    }

    // 3. Lava stream from ceiling crack (8 PPC for smooth fluid)
    let lava_min = (gs * 0.4375) as u32; // ~56
    let lava_max = (gs * 0.5625) as u32; // ~72
    let lava_y_min = (gs * 0.5625) as u32; // ~72
    let lava_y_max = (gs * 0.8125) as u32; // ~104
    for x in lava_min..lava_max {
        for z in lava_min..lava_max {
            for y in lava_y_min..lava_y_max {
                let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d = (pos - center) / radii;
                let dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;

                // Only place lava where there's no stone (inside cave or near surface)
                if dist_sq < 1.2 {
                    let start = particles.len();
                    spawn_cell(&mut particles, x as f32, y as f32, z as f32, 0.125, MAT_LAVA, PHASE_LIQUID);
                    // Set lava temperature to 2000K (above melting point, stays liquid and glows)
                    for p in &mut particles[start..] {
                        p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, 2000.0);
                    }
                }
            }
        }
    }

    tracing::info!(
        "Scene: {} stone + {} water + {} lava = {} total",
        particles
            .iter()
            .filter(|p| p.material_id() == MAT_STONE)
            .count(),
        particles
            .iter()
            .filter(|p| p.material_id() == MAT_WATER)
            .count(),
        particles
            .iter()
            .filter(|p| p.material_id() == MAT_LAVA)
            .count(),
        particles.len(),
    );

    particles
}

/// Run the engine in headless mode: simulate, render, and save a PNG.
fn run_headless(args: &Args) -> Result<()> {
    tracing::info!(
        "Headless mode: {} frames, {} substeps/frame, output: {}",
        args.frames,
        args.substeps,
        args.output.as_deref().unwrap_or("screenshot.png")
    );

    let ctx = VulkanContext::new()?;
    let mut sim = GpuSimulation::new(&ctx)?;

    let particles = create_initial_particles();
    tracing::info!("Created {} initial particles", particles.len());
    sim.init_particles(&ctx, &particles)?;

    // Run simulation frames (multiple substeps per frame for faster sim)
    for i in 0..args.frames {
        ctx.execute_one_shot(|cmd| {
            for _ in 0..args.substeps {
                sim.step(cmd);
            }
        })?;
        if i % 10 == 0 {
            tracing::info!("Frame {}/{}", i, args.frames);
        }
    }

    // Render final frame with camera inside the cave (positions scale with grid)
    let gs = shared::GRID_SIZE as f32;
    let eye = [gs * 0.5, gs * 0.3125, gs * 0.8125];
    let target = [gs * 0.5, gs * 0.25, gs * 0.5];
    ctx.execute_one_shot(|cmd| {
        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye, target);
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
    /// FPS camera for WASD + mouse look.
    camera: Camera,
    /// Set of currently pressed physical keys for continuous movement.
    pressed_keys: HashSet<PhysicalKey>,
    /// Whether the mouse cursor is captured for camera look.
    cursor_captured: bool,
    /// Last known cursor position in pixels (for spawn ray calculation).
    cursor_position: (f32, f32),
    /// Timestamp of the last frame for delta-time calculation.
    last_frame_time: Instant,
    /// Number of simulation substeps per frame.
    substeps: u32,
}

impl App {
    fn new(substeps: u32) -> Self {
        let particles = create_initial_particles();
        tracing::info!("Created {} initial particles", particles.len());

        // Camera inside the cave, looking toward the center/water lake
        // Positions scale with grid size so the scene works at any resolution
        let gs = shared::GRID_SIZE as f32;
        let camera = Camera::look_at(
            Vec3::new(gs * 0.5, gs * 0.3125, gs * 0.8125),  // inside cave, near wall
            Vec3::new(gs * 0.5, gs * 0.25, gs * 0.5),       // looking at center/water
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
            cursor_position: (0.0, 0.0),
            last_frame_time: Instant::now(),
            substeps,
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

    /// Spawn a small cube of particles at the camera's aim point.
    ///
    /// Creates a 2x2x2 block of particles at `camera.eye() + forward * SPAWN_DISTANCE`.
    /// `mat` and `phase` define the material type. `temperature` sets initial temperature.
    fn spawn_particles(&mut self, mat: u32, phase: u32, temperature: f32) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };

        let center = self.camera.eye() + self.camera.forward() * SPAWN_DISTANCE;
        let mut new_particles = Vec::new();
        spawn_cell(&mut new_particles, center.x - 0.5, center.y - 0.5, center.z - 0.5, 0.125, mat, phase);

        // Set temperature on all spawned particles
        if temperature != 0.0 {
            for p in &mut new_particles {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, temperature);
            }
        }

        if let Err(e) = sim.add_particles(ctx, &new_particles) {
            tracing::warn!("Failed to spawn particles: {}", e);
        }
    }

    /// Remove particles near the camera's aim point within [`REMOVE_RADIUS`].
    ///
    /// Reads back all particles, filters out those within the radius of the
    /// aim point, and re-uploads. Expensive but acceptable for MVP.
    fn remove_particles(&mut self) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };

        let center = self.camera.eye() + self.camera.forward() * SPAWN_DISTANCE;
        let radius_sq = REMOVE_RADIUS * REMOVE_RADIUS;

        match sim.readback_particles(ctx) {
            Ok(particles) => {
                let filtered: Vec<Particle> = particles
                    .into_iter()
                    .filter(|p| (p.position() - center).length_squared() > radius_sq)
                    .collect();
                let removed = sim.num_particles() as usize - filtered.len();
                if removed > 0 {
                    if let Err(e) = sim.init_particles(ctx, &filtered) {
                        tracing::warn!("Failed to re-upload after remove: {}", e);
                    } else {
                        tracing::info!("Removed {} particles", removed);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to readback for remove: {}", e);
            }
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

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = (position.x as f32, position.y as f32);
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if !self.cursor_captured {
                    // First click captures the cursor
                    self.capture_cursor();
                } else {
                    // While captured, clicks spawn or remove material
                    match button {
                        MouseButton::Left => {
                            self.spawn_particles(MAT_WATER, PHASE_LIQUID, 0.0);
                        }
                        MouseButton::Right => {
                            self.spawn_particles(MAT_LAVA, PHASE_LIQUID, 2000.0);
                        }
                        MouseButton::Middle => {
                            self.remove_particles();
                        }
                        _ => {}
                    }
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
                    // Run simulation substeps + render to buffer
                    let substeps = self.substeps;
                    let eye = self.camera.eye();
                    let target = self.camera.target();
                    if let Err(e) = ctx.execute_one_shot(|cmd| {
                        for _ in 0..substeps {
                            sim.step(cmd);
                        }
                        sim.render(
                            cmd,
                            RENDER_WIDTH,
                            RENDER_HEIGHT,
                            [eye.x, eye.y, eye.z],
                            [target.x, target.y, target.z],
                        );
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

    let args = parse_args();
    if args.headless {
        return run_headless(&args);
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(args.substeps);
    event_loop.run_app(&mut app)?;

    tracing::info!("VOX Engine shut down");
    Ok(())
}
