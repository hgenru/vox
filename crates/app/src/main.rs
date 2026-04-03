//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Creates a window, initializes Vulkan with surface support,
//! sets up the renderer, and runs the main event loop.
//!
//! Features:
//! - Island scene with mountain, lake, and lava flow
//! - FPS camera with walk/fly mode toggle (F key)
//! - Material toolbar with number key selection (1-3)
//! - Left click spawns selected material, right click removes
//!
//! Supports headless mode for automated visual testing:
//! ```bash
//! cargo run -p app -- --headless --frames 100 --output screenshot.png
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use client::{PlayerController, Camera, Renderer, renderer::required_instance_extensions};
use glam::Vec3;
use gpu_core::VulkanContext;
use server::{GpuSimulation, ToolbarPushConstants, TOOLBAR_MAX_MATERIALS};
use shared::{
    GRID_SIZE, MAT_LAVA, MAT_STONE, MAT_WATER, PHASE_LIQUID, PHASE_SOLID, Particle,
    RENDER_HEIGHT, RENDER_WIDTH,
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

/// Radius for particle removal (right click).
const REMOVE_RADIUS: f32 = 2.0;

/// A spawnable material entry for the toolbar.
struct MaterialSlot {
    /// Display name.
    name: &'static str,
    /// Material ID (MAT_STONE, MAT_WATER, MAT_LAVA).
    mat_id: u32,
    /// Phase (PHASE_SOLID, PHASE_LIQUID).
    phase: u32,
    /// Initial temperature for spawned particles.
    temperature: f32,
    /// Display color (RGBA, 0..1).
    color: glam::Vec4,
}

/// Material palette available in the toolbar.
fn material_palette() -> Vec<MaterialSlot> {
    vec![
        MaterialSlot {
            name: "Stone",
            mat_id: MAT_STONE,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
        },
        MaterialSlot {
            name: "Water",
            mat_id: MAT_WATER,
            phase: PHASE_LIQUID,
            temperature: 0.0,
            color: glam::Vec4::new(0.2, 0.4, 0.9, 1.0),
        },
        MaterialSlot {
            name: "Lava",
            mat_id: MAT_LAVA,
            phase: PHASE_LIQUID,
            temperature: 2000.0,
            color: glam::Vec4::new(1.0, 0.3, 0.0, 1.0),
        },
    ]
}

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

// ---------------------------------------------------------------------------
// Island scene generation
// ---------------------------------------------------------------------------

/// Compute terrain height at a given (x, z) world position.
///
/// Returns the Y coordinate of the terrain surface. The island has:
/// - Ocean floor at y=2
/// - Gentle island slopes rising from the shore
/// - Central volcano peak reaching ~y=19
fn island_height(x: f32, z: f32) -> f32 {
    let cx = 16.0;
    let cz = 16.0;
    let dx = x - cx;
    let dz = z - cz;
    let dist = (dx * dx + dz * dz).sqrt();

    let base = 2.0; // ocean floor

    // Island shape: smooth cosine falloff from center
    let island_radius = 12.0;
    if dist > island_radius {
        return base;
    }

    let t = dist / island_radius;
    // Gentle hill: rises ~6 units at center, smooth edges
    let island_rise = (1.0 - t * t) * 6.0;

    // Volcano peak: steep rise in center ~4 unit radius
    let peak_radius = 4.0;
    let peak_rise = if dist < peak_radius {
        let pt = dist / peak_radius;
        (1.0 - pt * pt) * 11.0
    } else {
        0.0
    };

    base + island_rise + peak_rise
}

/// Water level for the island scene.
const WATER_LEVEL: f32 = 5.0;

/// Spawn 8 particles (2x2x2 sub-grid) within a single cell.
///
/// Distributes particles at quarter-cell offsets for smoother fluid behavior.
fn spawn_cell(
    particles: &mut Vec<Particle>,
    x: f32,
    y: f32,
    z: f32,
    mass_per_particle: f32,
    mat: u32,
    phase: u32,
) {
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

/// Create the initial set of particles for the island demo scene.
///
/// Scene: volcanic island surrounded by ocean. Stone terrain with a
/// central peak, water ocean around the edges, and a lava pool at the
/// volcano summit flowing down the slope.
fn create_island_particles() -> Vec<Particle> {
    let mut particles = Vec::new();

    // 1. Terrain: stone particles following the heightmap
    for x in 2..30 {
        for z in 2..30 {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let height = island_height(fx, fz);
            let max_y = height.ceil() as i32;

            for y in 2..max_y.min(30) {
                particles.push(Particle::new(
                    Vec3::new(fx, y as f32 + 0.5, fz),
                    1.0,
                    MAT_STONE,
                    PHASE_SOLID,
                ));
            }
        }
    }

    // 2. Ocean water around the island (where terrain is below water level)
    for x in 2..30 {
        for z in 2..30 {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let terrain_h = island_height(fx, fz);

            // Only place water where terrain is below water level
            if terrain_h < WATER_LEVEL {
                let water_start = terrain_h.ceil() as i32;
                let water_end = WATER_LEVEL as i32;
                for y in water_start..water_end {
                    spawn_cell(
                        &mut particles,
                        x as f32,
                        y as f32,
                        z as f32,
                        0.125,
                        MAT_WATER,
                        PHASE_LIQUID,
                    );
                }
            }
        }
    }

    // 3. Lava pool at the volcano summit (small pool at top, flows down one side)
    for x in 14..18 {
        for z in 14..18 {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let terrain_h = island_height(fx, fz);
            let lava_top = (terrain_h + 2.0).min(28.0) as i32;
            let lava_bottom = terrain_h.ceil() as i32;

            for y in lava_bottom..lava_top {
                let start = particles.len();
                spawn_cell(
                    &mut particles,
                    x as f32,
                    y as f32,
                    z as f32,
                    0.125,
                    MAT_LAVA,
                    PHASE_LIQUID,
                );
                // Set lava temperature
                for p in &mut particles[start..] {
                    p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, 2000.0);
                }
            }
        }
    }

    tracing::info!(
        "Island scene: {} stone + {} water + {} lava = {} total",
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

/// Generate the CPU-side heightmap for player ground collision.
///
/// Returns a flat `Vec<f32>` of size `GRID_SIZE * GRID_SIZE`, indexed as
/// `heightmap[z * GRID_SIZE + x]`.
fn generate_heightmap() -> Vec<f32> {
    let size = GRID_SIZE;
    let mut heightmap = vec![0.0f32; (size * size) as usize];
    for z in 0..size {
        for x in 0..size {
            heightmap[(z * size + x) as usize] = island_height(x as f32 + 0.5, z as f32 + 0.5);
        }
    }
    heightmap
}

// ---------------------------------------------------------------------------
// Headless mode
// ---------------------------------------------------------------------------

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

    let particles = create_island_particles();
    tracing::info!("Created {} initial particles", particles.len());
    sim.init_particles(&ctx, &particles)?;

    // Run simulation frames
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

    // Render final frame: camera on the shore looking at the volcano
    let eye = [24.0_f32, 12.0, 24.0];
    let target = [16.0_f32, 8.0, 16.0];
    ctx.execute_one_shot(|cmd| {
        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye, target);
        sim.finalize_render(cmd);
    })?;

    // Readback and save
    let pixel_count = (RENDER_WIDTH as usize) * (RENDER_HEIGHT as usize) * 4;
    let pixels: Vec<u8> =
        gpu_core::buffer::readback::<u8>(&ctx, sim.render_output_gpu_buffer(), pixel_count)?;

    let mut rgba = pixels;
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2); // BGRA → RGBA
    }

    let output_path = args.output.as_deref().unwrap_or("screenshot.png");
    let img = image::RgbaImage::from_raw(RENDER_WIDTH, RENDER_HEIGHT, rgba)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from pixel data"))?;
    img.save(output_path)?;

    tracing::info!("Screenshot saved to {}", output_path);

    sim.destroy(&ctx);
    Ok(())
}

// ---------------------------------------------------------------------------
// Windowed application
// ---------------------------------------------------------------------------

/// Application state for the winit event loop.
struct App {
    window: Option<Arc<Window>>,
    ctx: Option<VulkanContext>,
    renderer: Option<Renderer>,
    sim: Option<GpuSimulation>,
    particles: Vec<Particle>,
    player: PlayerController,
    pressed_keys: HashSet<PhysicalKey>,
    cursor_captured: bool,
    last_frame_time: Instant,
    substeps: u32,
    /// Currently selected material slot index.
    selected_material: usize,
    /// Material palette for the toolbar.
    palette: Vec<MaterialSlot>,
}

impl App {
    fn new(substeps: u32) -> Self {
        let particles = create_island_particles();
        tracing::info!("Created {} initial particles", particles.len());

        // Camera on the shore, looking at the volcano
        let camera = Camera::look_at(
            Vec3::new(24.0, 12.0, 24.0), // shore position
            Vec3::new(16.0, 10.0, 16.0),  // looking at volcano
        );
        let mut player = PlayerController::new(camera);

        // Set up heightmap for ground collision
        let heightmap = generate_heightmap();
        if let Err(e) = player.set_heightmap(heightmap, GRID_SIZE) {
            tracing::warn!("Failed to set heightmap: {}", e);
        }

        let palette = material_palette();

        Self {
            window: None,
            ctx: None,
            renderer: None,
            sim: None,
            particles,
            player,
            pressed_keys: HashSet::new(),
            cursor_captured: false,
            last_frame_time: Instant::now(),
            substeps,
            selected_material: 0,
            palette,
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
    fn spawn_particles(&mut self) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };

        let slot = &self.palette[self.selected_material];
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        let mut new_particles = Vec::new();
        spawn_cell(
            &mut new_particles,
            center.x - 0.5,
            center.y - 0.5,
            center.z - 0.5,
            0.125,
            slot.mat_id,
            slot.phase,
        );

        if slot.temperature != 0.0 {
            for p in &mut new_particles {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, slot.temperature);
            }
        }

        if let Err(e) = sim.add_particles(ctx, &new_particles) {
            tracing::warn!("Failed to spawn particles: {}", e);
        } else {
            tracing::debug!("Spawned {} at aim point", slot.name);
        }
    }

    /// Remove particles near the camera's aim point.
    fn remove_particles(&mut self) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };

        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
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

    /// Build toolbar push constants from current state.
    fn toolbar_push_constants(&self) -> ToolbarPushConstants {
        let mut colors = [glam::Vec4::ZERO; TOOLBAR_MAX_MATERIALS];
        for (i, slot) in self.palette.iter().enumerate() {
            if i < TOOLBAR_MAX_MATERIALS {
                colors[i] = slot.color;
            }
        }
        ToolbarPushConstants {
            screen_width: RENDER_WIDTH,
            screen_height: RENDER_HEIGHT,
            selected_index: self.selected_material as u32,
            material_count: self.palette.len().min(TOOLBAR_MAX_MATERIALS) as u32,
            colors,
        }
    }

    /// Process continuous key movement based on currently pressed keys.
    fn update_movement(&mut self, dt: f32) {
        let keys: Vec<PhysicalKey> = self.pressed_keys.iter().copied().collect();
        for key in keys {
            if let PhysicalKey::Code(code) = key {
                self.player.process_keyboard(code, dt);
            }
        }
        self.player.update(dt);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("VOX Engine — Island Demo")
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

        let size = window.inner_size();
        self.player.camera.update_aspect(size.width, size.height);

        let ctx =
            match VulkanContext::new_with_instance_extensions(required_instance_extensions()) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("Failed to create VulkanContext: {}", e);
                    event_loop.exit();
                    return;
                }
            };

        let renderer = match Renderer::new(&ctx, &window) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to create Renderer: {}", e);
                event_loop.exit();
                return;
            }
        };

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
                self.player.camera.update_aspect(new_size.width, new_size.height);
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

                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            // Escape releases cursor
                            winit::keyboard::KeyCode::Escape => {
                                self.release_cursor();
                            }
                            // F toggles fly/walk mode
                            winit::keyboard::KeyCode::KeyF => {
                                self.player.toggle_fly_mode();
                            }
                            // Number keys select material
                            winit::keyboard::KeyCode::Digit1 => {
                                if !self.palette.is_empty() {
                                    self.selected_material = 0;
                                    tracing::info!("Selected: {}", self.palette[0].name);
                                }
                            }
                            winit::keyboard::KeyCode::Digit2 => {
                                if self.palette.len() > 1 {
                                    self.selected_material = 1;
                                    tracing::info!("Selected: {}", self.palette[1].name);
                                }
                            }
                            winit::keyboard::KeyCode::Digit3 => {
                                if self.palette.len() > 2 {
                                    self.selected_material = 2;
                                    tracing::info!("Selected: {}", self.palette[2].name);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if !self.cursor_captured {
                    self.capture_cursor();
                } else {
                    match button {
                        MouseButton::Left => {
                            self.spawn_particles();
                        }
                        MouseButton::Right => {
                            self.remove_particles();
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if self.cursor_captured && !self.palette.is_empty() {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                    };
                    let len = self.palette.len();
                    if scroll > 0.0 {
                        self.selected_material = (self.selected_material + len - 1) % len;
                    } else if scroll < 0.0 {
                        self.selected_material = (self.selected_material + 1) % len;
                    }
                    tracing::info!(
                        "Selected: {}",
                        self.palette[self.selected_material].name
                    );
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;

                self.update_movement(dt);

                if let (Some(renderer), Some(ctx), Some(sim)) =
                    (self.renderer.as_mut(), self.ctx.as_ref(), self.sim.as_ref())
                {
                    let substeps = self.substeps;
                    let eye = self.player.camera.eye();
                    let target = self.player.camera.target();
                    let toolbar_pc = self.toolbar_push_constants();

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
                        sim.render_toolbar(cmd, &toolbar_pc);
                        sim.finalize_render(cmd);
                    }) {
                        tracing::error!("Simulation/render error: {}", e);
                    }

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
                self.player.process_mouse(dx as f32, dy as f32);
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
