//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Features:
//! - Island scene with mountain, lake, and lava flow
//! - FPS camera with walk/fly mode toggle (F key)
//! - Material toolbar with number key selection (1-3)
//! - Left click spawns selected material, right click removes

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use client::{Camera, PlayerController, Renderer, renderer::required_instance_extensions};
use glam::Vec3;
use gpu_core::VulkanContext;
use server::{GpuSimulation, ToolbarPushConstants, TOOLBAR_MAX_MATERIALS};
use shared::{
    GRID_SIZE, MAT_ASH, MAT_LAVA, MAT_STONE, MAT_WATER, MAT_WOOD, PHASE_LIQUID, PHASE_SOLID,
    Particle, RENDER_HEIGHT, RENDER_WIDTH,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

const DEFAULT_SUBSTEPS: u32 = 2;
const SPAWN_DISTANCE: f32 = 15.0;
const REMOVE_RADIUS: f32 = 2.0;
const EXPLOSION_RADIUS: f32 = 20.0;
const EXPLOSION_STRENGTH: f32 = 800.0;
/// Water level scales with grid: ~12% of grid height.
const WATER_LEVEL_FRAC: f32 = 0.12;
const MARGIN: u32 = 2;
const SHELL_THICKNESS: i32 = 3;
const FLOOR_THICKNESS: i32 = 2;

struct MaterialSlot {
    name: &'static str,
    mat_id: u32,
    phase: u32,
    temperature: f32,
    color: glam::Vec4,
}

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
        MaterialSlot {
            name: "Wood",
            mat_id: MAT_WOOD,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.55, 0.35, 0.15, 1.0),
        },
        MaterialSlot {
            name: "Ash",
            mat_id: MAT_ASH,
            phase: PHASE_SOLID,
            temperature: 0.0,
            color: glam::Vec4::new(0.4, 0.4, 0.4, 1.0),
        },
    ]
}

struct Args {
    headless: bool,
    frames: u32,
    output: Option<String>,
    substeps: u32,
}

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
    Args { headless, frames, output, substeps }
}

/// Compute terrain height at a given (x, z) world position.
///
/// Uses multi-octave sine-based noise combined with island falloff to produce
/// natural-looking terrain with rolling hills, a central ridge/plateau, valleys,
/// and gentle beach slopes. All dimensions scale with GRID_SIZE.
fn island_height(x: f32, z: f32) -> f32 {
    let gs = GRID_SIZE as f32;
    let cx = gs * 0.5;
    let cz = gs * 0.5;
    let dx = x - cx;
    let dz = z - cz;
    let dist = (dx * dx + dz * dz).sqrt();

    let base = MARGIN as f32;
    let island_radius = gs * 0.38;

    // Island falloff: smooth drop to ocean at edges
    if dist > island_radius {
        return base;
    }
    let edge_t = dist / island_radius;
    let falloff = (1.0 - edge_t * edge_t).max(0.0);

    // Multi-octave terrain noise (no external deps, just sin/cos)
    let n1 = (x * 0.05).sin() * (z * 0.07).cos() * gs * 0.06; // big rolling hills
    let n2 = (x * 0.13 + 1.7).sin() * (z * 0.11 + 2.3).cos() * gs * 0.03; // medium features
    let n3 = (x * 0.31 + 0.5).cos() * (z * 0.29 + 1.1).sin() * gs * 0.015; // small details

    // Ridge/plateau at center instead of cone peak
    let center_dist = dist / (gs * 0.12);
    let ridge = if center_dist < 1.0 {
        (1.0 - center_dist * center_dist) * gs * 0.22 // plateau (~56 cells at 256)
    } else {
        0.0
    };

    let terrain = n1 + n2 + n3 + ridge;
    base + falloff * (gs * 0.12 + terrain.max(0.0))
}

/// Spawn 8 particles (2x2x2 sub-grid) within a single cell.
fn spawn_cell(
    particles: &mut Vec<Particle>,
    x: f32, y: f32, z: f32,
    mass_per_particle: f32, mat: u32, phase: u32,
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
/// All ranges scale with GRID_SIZE so the scene works at any resolution.
fn create_island_particles() -> Vec<Particle> {
    let mut particles = Vec::new();
    let gs = GRID_SIZE;
    let margin = MARGIN;
    let upper = gs - margin;
    let water_level = (gs as f32 * WATER_LEVEL_FRAC) as i32;

    // 1a. Thin floor at the bottom
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            for y in (margin as i32)..(margin as i32 + FLOOR_THICKNESS).min(upper as i32) {
                particles.push(Particle::new(
                    Vec3::new(fx, y as f32 + 0.5, fz),
                    1.0, MAT_STONE, PHASE_SOLID,
                ));
            }
        }
    }

    // 1b. Terrain shell: only top SHELL_THICKNESS layers
    for x in margin..upper {
        for z in margin..upper {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let height = island_height(fx, fz);
            let max_y = (height.ceil() as i32).min(upper as i32);
            let min_y = (max_y - SHELL_THICKNESS).max(margin as i32 + FLOOR_THICKNESS);
            for y in min_y..max_y {
                particles.push(Particle::new(
                    Vec3::new(fx, y as f32 + 0.5, fz),
                    1.0, MAT_STONE, PHASE_SOLID,
                ));
            }
        }
    }

    // 2. Ocean water — shoreline band, 1 layer deep, 8 PPC for proper MPM flow
    let cx = gs as f32 * 0.5;
    let cz = gs as f32 * 0.5;
    let island_radius = gs as f32 * 0.375;
    let water_inner = island_radius * 0.6;  // inner edge of water band
    let water_outer = island_radius * 1.3;  // outer edge of water band
    for x in margin..(gs - margin) {
        for z in margin..(gs - margin) {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let dx = fx - cx;
            let dz = fz - cz;
            let dist = (dx * dx + dz * dz).sqrt();
            // Only generate water in a band around the island
            if dist < water_inner || dist > water_outer {
                continue;
            }
            let terrain_h = island_height(fx, fz);
            if terrain_h < water_level as f32 {
                // Only spawn the top 1 layer of water (8 PPC for proper MPM fluid behavior)
                let water_top = water_level;
                let water_bottom = (water_top - 1).max(terrain_h.ceil() as i32);
                for y in water_bottom..water_top {
                    spawn_cell(&mut particles, fx - 0.5, y as f32, fz - 0.5, 0.125, MAT_WATER, PHASE_LIQUID);
                }
            }
        }
    }

    // 3. Lava pool at the volcano summit — 8 PPC, max 2 layers for proper MPM flow
    let lava_min = (gs as f32 * 0.44) as u32;
    let lava_max = (gs as f32 * 0.56) as u32;
    let max_lava_layers: i32 = 2;
    for x in lava_min..lava_max {
        for z in lava_min..lava_max {
            let fx = x as f32 + 0.5;
            let fz = z as f32 + 0.5;
            let terrain_h = island_height(fx, fz);
            let lava_top = (terrain_h + gs as f32 * 0.03).min((gs - margin) as f32) as i32;
            let lava_bottom = (lava_top - max_lava_layers).max(terrain_h.ceil() as i32);
            for y in lava_bottom..lava_top {
                spawn_cell(&mut particles, fx - 0.5, y as f32, fz - 0.5, 0.125, MAT_LAVA, PHASE_LIQUID);
            }
            // Set temperature on newly spawned lava particles
            let lava_count = ((lava_top - lava_bottom).max(0) as usize) * 8;
            let start = particles.len() - lava_count;
            for p in &mut particles[start..] {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, 2000.0);
            }
        }
    }

    tracing::info!(
        "Island scene: {} stone + {} water + {} lava = {} total",
        particles.iter().filter(|p| p.material_id() == MAT_STONE).count(),
        particles.iter().filter(|p| p.material_id() == MAT_WATER).count(),
        particles.iter().filter(|p| p.material_id() == MAT_LAVA).count(),
        particles.len(),
    );
    particles
}

/// Generate the CPU-side heightmap for player ground collision.
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

fn run_headless(args: &Args) -> Result<()> {
    tracing::info!("Headless mode: {} frames, {} substeps/frame", args.frames, args.substeps);
    let ctx = VulkanContext::new()?;
    let mut sim = GpuSimulation::new(&ctx)?;
    let particles = create_island_particles();
    sim.init_particles(&ctx, &particles)?;

    for i in 0..args.frames {
        ctx.execute_one_shot(|cmd| {
            for _ in 0..args.substeps { sim.step(cmd); }
        })?;
        if i % 10 == 0 { tracing::info!("Frame {}/{}", i, args.frames); }
    }

    let gs = GRID_SIZE as f32;
    let eye = [gs * 0.75, gs * 0.4, gs * 0.75];
    let target = [gs * 0.5, gs * 0.3, gs * 0.5];
    ctx.execute_one_shot(|cmd| {
        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye, target);
        sim.finalize_render(cmd);
    })?;

    let pixel_count = (RENDER_WIDTH as usize) * (RENDER_HEIGHT as usize) * 4;
    let pixels: Vec<u8> = gpu_core::buffer::readback::<u8>(&ctx, sim.render_output_gpu_buffer(), pixel_count)?;
    let mut rgba = pixels;
    for pixel in rgba.chunks_exact_mut(4) { pixel.swap(0, 2); }

    let output_path = args.output.as_deref().unwrap_or("screenshot.png");
    let img = image::RgbaImage::from_raw(RENDER_WIDTH, RENDER_HEIGHT, rgba)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from pixel data"))?;
    img.save(output_path)?;
    tracing::info!("Screenshot saved to {}", output_path);
    sim.destroy(&ctx);
    Ok(())
}

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
    selected_material: usize,
    palette: Vec<MaterialSlot>,
    pending_explosion: Option<[f32; 3]>,
}

impl App {
    fn new(substeps: u32) -> Self {
        let particles = create_island_particles();
        tracing::info!("Created {} initial particles", particles.len());

        let gs = GRID_SIZE as f32;
        let camera = Camera::look_at(
            Vec3::new(gs * 0.75, gs * 0.4, gs * 0.75),
            Vec3::new(gs * 0.5, gs * 0.3, gs * 0.5),
        );
        let mut player = PlayerController::new(camera);
        let heightmap = generate_heightmap();
        if let Err(e) = player.set_heightmap(heightmap, GRID_SIZE) {
            tracing::warn!("Failed to set heightmap: {}", e);
        }

        Self {
            window: None, ctx: None, renderer: None, sim: None,
            particles, player,
            pressed_keys: HashSet::new(),
            cursor_captured: false,
            last_frame_time: Instant::now(),
            substeps,
            selected_material: 0,
            palette: material_palette(),
            pending_explosion: None,
        }
    }

    fn capture_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(false);
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            self.cursor_captured = true;
        }
    }

    fn release_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(true);
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            self.cursor_captured = false;
        }
    }

    fn spawn_particles(&mut self) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };
        let slot = &self.palette[self.selected_material];
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        let mut new_particles = Vec::new();
        // Spawn a larger cluster for liquids so they are visible (3x3x3 = 27 cells x 8 PPC = 216 particles)
        let spawn_size: u32 = if slot.phase == PHASE_LIQUID { 3 } else { 1 };
        let half = spawn_size as f32 * 0.5;
        for dx in 0..spawn_size {
            for dy in 0..spawn_size {
                for dz in 0..spawn_size {
                    spawn_cell(
                        &mut new_particles,
                        center.x - half + dx as f32,
                        center.y - half + dy as f32,
                        center.z - half + dz as f32,
                        0.125, slot.mat_id, slot.phase,
                    );
                }
            }
        }
        if slot.temperature != 0.0 {
            for p in &mut new_particles {
                p.vel_temp = glam::Vec4::new(0.0, 0.0, 0.0, slot.temperature);
            }
        }
        if let Err(e) = sim.add_particles(ctx, &new_particles) {
            tracing::warn!("Failed to spawn particles: {}", e);
        }
    }

    fn trigger_explosion(&mut self) {
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        self.pending_explosion = Some([center.x, center.y, center.z]);
        tracing::info!("Explosion at [{:.1}, {:.1}, {:.1}]", center.x, center.y, center.z);
    }

    fn remove_particles(&mut self) {
        let (ctx, sim) = match (self.ctx.as_ref(), self.sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        let radius_sq = REMOVE_RADIUS * REMOVE_RADIUS;
        match sim.readback_particles(ctx) {
            Ok(particles) => {
                let filtered: Vec<Particle> = particles.into_iter()
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
            Err(e) => tracing::warn!("Failed to readback for remove: {}", e),
        }
    }

    fn toolbar_push_constants(&self) -> ToolbarPushConstants {
        let mut colors = [glam::Vec4::ZERO; TOOLBAR_MAX_MATERIALS];
        for (i, slot) in self.palette.iter().enumerate() {
            if i < TOOLBAR_MAX_MATERIALS { colors[i] = slot.color; }
        }
        ToolbarPushConstants {
            screen_width: RENDER_WIDTH, screen_height: RENDER_HEIGHT,
            selected_index: self.selected_material as u32,
            material_count: self.palette.len().min(TOOLBAR_MAX_MATERIALS) as u32,
            colors,
        }
    }

    fn update_movement(&mut self, dt: f32) {
        let keys: Vec<PhysicalKey> = self.pressed_keys.iter().copied().collect();
        for key in keys {
            if let PhysicalKey::Code(code) = key { self.player.process_keyboard(code, dt); }
        }
        self.player.update(dt);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = WindowAttributes::default()
            .with_title("VOX Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(RENDER_WIDTH, RENDER_HEIGHT));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => { tracing::error!("Failed to create window: {}", e); event_loop.exit(); return; }
        };

        let size = window.inner_size();
        self.player.camera.update_aspect(size.width, size.height);

        let ctx = match VulkanContext::new_with_instance_extensions(required_instance_extensions()) {
            Ok(c) => c,
            Err(e) => { tracing::error!("Failed to create VulkanContext: {}", e); event_loop.exit(); return; }
        };
        let renderer = match Renderer::new(&ctx, &window) {
            Ok(r) => r,
            Err(e) => { tracing::error!("Failed to create Renderer: {}", e); event_loop.exit(); return; }
        };
        let mut sim = match GpuSimulation::new(&ctx) {
            Ok(s) => s,
            Err(e) => { tracing::error!("Failed to create GpuSimulation: {}", e); event_loop.exit(); return; }
        };
        if let Err(e) = sim.init_particles(&ctx, &self.particles) {
            tracing::error!("Failed to upload particles: {}", e); event_loop.exit(); return;
        }
        tracing::info!("GpuSimulation initialized with {} particles", self.particles.len());

        self.window = Some(window);
        self.ctx = Some(ctx);
        self.renderer = Some(renderer);
        self.sim = Some(sim);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ctx) = self.ctx.as_ref() {
                    if let Some(sim) = self.sim.as_mut() { sim.destroy(ctx); }
                    if let Some(renderer) = self.renderer.as_mut() { renderer.destroy(ctx); }
                }
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = self.renderer.as_mut() { renderer.notify_resized(new_size.width, new_size.height); }
                self.player.camera.update_aspect(new_size.width, new_size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => { self.pressed_keys.insert(event.physical_key); }
                    ElementState::Released => { self.pressed_keys.remove(&event.physical_key); }
                }
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            winit::keyboard::KeyCode::Escape => self.release_cursor(),
                            winit::keyboard::KeyCode::KeyF => self.player.toggle_fly_mode(),
                            winit::keyboard::KeyCode::Digit1 if !self.palette.is_empty() => {
                                self.selected_material = 0;
                                tracing::info!("Selected: {}", self.palette[0].name);
                            }
                            winit::keyboard::KeyCode::Digit2 if self.palette.len() > 1 => {
                                self.selected_material = 1;
                                tracing::info!("Selected: {}", self.palette[1].name);
                            }
                            winit::keyboard::KeyCode::Digit3 if self.palette.len() > 2 => {
                                self.selected_material = 2;
                                tracing::info!("Selected: {}", self.palette[2].name);
                            }
                            winit::keyboard::KeyCode::Digit4 if self.palette.len() > 3 => {
                                self.selected_material = 3;
                                tracing::info!("Selected: {}", self.palette[3].name);
                            }
                            winit::keyboard::KeyCode::Digit5 if self.palette.len() > 4 => {
                                self.selected_material = 4;
                                tracing::info!("Selected: {}", self.palette[4].name);
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.cursor_captured {
                    self.capture_cursor();
                } else {
                    match button {
                        MouseButton::Left => self.spawn_particles(),
                        MouseButton::Right => self.remove_particles(),
                        MouseButton::Middle => self.trigger_explosion(),
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
                    if scroll > 0.0 { self.selected_material = (self.selected_material + len - 1) % len; }
                    else if scroll < 0.0 { self.selected_material = (self.selected_material + 1) % len; }
                    tracing::info!("Selected: {}", self.palette[self.selected_material].name);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;
                self.update_movement(dt);

                let substeps = self.substeps;
                let eye = self.player.camera.eye();
                let target = self.player.camera.target();
                let toolbar_pc = self.toolbar_push_constants();
                let explosion = self.pending_explosion.take();
                if let (Some(renderer), Some(ctx), Some(sim)) = (self.renderer.as_mut(), self.ctx.as_ref(), self.sim.as_ref()) {
                    if let Err(e) = ctx.execute_one_shot(|cmd| {
                        for _ in 0..substeps { sim.step_physics(cmd); }
                        sim.step_react(cmd);
                        if let Some(center) = explosion {
                            sim.apply_explosion(cmd, center, EXPLOSION_RADIUS, EXPLOSION_STRENGTH);
                        }
                        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, [eye.x, eye.y, eye.z], [target.x, target.y, target.z]);
                        sim.render_toolbar(cmd, &toolbar_pc);
                        sim.finalize_render(cmd);
                    }) {
                        tracing::error!("Simulation/render error: {}", e);
                    }
                    match renderer.draw_frame_with_buffer(ctx, sim.render_output_buffer(), RENDER_WIDTH, RENDER_HEIGHT) {
                        Ok(_) => {}
                        Err(e) => tracing::error!("Frame error: {}", e),
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.cursor_captured { self.player.process_mouse(dx as f32, dy as f32); }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window { window.request_redraw(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn island_particles_under_limit() {
        let particles = create_island_particles();
        assert!(
            particles.len() < shared::MAX_PARTICLES as usize,
            "Scene has {} particles, max is {}",
            particles.len(),
            shared::MAX_PARTICLES
        );
    }

    #[test]
    fn heightmap_values_valid() {
        let hm = generate_heightmap();
        assert_eq!(hm.len(), (GRID_SIZE * GRID_SIZE) as usize);
        for h in &hm {
            assert!(h.is_finite(), "NaN in heightmap");
            assert!(*h >= 0.0 && *h < GRID_SIZE as f32, "Height {} out of bounds", h);
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    tracing::info!("VOX Engine starting");
    let args = parse_args();
    if args.headless { return run_headless(&args); }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(args.substeps);
    event_loop.run_app(&mut app)?;
    tracing::info!("VOX Engine shut down");
    Ok(())
}
