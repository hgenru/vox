//! Application struct and winit event handling.
//!
//! Contains the `App` struct that holds all runtime state (window, Vulkan
//! context, renderer, simulation, player controller) and the
//! `ApplicationHandler` implementation that drives the main loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use client::{Camera, PlayerController, Renderer, renderer::required_instance_extensions};
use content::MaterialDatabase;
use glam::Vec3;
use gpu_core::VulkanContext;
use server::{GpuSimulation, ToolbarPushConstants, TOOLBAR_MAX_MATERIALS};
use server::ca_simulation::CaSimulation;
use server::pbmpm_zones::ActivationTrigger;
use shared::{GRID_SIZE, PHASE_LIQUID, Particle, RENDER_HEIGHT, RENDER_WIDTH};
use shared::constants::CA_CHUNK_SIZE;
use shared::voxel::Voxel;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use crate::scene::{
    MaterialSlot, ca_material_palette, create_island_particles, create_mountain_particles,
    generate_heightmap, generate_mountain_heightmap, material_palette, spawn_cell,
};

use world::{ChunkCoord, WorldManager};

/// Distance (in grid units) from the camera eye at which particles are
/// spawned, removed, or explosions are centered.
const SPAWN_DISTANCE: f32 = 15.0;
/// Radius used when removing particles with right-click.
const REMOVE_RADIUS: f32 = 2.0;
/// Radius of the explosion effect triggered by middle-click.
const EXPLOSION_RADIUS: f32 = 20.0;
/// Impulse strength of the explosion effect.
const EXPLOSION_STRENGTH: f32 = 800.0;

/// Top-level application state for the windowed mode.
pub(crate) struct App {
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
    material_db: Option<MaterialDatabase>,
    last_react_time: Instant,
    /// Previous camera eye position for change detection.
    prev_eye: [f32; 3],
    /// Previous camera target position for change detection.
    prev_target: [f32; 3],
    /// Cached result from `readback_any_active()` — one frame late but safe
    /// to query outside command buffer recording.
    world_is_static: bool,
    /// Last time OOB flag was checked (throttled to avoid GPU stalls).
    last_oob_check: Instant,
    /// Number of consecutive frames with no render (for logging).
    frames_skipped: u64,
    /// Frame counter for FPS calculation, reset every second.
    fps_frame_count: u32,
    /// Timer reset every second for FPS calculation.
    fps_timer: Instant,
    /// Most recently computed FPS value.
    current_fps: f32,
    /// World manager for chunk streaming (used when `--world` flag is set or as default).
    world: Option<WorldManager>,
    /// Current chunk coordinate the camera is in.
    current_chunk: ChunkCoord,
    /// If true, use CaSimulation (sim2) instead of GpuSimulation.
    use_ca_sim: bool,
    /// CaSimulation instance (when `--sim2` is used).
    ca_sim: Option<CaSimulation>,
}

impl App {
    /// Create a new `App` with the given number of physics substeps per frame.
    ///
    /// If `scene_path` is `Some`, loads a RON scene definition from that file.
    /// If `use_world` is `true`, the world manager generates terrain procedurally.
    /// Otherwise falls back to the default procedural island scene.
    /// If `model_path` is provided, the `.vox` model is loaded and its
    /// particles are added to the scene at `model_pos` (default 32,20,32).
    pub(crate) fn new(
        substeps: u32,
        scene_path: Option<&str>,
        big: bool,
        model_path: Option<&str>,
        model_pos: Option<(f32, f32, f32)>,
        use_world: bool,
        use_ca_sim: bool,
    ) -> Self {
        // Try loading materials from RON file, fall back to hardcoded defaults
        let material_db = match content::load_material_database("assets/materials.ron") {
            Ok(db) => {
                tracing::info!("Loaded material database from assets/materials.ron");
                Some(db)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load assets/materials.ron, using hardcoded defaults: {}",
                    e
                );
                None
            }
        };

        // Determine scene source: WorldManager, RON scene, or built-in procedural.
        let (particles, scene_camera, use_mountain, world_mgr, start_chunk) = if use_world {
            tracing::info!("Using WorldManager for chunk-streamed terrain");
            let seed = 12345u64;
            let cache_dir = std::path::PathBuf::from("world_cache");
            std::fs::create_dir_all(&cache_dir).ok();
            match WorldManager::new(seed, cache_dir) {
                Ok(mut world_mgr) => {
                    // Start at world origin chunk
                    let start_chunk = ChunkCoord::new(0, 0, 0);
                    if let Err(e) = world_mgr.update_center(start_chunk) {
                        tracing::error!("Failed to initialize world center: {}", e);
                    }
                    let mut particles = world_mgr.pack_particles_for_gpu();
                    // Offset particles so all positions are within [0, GRID_SIZE).
                    // pack_particles_for_gpu places center chunk at origin, so chunks
                    // at negative offsets produce negative coordinates. Shift by
                    // sim_radius * CHUNK_SIZE in X and Z to make everything positive.
                    let offset_xz =
                        world_mgr.sim_radius() as f32 * shared::constants::CHUNK_SIZE as f32;
                    for p in &mut particles {
                        let pos = p.position();
                        p.set_position(Vec3::new(pos.x + offset_xz, pos.y, pos.z + offset_xz));
                    }
                    tracing::info!(
                        "World: {} chunks loaded, {} particles",
                        world_mgr.active_chunk_count(),
                        particles.len(),
                    );
                    (particles, None, false, Some(world_mgr), start_chunk)
                }
                Err(e) => {
                    tracing::error!("Failed to create WorldManager: {}, falling back to island", e);
                    (create_island_particles(), None, false, None, ChunkCoord::new(0, 0, 0))
                }
            }
        } else if let Some(path) = scene_path {
            match content::load_scene(path) {
                Ok(scene) => {
                    tracing::info!("Loaded scene '{}' from {}", scene.name, path);
                    let cam = scene.camera.clone();
                    let p = scene.spawn_particles();
                    (p, cam, false, None, ChunkCoord::new(0, 0, 0))
                }
                Err(e) => {
                    tracing::warn!("Failed to load scene '{}': {}, using default", path, e);
                    (create_island_particles(), None, false, None, ChunkCoord::new(0, 0, 0))
                }
            }
        } else if big {
            tracing::info!("Using mountain range scene (--big)");
            (create_mountain_particles(), None, true, None, ChunkCoord::new(0, 0, 0))
        } else {
            (create_island_particles(), None, false, None, ChunkCoord::new(0, 0, 0))
        };

        let mut particles = particles;
        if let Some(path) = model_path {
            let pos = model_pos.unwrap_or((32.0, 20.0, 32.0));
            let offset = Vec3::new(pos.0, pos.1, pos.2);
            let palette = content::vox_loader::default_palette_mapping();
            match content::load_vox_model(path, offset, &palette) {
                Ok(model_particles) => {
                    tracing::info!("Loaded {} particles from model '{}'", model_particles.len(), path);
                    particles.extend(model_particles);
                }
                Err(e) => {
                    tracing::warn!("Failed to load .vox model '{}': {}", path, e);
                }
            }
        }
        tracing::info!("Created {} initial particles", particles.len());

        let camera = if let Some(ref cam) = scene_camera {
            Camera::look_at(
                Vec3::new(cam.eye.0, cam.eye.1, cam.eye.2),
                Vec3::new(cam.target.0, cam.target.1, cam.target.2),
            )
        } else if world_mgr.is_some() {
            // For world mode: start camera in the center of the active grid, looking forward
            let gs = GRID_SIZE as f32;
            Camera::look_at(
                Vec3::new(gs * 0.5, gs * 0.7, gs * 0.5),
                Vec3::new(gs * 0.5, gs * 0.3, gs * 0.5 + 1.0),
            )
        } else {
            let gs = GRID_SIZE as f32;
            Camera::look_at(
                Vec3::new(gs * 0.75, gs * 0.4, gs * 0.75),
                Vec3::new(gs * 0.5, gs * 0.3, gs * 0.5),
            )
        };
        let mut player = PlayerController::new(camera);
        // Only generate heightmap for procedural scenes (RON scenes don't have procedural terrain)
        if scene_camera.is_none() && world_mgr.is_none() {
            let heightmap = if use_mountain {
                generate_mountain_heightmap()
            } else {
                generate_heightmap()
            };
            if let Err(e) = player.set_heightmap(heightmap, GRID_SIZE) {
                tracing::warn!("Failed to set heightmap: {}", e);
            }
        }

        Self {
            window: None, ctx: None, renderer: None, sim: None,
            particles, player,
            pressed_keys: HashSet::new(),
            cursor_captured: false,
            last_frame_time: Instant::now(),
            substeps,
            selected_material: 0,
            palette: if use_ca_sim { ca_material_palette() } else { material_palette() },
            pending_explosion: None,
            material_db,
            last_react_time: Instant::now(),
            prev_eye: [0.0; 3],
            prev_target: [0.0; 3],
            world_is_static: false,
            last_oob_check: Instant::now(),
            frames_skipped: 0,
            fps_frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            world: world_mgr,
            current_chunk: start_chunk,
            use_ca_sim,
            ca_sim: None,
        }
    }

    /// Lock the cursor inside the window and hide it (FPS-style grab).
    fn capture_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(false);
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            self.cursor_captured = true;
        }
    }

    /// Release the cursor so the user can interact with the OS.
    fn release_cursor(&mut self) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(true);
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            self.cursor_captured = false;
        }
    }

    /// Spawn a cluster of particles of the currently selected material in
    /// front of the camera.
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

    /// Queue an explosion at the point in front of the camera.
    fn trigger_explosion(&mut self) {
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        self.pending_explosion = Some([center.x, center.y, center.z]);
        tracing::info!("Explosion at [{:.1}, {:.1}, {:.1}]", center.x, center.y, center.z);
    }

    /// Remove all particles within `REMOVE_RADIUS` of the point in front of
    /// the camera.
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

    /// Build the push-constants struct describing the current toolbar state.
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

    /// Returns `true` roughly every 100ms to throttle reaction dispatch.
    fn should_run_react(&mut self, now: Instant) -> bool {
        if now.duration_since(self.last_react_time).as_millis() >= 100 {
            self.last_react_time = now;
            true
        } else {
            false
        }
    }

    /// Spawn a cluster of CA voxels of the selected material in front of the camera.
    fn ca_spawn_voxels(&mut self) {
        let ca_mat_id = self.ca_palette_material_id();
        let temp: u8 = if ca_mat_id == 4 { 250 } else { 128 };
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;

        let (ctx, ca_sim) = match (self.ctx.as_ref(), self.ca_sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };
        let wx = center.x as i32;
        let wy = center.y as i32;
        let wz = center.z as i32;
        let cs = CA_CHUNK_SIZE as i32;

        let chunk_coord = [wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs)];
        let mut data = match ca_sim.download_chunk(ctx, chunk_coord) {
            Ok(d) => d,
            Err(e) => {
                tracing::debug!("Cannot spawn CA voxels: {}", e);
                return;
            }
        };

        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;

        // Place a 3x3x3 cube of material
        for dz in 0..3i32 {
            for dy in 0..3i32 {
                for dx in 0..3i32 {
                    let px = (lx as i32 + dx).clamp(0, 31) as usize;
                    let py = (ly as i32 + dy).clamp(0, 31) as usize;
                    let pz = (lz as i32 + dz).clamp(0, 31) as usize;
                    let idx = pz * 32 * 32 + py * 32 + px;
                    data[idx] = Voxel::new(ca_mat_id, temp, 0, 15, 0).0;
                }
            }
        }

        if let Err(e) = ca_sim.upload_chunk_voxels(ctx, chunk_coord, &data) {
            tracing::warn!("Failed to upload CA voxels: {}", e);
        } else {
            tracing::debug!("Spawned CA material {} at chunk {:?}", ca_mat_id, chunk_coord);
        }
    }

    /// Remove CA voxels (set to air) in front of the camera.
    fn ca_remove_voxels(&mut self) {
        let (ctx, ca_sim) = match (self.ctx.as_ref(), self.ca_sim.as_mut()) {
            (Some(c), Some(s)) => (c, s),
            _ => return,
        };
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        let wx = center.x as i32;
        let wy = center.y as i32;
        let wz = center.z as i32;
        let cs = CA_CHUNK_SIZE as i32;

        let chunk_coord = [wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs)];
        let mut data = match ca_sim.download_chunk(ctx, chunk_coord) {
            Ok(d) => d,
            Err(e) => {
                tracing::debug!("Cannot remove CA voxels: {}", e);
                return;
            }
        };

        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;

        // Clear a 3x3x3 cube to air
        for dz in 0..3i32 {
            for dy in 0..3i32 {
                for dx in 0..3i32 {
                    let px = (lx as i32 + dx).clamp(0, 31) as usize;
                    let py = (ly as i32 + dy).clamp(0, 31) as usize;
                    let pz = (lz as i32 + dz).clamp(0, 31) as usize;
                    let idx = pz * 32 * 32 + py * 32 + px;
                    data[idx] = Voxel::air().0;
                }
            }
        }

        if let Err(e) = ca_sim.upload_chunk_voxels(ctx, chunk_coord, &data) {
            tracing::warn!("Failed to upload CA voxels (remove): {}", e);
        } else {
            tracing::debug!("Removed CA voxels at chunk {:?}", chunk_coord);
        }
    }

    /// Trigger a PB-MPM physics zone explosion in the CA simulation.
    fn ca_trigger_explosion(&mut self) {
        let center = self.player.camera.eye() + self.player.camera.forward() * SPAWN_DISTANCE;
        self.pending_explosion = Some([center.x, center.y, center.z]);
        tracing::info!("CA explosion at [{:.1}, {:.1}, {:.1}]", center.x, center.y, center.z);
    }

    /// Map selected_material index to a CA material ID.
    ///
    /// CA materials: 0=air, 1=stone, 2=sand, 3=water, 4=lava, 5=steam, 6=ice.
    fn ca_palette_material_id(&self) -> u16 {
        // Palette: 0=Stone(1), 1=Water(3), 2=Lava(4), 3=Sand(2), 4=Ice(6), 5=Wood(7)
        match self.selected_material {
            0 => 1,  // Stone
            1 => 3,  // Water
            2 => 4,  // Lava
            3 => 2,  // Sand
            4 => 6,  // Ice
            5 => 7,  // Wood (if exists)
            _ => 1,  // Default to stone
        }
    }

    /// Apply keyboard-driven movement to the player controller.
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
        if self.use_ca_sim {
            // Create CaSimulation (sim2)
            let ca_mats = server::ca_simulation::default_ca_materials();
            let ca_reactions = server::ca_simulation::default_ca_reactions();
            let mut ca_sim = match CaSimulation::new(&ctx, &ca_mats, &ca_reactions, 64) {
                Ok(s) => s,
                Err(e) => { tracing::error!("Failed to create CaSimulation: {}", e); event_loop.exit(); return; }
            };

            // Generate and load test scene
            let scene = server::ca_simulation::generate_test_scene();
            tracing::info!("Loading {} CA chunks for test scene", scene.len());
            for (coord, voxel_data) in &scene {
                if let Err(e) = ca_sim.load_chunk(&ctx, *coord, voxel_data, [-1; 6]) {
                    tracing::error!("Failed to load chunk {:?}: {}", coord, e);
                }
            }
            tracing::info!("CaSimulation initialized with {} chunks", ca_sim.loaded_count());

            // Place camera above and behind the terrain, looking at the center
            // Terrain center is at (64, ~40, 64), volcano peak around y=90
            self.player = PlayerController::new(Camera::look_at(
                Vec3::new(64.0, 80.0, -20.0),
                Vec3::new(64.0, 30.0, 64.0),
            ));

            self.ca_sim = Some(ca_sim);
        } else {
            let mut sim = match if let Some(db) = &self.material_db {
                let materials = db.material_params();
                let (transitions, count) = db.phase_transition_rules();
                GpuSimulation::new_with_materials(&ctx, &materials, &transitions, count)
            } else {
                GpuSimulation::new(&ctx)
            } {
                Ok(s) => s,
                Err(e) => { tracing::error!("Failed to create GpuSimulation: {}", e); event_loop.exit(); return; }
            };
            if let Err(e) = sim.init_particles(&ctx, &self.particles) {
                tracing::error!("Failed to upload particles: {}", e); event_loop.exit(); return;
            }
            tracing::info!("GpuSimulation initialized with {} particles", self.particles.len());
            self.sim = Some(sim);
        }

        self.window = Some(window);
        self.ctx = Some(ctx);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ctx) = self.ctx.as_ref() {
                    if let Some(sim) = self.sim.as_mut() { sim.destroy(ctx); }
                    if let Some(ca_sim) = self.ca_sim.take() { ca_sim.destroy(ctx); }
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
                            winit::keyboard::KeyCode::Digit6 if self.palette.len() > 5 => {
                                self.selected_material = 5;
                                tracing::info!("Selected: {}", self.palette[5].name);
                            }
                            winit::keyboard::KeyCode::Digit7 if self.palette.len() > 6 => {
                                self.selected_material = 6;
                                tracing::info!("Selected: {}", self.palette[6].name);
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.cursor_captured {
                    self.capture_cursor();
                } else if self.ca_sim.is_some() {
                    // CA simulation (--sim2) interactions
                    match button {
                        MouseButton::Left => self.ca_spawn_voxels(),
                        MouseButton::Right => self.ca_remove_voxels(),
                        MouseButton::Middle => self.ca_trigger_explosion(),
                        _ => {}
                    }
                } else {
                    // Standard MPM simulation interactions
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
                let run_react = self.should_run_react(now);

                // Detect camera movement since last frame.
                let eye_arr = [eye.x, eye.y, eye.z];
                let target_arr = [target.x, target.y, target.z];
                let camera_moved = self.prev_eye != eye_arr || self.prev_target != target_arr;
                self.prev_eye = eye_arr;
                self.prev_target = target_arr;

                // Explosions and spawning always dirty the world.
                let has_explosion = explosion.is_some();
                let need_render = camera_moved || !self.world_is_static || has_explosion;

                if let (Some(renderer), Some(ctx)) = (self.renderer.as_mut(), self.ctx.as_ref()) {
                    // --- CA simulation path (--sim2) ---
                    if let Some(ref mut ca_sim) = self.ca_sim {
                        // Handle pending explosion via PB-MPM zone activation
                        if let Some(explosion_pos) = explosion {
                            let trigger = ActivationTrigger {
                                center: explosion_pos,
                                radius: EXPLOSION_RADIUS,
                                impulse: EXPLOSION_STRENGTH,
                            };
                            if let Err(e) = ca_sim.activate_physics_zone(ctx, trigger) {
                                tracing::warn!("Failed to activate physics zone: {}", e);
                            }
                        }

                        if let Err(e) = ctx.execute_one_shot(|cmd| {
                            // CA needs many substeps: Margolus moves 1 voxel/step.
                            // At 128³ scale, need ~30 steps/frame to see movement.
                            let ca_substeps = substeps.max(10);
                            for _ in 0..ca_substeps { ca_sim.step(cmd, ctx); }
                            ca_sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye_arr, target_arr);
                            ca_sim.finalize_render(cmd);
                        }) {
                            tracing::error!("CA simulation/render error: {}", e);
                        }
                        // Debug physics: readback floating water cube chunk every second
                        // debug readback removed

                        match renderer.draw_frame_with_buffer(ctx, ca_sim.render_output_buffer(), RENDER_WIDTH, RENDER_HEIGHT) {
                            Ok(_) => {}
                            Err(e) => tracing::error!("Frame error: {}", e),
                        }
                    }
                    // --- Standard MPM simulation path ---
                    else if let Some(ref sim) = self.sim {
                        if let Err(e) = ctx.execute_one_shot(|cmd| {
                            for _ in 0..substeps { sim.step_physics_with_time(cmd, now); }
                            if run_react {
                                sim.step_react(cmd);
                            }
                            if let Some(center) = explosion {
                                sim.apply_explosion(cmd, center, EXPLOSION_RADIUS, EXPLOSION_STRENGTH);
                            }
                            if need_render {
                                sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye_arr, target_arr);
                                sim.render_toolbar(cmd, &toolbar_pc);
                                sim.finalize_render(cmd);
                            }
                        }) {
                            tracing::error!("Simulation/render error: {}", e);
                        }

                        // Readback the any_active flag AFTER GPU work completes.
                        if now.duration_since(self.last_oob_check).as_millis() >= 250 {
                            if let Ok(any_active) = sim.readback_any_active(ctx) {
                                self.world_is_static = !any_active;
                            }
                        }

                        if need_render {
                            if self.frames_skipped > 0 {
                                tracing::debug!("Render resumed after {} skipped frames", self.frames_skipped);
                                self.frames_skipped = 0;
                            }
                            match renderer.draw_frame_with_buffer(ctx, sim.render_output_buffer(), RENDER_WIDTH, RENDER_HEIGHT) {
                                Ok(_) => {}
                                Err(e) => tracing::error!("Frame error: {}", e),
                            }
                        } else {
                            self.frames_skipped += 1;
                            match renderer.draw_frame_with_buffer(ctx, sim.render_output_buffer(), RENDER_WIDTH, RENDER_HEIGHT) {
                                Ok(_) => {}
                                Err(e) => tracing::error!("Frame error: {}", e),
                            }
                        }
                    }
                }

                // --- Chunk streaming: check if camera crossed a chunk boundary ---
                if let Some(ref mut world) = self.world {
                    let cam_eye = self.player.camera.eye();
                    // Convert grid-space eye back to world-space for chunk lookup.
                    // Grid-space has an offset of sim_radius * CHUNK_SIZE in X/Z.
                    let offset_xz =
                        world.sim_radius() as f32 * shared::constants::CHUNK_SIZE as f32;
                    let world_x = cam_eye.x - offset_xz;
                    let world_z = cam_eye.z - offset_xz;
                    let cam_chunk = world::chunk::chunk_coord_for_position(world_x, world_z);

                    // TODO: async chunk streaming. For now, streaming is disabled
                    // because synchronous readback (144MB particle download) causes
                    // 50-100ms frame stutters every time the camera crosses a chunk
                    // boundary. Player stays in the initially-loaded 36 chunks.
                    if false && cam_chunk != self.current_chunk {
                        tracing::info!(
                            "Chunk boundary crossed: {:?} -> {:?}",
                            self.current_chunk,
                            cam_chunk,
                        );

                        // Read back current particles so we can save them to chunks
                        if let (Some(sim), Some(ctx)) =
                            (self.sim.as_ref(), self.ctx.as_ref())
                        {
                            match sim.readback_particles(ctx) {
                                Ok(mut gpu_particles) => {
                                    // Remove the grid offset before unpacking
                                    for p in &mut gpu_particles {
                                        let pos = p.position();
                                        p.set_position(Vec3::new(
                                            pos.x - offset_xz,
                                            pos.y,
                                            pos.z - offset_xz,
                                        ));
                                    }
                                    world.unpack_particles_from_gpu(&gpu_particles);
                                }
                                Err(e) => {
                                    tracing::error!("Failed to readback particles for chunk save: {}", e);
                                }
                            }
                        }

                        // Update center: evict old chunks, load new ones
                        if let Err(e) = world.update_center(cam_chunk) {
                            tracing::error!("Failed to update world center: {}", e);
                        }
                        self.current_chunk = cam_chunk;

                        // Pack new particle set with grid offset and upload
                        let mut new_particles = world.pack_particles_for_gpu();
                        for p in &mut new_particles {
                            let pos = p.position();
                            p.set_position(Vec3::new(
                                pos.x + offset_xz,
                                pos.y,
                                pos.z + offset_xz,
                            ));
                        }
                        if let (Some(sim), Some(ctx)) =
                            (self.sim.as_mut(), self.ctx.as_ref())
                        {
                            if let Err(e) = sim.reload_particles(ctx, &new_particles) {
                                tracing::error!("Failed to reload particles: {}", e);
                            } else {
                                tracing::info!(
                                    "Reloaded {} particles after chunk shift",
                                    new_particles.len(),
                                );
                            }
                        }
                    }
                }

                // --- OOB particle detection (throttled to avoid GPU stalls) ---
                if self.world.is_some() && now.duration_since(self.last_oob_check).as_millis() >= 500 {
                    self.last_oob_check = now;
                    if let (Some(sim), Some(ctx)) = (self.sim.as_ref(), self.ctx.as_ref()) {
                        if sim.readback_oob_flag(ctx).unwrap_or(false) {
                            tracing::warn!("Particles out of bounds detected");
                        }
                    }
                }

                // FPS tracking: count frames and update once per second.
                let frame_time_ms = dt as f64 * 1000.0;
                self.fps_frame_count += 1;
                let elapsed = self.fps_timer.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    self.current_fps = self.fps_frame_count as f32 / elapsed;
                    let info_str = if let Some(ref ca) = self.ca_sim {
                        format!(
                            "FPS: {:.0} | Frame: {:.1}ms | CA chunks: {}",
                            self.current_fps, frame_time_ms, ca.loaded_count(),
                        )
                    } else {
                        let num_particles = self.sim.as_ref().map_or(0, |s| s.num_particles());
                        format!(
                            "FPS: {:.0} | Frame: {:.1}ms | Particles: {}",
                            self.current_fps, frame_time_ms, num_particles,
                        )
                    };
                    tracing::info!("{}", info_str);
                    if let Some(window) = &self.window {
                        window.set_title(&format!("VOX | {}", info_str));
                    }
                    self.fps_frame_count = 0;
                    self.fps_timer = Instant::now();
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
