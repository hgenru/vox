//! Player controller with gravity and ground collision for walk mode.
//!
//! Wraps [`Camera`] to provide two movement modes:
//! - **Fly mode** (default): free-fly FPS camera, no gravity.
//! - **Walk mode**: gravity pulls the player down, WASD moves horizontally,
//!   Space jumps when grounded, and the player collides with a CPU-side heightmap.

use glam::Vec3;
use winit::keyboard::KeyCode;

use crate::camera::Camera;

/// Default eye height above the ground surface in world units.
const DEFAULT_PLAYER_HEIGHT: f32 = 1.8;

/// Default jump speed (initial upward velocity) in units per second.
const DEFAULT_JUMP_SPEED: f32 = 6.0;

/// Gravitational acceleration in units per second squared.
const GRAVITY: f32 = 9.81;

/// Player controller that wraps a [`Camera`] with optional gravity and ground collision.
///
/// In fly mode, behaves identically to the underlying camera.
/// In walk mode, applies gravity, horizontal-only WASD movement, jumping, and
/// heightmap-based ground collision.
pub struct PlayerController {
    /// The underlying FPS camera.
    pub camera: Camera,
    /// Vertical velocity (positive = upward) used in walk mode.
    pub velocity_y: f32,
    /// Whether the player is currently standing on the ground.
    pub on_ground: bool,
    /// If true, the camera uses free-fly mode; if false, walk mode with gravity.
    pub fly_mode: bool,
    /// Eye height above the ground surface in world units.
    pub player_height: f32,
    /// Jump speed (initial upward velocity) in units per second.
    pub jump_speed: f32,
    /// CPU-side heightmap for ground collision, stored row-major (Z * grid_size + X).
    heightmap: Option<Vec<f32>>,
    /// Side length of the square heightmap grid.
    heightmap_grid_size: u32,
}

impl PlayerController {
    /// Create a new player controller wrapping the given camera.
    ///
    /// Starts in fly mode with no heightmap. Call [`toggle_fly_mode`] to switch
    /// to walk mode, and [`set_heightmap`] to provide ground collision data.
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            velocity_y: 0.0,
            on_ground: false,
            fly_mode: false,
            player_height: DEFAULT_PLAYER_HEIGHT,
            jump_speed: DEFAULT_JUMP_SPEED,
            heightmap: None,
            heightmap_grid_size: 0,
        }
    }

    /// Toggle between fly mode and walk mode.
    ///
    /// When switching to walk mode, vertical velocity is reset to zero.
    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        if !self.fly_mode {
            self.velocity_y = 0.0;
            self.on_ground = false;
            tracing::info!("Switched to walk mode");
        } else {
            tracing::info!("Switched to fly mode");
        }
    }

    /// Set the CPU-side heightmap used for ground collision in walk mode.
    ///
    /// `heightmap` is a row-major flat array of ground Y values, indexed as
    /// `heightmap[z * grid_size + x]`. `grid_size` is the side length of the
    /// square grid (e.g., 32 for a 32x32 heightmap).
    ///
    /// Returns an error string if the heightmap length does not match `grid_size * grid_size`.
    pub fn set_heightmap(&mut self, heightmap: Vec<f32>, grid_size: u32) -> Result<(), String> {
        let expected = (grid_size as usize) * (grid_size as usize);
        if heightmap.len() != expected {
            return Err(format!(
                "Heightmap length {} does not match grid_size^2 = {}",
                heightmap.len(),
                expected
            ));
        }
        self.heightmap = Some(heightmap);
        self.heightmap_grid_size = grid_size;
        Ok(())
    }

    /// Sample the heightmap at a world XZ position using bilinear interpolation.
    ///
    /// Returns `None` if no heightmap is set or the position is outside the grid bounds.
    pub fn sample_heightmap(&self, x: f32, z: f32) -> Option<f32> {
        let hm = self.heightmap.as_ref()?;
        let size = self.heightmap_grid_size;
        if size == 0 {
            return None;
        }

        // Clamp to valid range [0, size-1]
        let max_coord = (size - 1) as f32;
        let fx = x.clamp(0.0, max_coord);
        let fz = z.clamp(0.0, max_coord);

        let x0 = fx.floor() as u32;
        let z0 = fz.floor() as u32;
        let x1 = (x0 + 1).min(size - 1);
        let z1 = (z0 + 1).min(size - 1);

        let tx = fx - fx.floor();
        let tz = fz - fz.floor();

        let idx = |xi: u32, zi: u32| -> usize { (zi * size + xi) as usize };

        let h00 = hm[idx(x0, z0)];
        let h10 = hm[idx(x1, z0)];
        let h01 = hm[idx(x0, z1)];
        let h11 = hm[idx(x1, z1)];

        // Bilinear interpolation
        let h = h00 * (1.0 - tx) * (1.0 - tz)
            + h10 * tx * (1.0 - tz)
            + h01 * (1.0 - tx) * tz
            + h11 * tx * tz;

        Some(h)
    }

    /// Process a keyboard key press for movement.
    ///
    /// In fly mode, delegates directly to [`Camera::process_keyboard`].
    /// In walk mode, WASD moves horizontally (ignoring pitch), Space triggers
    /// a jump if grounded, and Shift is reserved for future use (crouch).
    pub fn process_keyboard(&mut self, key: KeyCode, dt: f32) {
        if self.fly_mode {
            self.camera.process_keyboard(key, dt);
            return;
        }

        // Walk mode: horizontal movement only
        let velocity = self.camera.speed * dt;
        let yaw = self.camera.yaw;

        // Horizontal forward/right vectors (pitch ignored)
        let forward_h = Vec3::new(yaw.sin(), 0.0, -yaw.cos()).normalize();
        let right_h = Vec3::new(yaw.cos(), 0.0, yaw.sin()).normalize();

        match key {
            KeyCode::KeyW => self.camera.position += forward_h * velocity,
            KeyCode::KeyS => self.camera.position -= forward_h * velocity,
            KeyCode::KeyA => self.camera.position -= right_h * velocity,
            KeyCode::KeyD => self.camera.position += right_h * velocity,
            KeyCode::Space => {
                if self.on_ground {
                    self.velocity_y = self.jump_speed;
                    self.on_ground = false;
                }
            }
            // Shift reserved for future crouch
            KeyCode::ShiftLeft => {}
            _ => {}
        }
    }

    /// Process raw mouse delta (delegates to the underlying camera).
    ///
    /// `dx` and `dy` are pixel deltas from the mouse device.
    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        self.camera.process_mouse(dx, dy);
    }

    /// Update physics for walk mode: apply gravity, move vertically, collide with ground.
    ///
    /// Call this once per frame with the frame delta time `dt` in seconds.
    /// In fly mode, this is a no-op.
    pub fn update(&mut self, dt: f32) {
        if self.fly_mode {
            return;
        }

        // Apply gravity
        self.velocity_y -= GRAVITY * dt;

        // Move vertically
        self.camera.position.y += self.velocity_y * dt;

        // Ground collision
        let ground_y = self.sample_heightmap(self.camera.position.x, self.camera.position.z);
        if let Some(gy) = ground_y {
            let min_y = gy + self.player_height;
            if self.camera.position.y < min_y {
                self.camera.position.y = min_y;
                self.velocity_y = 0.0;
                self.on_ground = true;
            } else {
                self.on_ground = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> PlayerController {
        let camera = Camera::new(Vec3::new(5.0, 10.0, 5.0), 0.0, 0.0);
        PlayerController::new(camera)
    }

    fn flat_heightmap(size: u32, height: f32) -> Vec<f32> {
        vec![height; (size * size) as usize]
    }

    #[test]
    fn test_new_defaults() {
        let pc = make_controller();
        assert!(pc.fly_mode);
        assert!(!pc.on_ground);
        assert_eq!(pc.velocity_y, 0.0);
        assert_eq!(pc.player_height, DEFAULT_PLAYER_HEIGHT);
        assert_eq!(pc.jump_speed, DEFAULT_JUMP_SPEED);
        assert!(pc.heightmap.is_none());
    }

    #[test]
    fn test_toggle_fly_mode() {
        let mut pc = make_controller();
        assert!(pc.fly_mode);
        pc.toggle_fly_mode();
        assert!(!pc.fly_mode);
        pc.toggle_fly_mode();
        assert!(pc.fly_mode);
    }

    #[test]
    fn test_toggle_resets_velocity() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        pc.velocity_y = 5.0;
        // Switch to fly mode and back to walk
        pc.toggle_fly_mode(); // -> fly
        pc.toggle_fly_mode(); // -> walk, velocity should reset
        assert_eq!(pc.velocity_y, 0.0);
    }

    #[test]
    fn test_set_heightmap_valid() {
        let mut pc = make_controller();
        let hm = flat_heightmap(4, 0.0);
        assert!(pc.set_heightmap(hm, 4).is_ok());
        assert!(pc.heightmap.is_some());
        assert_eq!(pc.heightmap_grid_size, 4);
    }

    #[test]
    fn test_set_heightmap_invalid_size() {
        let mut pc = make_controller();
        let hm = vec![0.0; 10]; // not 4*4=16
        assert!(pc.set_heightmap(hm, 4).is_err());
    }

    #[test]
    fn test_sample_heightmap_no_map() {
        let pc = make_controller();
        assert!(pc.sample_heightmap(0.0, 0.0).is_none());
    }

    #[test]
    fn test_sample_heightmap_flat() {
        let mut pc = make_controller();
        pc.set_heightmap(flat_heightmap(8, 5.0), 8).unwrap();
        let h = pc.sample_heightmap(3.5, 4.2);
        assert!((h.unwrap() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_heightmap_corners() {
        let mut pc = make_controller();
        // 2x2 heightmap with distinct corners
        let hm = vec![0.0, 1.0, 2.0, 3.0];
        pc.set_heightmap(hm, 2).unwrap();

        assert!((pc.sample_heightmap(0.0, 0.0).unwrap() - 0.0).abs() < 1e-5);
        assert!((pc.sample_heightmap(1.0, 0.0).unwrap() - 1.0).abs() < 1e-5);
        assert!((pc.sample_heightmap(0.0, 1.0).unwrap() - 2.0).abs() < 1e-5);
        assert!((pc.sample_heightmap(1.0, 1.0).unwrap() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_heightmap_bilinear() {
        let mut pc = make_controller();
        // 2x2: top-left=0, top-right=4, bottom-left=0, bottom-right=4
        let hm = vec![0.0, 4.0, 0.0, 4.0];
        pc.set_heightmap(hm, 2).unwrap();

        // Midpoint along X at Z=0 should be 2.0
        let h = pc.sample_heightmap(0.5, 0.0).unwrap();
        assert!((h - 2.0).abs() < 1e-5, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_sample_heightmap_clamps_out_of_bounds() {
        let mut pc = make_controller();
        pc.set_heightmap(flat_heightmap(4, 3.0), 4).unwrap();

        // Negative coords should clamp to edge
        let h = pc.sample_heightmap(-5.0, -5.0).unwrap();
        assert!((h - 3.0).abs() < 1e-5);

        // Beyond grid should clamp to edge
        let h = pc.sample_heightmap(100.0, 100.0).unwrap();
        assert!((h - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_fly_mode_delegates_keyboard() {
        let mut pc = make_controller();
        assert!(pc.fly_mode);
        let pos_before = pc.camera.position;
        pc.process_keyboard(KeyCode::KeyW, 1.0);
        // In fly mode, W should move along forward (which includes pitch)
        assert_ne!(pc.camera.position, pos_before);
    }

    #[test]
    fn test_walk_mode_horizontal_movement() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        // yaw=0 means forward is -Z
        let y_before = pc.camera.position.y;
        pc.process_keyboard(KeyCode::KeyW, 1.0);
        // Y should not change in walk mode from WASD
        assert_eq!(pc.camera.position.y, y_before);
        // Z should decrease (forward is -Z at yaw=0)
        assert!(pc.camera.position.z < 5.0);
    }

    #[test]
    fn test_walk_mode_jump() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        pc.on_ground = true;
        pc.process_keyboard(KeyCode::Space, 1.0);
        assert_eq!(pc.velocity_y, DEFAULT_JUMP_SPEED);
        assert!(!pc.on_ground);
    }

    #[test]
    fn test_walk_mode_no_jump_midair() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        pc.on_ground = false;
        pc.process_keyboard(KeyCode::Space, 1.0);
        assert_eq!(pc.velocity_y, 0.0);
    }

    #[test]
    fn test_update_fly_mode_noop() {
        let mut pc = make_controller();
        assert!(pc.fly_mode);
        let pos_before = pc.camera.position;
        pc.update(1.0);
        assert_eq!(pc.camera.position, pos_before);
    }

    #[test]
    fn test_update_gravity_pulls_down() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        // No heightmap, so no ground collision, just gravity
        let y_before = pc.camera.position.y;
        pc.update(0.1);
        assert!(pc.camera.position.y < y_before);
        assert!(pc.velocity_y < 0.0);
    }

    #[test]
    fn test_update_ground_collision() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        pc.set_heightmap(flat_heightmap(32, 0.0), 32).unwrap();
        // Place player just above ground
        pc.camera.position = Vec3::new(5.0, DEFAULT_PLAYER_HEIGHT + 0.01, 5.0);
        pc.velocity_y = -5.0; // falling

        pc.update(1.0); // large dt to ensure we hit ground

        assert!(pc.on_ground);
        assert_eq!(pc.velocity_y, 0.0);
        assert!((pc.camera.position.y - DEFAULT_PLAYER_HEIGHT).abs() < 1e-5);
    }

    #[test]
    fn test_update_respects_heightmap_elevation() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        // Heightmap with ground at y=5.0
        pc.set_heightmap(flat_heightmap(32, 5.0), 32).unwrap();
        pc.camera.position = Vec3::new(5.0, 5.0 + DEFAULT_PLAYER_HEIGHT + 0.01, 5.0);
        pc.velocity_y = -10.0;

        pc.update(1.0);

        assert!(pc.on_ground);
        assert!((pc.camera.position.y - (5.0 + DEFAULT_PLAYER_HEIGHT)).abs() < 1e-5);
    }

    #[test]
    fn test_process_mouse_delegates() {
        let mut pc = make_controller();
        let yaw_before = pc.camera.yaw;
        pc.process_mouse(100.0, 0.0);
        assert_ne!(pc.camera.yaw, yaw_before);
    }

    #[test]
    fn test_walk_mode_shift_does_nothing() {
        let mut pc = make_controller();
        pc.fly_mode = false;
        let pos_before = pc.camera.position;
        pc.process_keyboard(KeyCode::ShiftLeft, 1.0);
        assert_eq!(pc.camera.position, pos_before);
    }
}
