//! FPS camera with WASD movement and mouse look.
//!
//! Provides a first-person camera that tracks position, orientation (yaw/pitch),
//! and projection parameters. Integrates with winit key codes for movement input
//! and raw mouse deltas for look control.

use glam::Vec3;
use winit::keyboard::KeyCode;

/// An FPS-style camera with position, orientation, and projection parameters.
///
/// Yaw 0 looks along -Z (into the screen). Pitch is clamped to +/- 89 degrees
/// to avoid gimbal lock at the poles.
pub struct Camera {
    /// World-space position of the camera eye.
    pub position: Vec3,
    /// Horizontal rotation in radians. 0 = looking along -Z.
    pub yaw: f32,
    /// Vertical rotation in radians, clamped to [-89deg, 89deg].
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Movement speed in units per second.
    pub speed: f32,
    /// Mouse sensitivity (radians per pixel of mouse delta).
    pub sensitivity: f32,
}

impl Camera {
    /// Create a new FPS camera at the given position and orientation.
    ///
    /// Uses default values: FOV = 60 degrees, speed = 10 units/s,
    /// sensitivity = 0.003 rad/pixel, aspect = 16/9.
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            fov: 60.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            speed: 10.0,
            sensitivity: 0.003,
        }
    }

    /// Return the forward direction vector (unit length) based on yaw and pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -(self.yaw.cos()) * self.pitch.cos(),
        )
        .normalize()
    }

    /// Return the right direction vector (unit length), perpendicular to forward on the XZ plane.
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    /// Process a keyboard key press for movement.
    ///
    /// W/S move forward/backward, A/D strafe left/right,
    /// Space moves up, LShift moves down. Movement is scaled by `dt` (seconds).
    pub fn process_keyboard(&mut self, key: KeyCode, dt: f32) {
        let velocity = self.speed * dt;
        let forward = self.forward();
        let right = self.right();
        let up = Vec3::Y;

        match key {
            KeyCode::KeyW => self.position += forward * velocity,
            KeyCode::KeyS => self.position -= forward * velocity,
            KeyCode::KeyA => self.position -= right * velocity,
            KeyCode::KeyD => self.position += right * velocity,
            KeyCode::Space => self.position += up * velocity,
            KeyCode::ShiftLeft => self.position -= up * velocity,
            _ => {}
        }
    }

    /// Process raw mouse delta to update yaw and pitch.
    ///
    /// `dx` and `dy` are pixel deltas from the mouse device.
    /// Pitch is clamped to +/- 89 degrees to prevent flipping.
    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.sensitivity;
        self.pitch -= dy * self.sensitivity;

        let max_pitch = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Return the camera eye position (same as `position`).
    pub fn eye(&self) -> Vec3 {
        self.position
    }

    /// Return the point the camera is looking at (`position + forward`).
    pub fn target(&self) -> Vec3 {
        self.position + self.forward()
    }

    /// Update the aspect ratio from window dimensions.
    ///
    /// Does nothing if `height` is zero to avoid division by zero.
    pub fn update_aspect(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_new_defaults() {
        let cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        assert!((cam.fov - 60.0_f32.to_radians()).abs() < 1e-6);
        assert_eq!(cam.speed, 10.0);
        assert_eq!(cam.sensitivity, 0.003);
    }

    #[test]
    fn test_forward_at_zero_yaw() {
        let cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        let fwd = cam.forward();
        // yaw=0, pitch=0 => looking along -Z
        assert!(fwd.x.abs() < 1e-5);
        assert!(fwd.y.abs() < 1e-5);
        assert!((fwd.z - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_forward_at_half_pi_yaw() {
        let cam = Camera::new(Vec3::ZERO, PI / 2.0, 0.0);
        let fwd = cam.forward();
        // yaw=PI/2 => looking along +X
        assert!((fwd.x - 1.0).abs() < 1e-5);
        assert!(fwd.y.abs() < 1e-5);
        assert!(fwd.z.abs() < 1e-5);
    }

    #[test]
    fn test_right_at_zero_yaw() {
        let cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        let r = cam.right();
        // yaw=0 => right is +X
        assert!((r.x - 1.0).abs() < 1e-5);
        assert!(r.y.abs() < 1e-5);
        assert!(r.z.abs() < 1e-5);
    }

    #[test]
    fn test_pitch_clamp() {
        let mut cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        // Large positive dy should clamp pitch
        cam.process_mouse(0.0, -100000.0);
        let max_pitch = 89.0_f32.to_radians();
        assert!((cam.pitch - max_pitch).abs() < 1e-5);

        cam.process_mouse(0.0, 200000.0);
        assert!((cam.pitch - (-max_pitch)).abs() < 1e-5);
    }

    #[test]
    fn test_target_equals_position_plus_forward() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0), 0.5, 0.2);
        let target = cam.target();
        let expected = cam.position + cam.forward();
        assert!((target - expected).length() < 1e-5);
    }

    #[test]
    fn test_update_aspect() {
        let mut cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        cam.update_aspect(1920, 1080);
        assert!((cam.aspect - 1920.0 / 1080.0).abs() < 1e-5);
    }

    #[test]
    fn test_update_aspect_zero_height() {
        let mut cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        let original = cam.aspect;
        cam.update_aspect(1920, 0);
        assert_eq!(cam.aspect, original);
    }

    #[test]
    fn test_process_keyboard_forward() {
        let mut cam = Camera::new(Vec3::ZERO, 0.0, 0.0);
        let pos_before = cam.position;
        cam.process_keyboard(KeyCode::KeyW, 1.0);
        // Should have moved along forward (-Z)
        assert!(cam.position.z < pos_before.z);
    }

    #[test]
    fn test_eye() {
        let cam = Camera::new(Vec3::new(5.0, 6.0, 7.0), 0.0, 0.0);
        assert_eq!(cam.eye(), cam.position);
    }
}
