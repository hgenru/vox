//! # protocol
//!
//! Server↔Client communication contract.
//! Defines `WorldSnapshot` (server→client) and `PlayerInput` (client→server).
//! Uses `bitcode` for fast binary serialization.
//! Designed so channel can be swapped for `quinn` QUIC later.

use bitcode::{Decode, Encode};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A compact particle state for network/IPC transfer.
///
/// Contains only the fields needed for rendering, not the full simulation state.
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq)]
pub struct ParticleState {
    /// Position xyz.
    pub position: [f32; 3],
    /// Material ID.
    pub material_id: u32,
    /// Phase (0=solid, 1=liquid, 2=gas).
    pub phase: u32,
    /// Temperature.
    pub temperature: f32,
}

impl ParticleState {
    /// Create a new particle state.
    pub fn new(position: Vec3, material_id: u32, phase: u32, temperature: f32) -> Self {
        Self {
            position: [position.x, position.y, position.z],
            material_id,
            phase,
            temperature,
        }
    }

    /// Get position as Vec3.
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.position[0], self.position[1], self.position[2])
    }
}

/// Server -> Client: snapshot of the world state.
///
/// Sent every frame (or every N frames) from the simulation server to the renderer.
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// All particle states.
    pub particles: Vec<ParticleState>,
    /// Grid occupancy bitmap (1 bit per cell, packed into u32s).
    /// Used for acceleration structure updates.
    pub grid_occupancy: Vec<u32>,
    /// Current frame number.
    pub frame: u64,
    /// Current simulation time in seconds.
    pub sim_time: f32,
    /// Grid dimension (cells per axis).
    pub grid_size: u32,
}

impl WorldSnapshot {
    /// Create a new empty snapshot.
    pub fn new(grid_size: u32) -> Self {
        let occupancy_words = (grid_size * grid_size * grid_size + 31) / 32;
        Self {
            particles: Vec::new(),
            grid_occupancy: vec![0; occupancy_words as usize],
            frame: 0,
            sim_time: 0.0,
            grid_size,
        }
    }

    /// Number of particles in the snapshot.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

/// Mouse button action.
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq, Eq)]
pub enum MouseAction {
    /// No mouse action.
    None,
    /// Spawn material at cursor position.
    Spawn,
    /// Remove material at cursor position.
    Remove,
}

/// Client -> Server: player input for the current frame.
///
/// Sent every frame from the client to the simulation.
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize)]
pub struct PlayerInput {
    /// Camera position xyz.
    pub camera_pos: [f32; 3],
    /// Camera look direction (unit vector) xyz.
    pub camera_dir: [f32; 3],
    /// Mouse action (spawn/remove/none).
    pub mouse_action: MouseAction,
    /// Material to spawn (only used when mouse_action = Spawn).
    pub spawn_material: u32,
    /// Keyboard state bitfield:
    /// bit 0: W (forward), bit 1: S (backward),
    /// bit 2: A (left), bit 3: D (right),
    /// bit 4: Space (up), bit 5: Shift (down)
    pub keyboard_state: u32,
    /// Frame number this input corresponds to.
    pub frame: u64,
}

impl PlayerInput {
    /// Create a default (no-op) input.
    pub fn idle() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 0.0],
            camera_dir: [0.0, 0.0, -1.0],
            mouse_action: MouseAction::None,
            spawn_material: 0,
            keyboard_state: 0,
            frame: 0,
        }
    }

    /// Get camera position as Vec3.
    pub fn camera_position(&self) -> Vec3 {
        Vec3::new(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
    }

    /// Get camera direction as Vec3.
    pub fn camera_direction(&self) -> Vec3 {
        Vec3::new(self.camera_dir[0], self.camera_dir[1], self.camera_dir[2])
    }

    /// Check if a specific key is pressed.
    ///
    /// Key indices: 0=W, 1=S, 2=A, 3=D, 4=Space, 5=Shift
    pub fn key_pressed(&self, key_index: u32) -> bool {
        (self.keyboard_state >> key_index) & 1 == 1
    }
}

/// Convert a full simulation Particle to a compact ParticleState for protocol.
///
/// Extracts only the fields needed for rendering.
pub fn particle_to_state(particle: &shared::particle::Particle) -> ParticleState {
    ParticleState {
        position: [
            particle.position().x,
            particle.position().y,
            particle.position().z,
        ],
        material_id: particle.material_id(),
        phase: particle.phase(),
        temperature: particle.temperature(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::material::{MAT_WATER, PHASE_LIQUID};

    #[test]
    fn world_snapshot_bitcode_roundtrip() {
        let mut snapshot = WorldSnapshot::new(32);
        snapshot.particles.push(ParticleState::new(
            Vec3::new(0.5, 0.6, 0.7),
            MAT_WATER,
            PHASE_LIQUID,
            25.0,
        ));
        snapshot.frame = 42;
        snapshot.sim_time = 0.042;

        let encoded = bitcode::encode(&snapshot);
        let decoded: WorldSnapshot = bitcode::decode(&encoded).unwrap();

        assert_eq!(decoded.frame, 42);
        assert_eq!(decoded.particles.len(), 1);
        assert_eq!(decoded.particles[0].material_id, MAT_WATER);
        assert!((decoded.sim_time - 0.042).abs() < 1e-6);
    }

    #[test]
    fn player_input_bitcode_roundtrip() {
        let input = PlayerInput {
            camera_pos: [1.0, 2.0, 3.0],
            camera_dir: [0.0, 0.0, -1.0],
            mouse_action: MouseAction::Spawn,
            spawn_material: MAT_WATER,
            keyboard_state: 0b000101, // W + A
            frame: 100,
        };

        let encoded = bitcode::encode(&input);
        let decoded: PlayerInput = bitcode::decode(&encoded).unwrap();

        assert_eq!(decoded.frame, 100);
        assert_eq!(decoded.mouse_action, MouseAction::Spawn);
        assert_eq!(decoded.spawn_material, MAT_WATER);
        assert!(decoded.key_pressed(0)); // W
        assert!(!decoded.key_pressed(1)); // not S
        assert!(decoded.key_pressed(2)); // A
    }

    #[test]
    fn particle_state_position_roundtrip() {
        let state = ParticleState::new(Vec3::new(0.123, 0.456, 0.789), 0, 0, 100.0);
        let pos = state.position();
        assert!((pos.x - 0.123).abs() < 1e-6);
        assert!((pos.y - 0.456).abs() < 1e-6);
        assert!((pos.z - 0.789).abs() < 1e-6);
    }

    #[test]
    fn empty_snapshot_roundtrip() {
        let snapshot = WorldSnapshot::new(32);
        let encoded = bitcode::encode(&snapshot);
        let decoded: WorldSnapshot = bitcode::decode(&encoded).unwrap();
        assert_eq!(decoded.particle_count(), 0);
        assert_eq!(decoded.grid_size, 32);
    }

    #[test]
    fn large_snapshot_roundtrip() {
        let mut snapshot = WorldSnapshot::new(32);
        for i in 0..1000 {
            snapshot.particles.push(ParticleState::new(
                Vec3::new(i as f32 * 0.001, 0.5, 0.5),
                MAT_WATER,
                PHASE_LIQUID,
                20.0,
            ));
        }
        snapshot.frame = 999;

        let encoded = bitcode::encode(&snapshot);
        let decoded: WorldSnapshot = bitcode::decode(&encoded).unwrap();

        assert_eq!(decoded.particles.len(), 1000);
        assert_eq!(decoded.frame, 999);
    }

    #[test]
    fn player_input_idle_defaults() {
        let input = PlayerInput::idle();
        assert_eq!(input.mouse_action, MouseAction::None);
        assert_eq!(input.keyboard_state, 0);
        assert_eq!(input.frame, 0);
    }

    #[test]
    fn particle_to_state_conversion() {
        let p = shared::particle::Particle::new(Vec3::new(0.1, 0.2, 0.3), 1.0, MAT_WATER, PHASE_LIQUID);
        let state = particle_to_state(&p);
        assert_eq!(state.material_id, MAT_WATER);
        assert_eq!(state.phase, PHASE_LIQUID);
        assert!((state.position[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn keyboard_state_bits() {
        let input = PlayerInput {
            keyboard_state: 0b110011,
            ..PlayerInput::idle()
        };
        assert!(input.key_pressed(0));  // W
        assert!(input.key_pressed(1));  // S
        assert!(!input.key_pressed(2)); // not A
        assert!(!input.key_pressed(3)); // not D
        assert!(input.key_pressed(4));  // Space
        assert!(input.key_pressed(5));  // Shift
    }
}
