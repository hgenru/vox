//! CPU-side rigid body state tracking and shape matching for PB-MPM zones.
//!
//! Manages center-of-mass integration, shape matching constraint enforcement,
//! and fracture detection for rigid body zones.

/// Velocity magnitude change threshold for fracture detection.
const FRACTURE_THRESHOLD: f32 = 5.0;

/// Squared velocity threshold below which a zone is considered settled.
pub const SLEEP_VEL_THRESHOLD_SQ: f32 = 0.1;

/// Rigid body state for a single PB-MPM zone.
#[derive(Debug, Clone)]
pub struct RigidBodyState {
    /// Center of mass position (world-local to zone).
    pub com: [f32; 3],
    /// Center of mass velocity.
    pub velocity: [f32; 3],
    /// Previous frame velocity (for fracture detection).
    pub prev_velocity: [f32; 3],
    /// Rest positions relative to center of mass.
    pub rest_positions: Vec<[f32; 3]>,
    /// Whether this rigid body is active.
    pub active: bool,
    /// Consecutive sleep checks below threshold.
    pub sleep_count: u32,
}

impl RigidBodyState {
    /// Create a new rigid body state from voxel positions and center of mass.
    ///
    /// `voxel_positions` are in zone-local grid coordinates.
    /// `center_of_mass` is the computed CoM in zone-local coordinates.
    pub fn new(voxel_positions: &[[f32; 3]], center_of_mass: [f32; 3]) -> Self {
        let rest_positions = voxel_positions
            .iter()
            .map(|p| {
                [
                    p[0] - center_of_mass[0],
                    p[1] - center_of_mass[1],
                    p[2] - center_of_mass[2],
                ]
            })
            .collect();

        Self {
            com: center_of_mass,
            velocity: [0.0; 3],
            prev_velocity: [0.0; 3],
            rest_positions,
            active: true,
            sleep_count: 0,
        }
    }
}

/// Integrate rigid body center of mass under gravity.
///
/// Updates velocity with gravity and integrates position.
pub fn integrate(state: &mut RigidBodyState, dt: f32, gravity: f32) {
    state.prev_velocity = state.velocity;
    state.velocity[1] += gravity * dt;
    for i in 0..3 {
        state.com[i] += state.velocity[i] * dt;
    }
}

/// Apply shape matching constraint to particles.
///
/// For each particle, computes the target position as `com + rest_position`
/// and blends the current position toward it by `alpha`.
///
/// For POC: no rotation (translation only). Pieces fall straight down.
///
/// # Arguments
/// * `particles` — raw f32 particle buffer (24 f32s per particle)
/// * `particle_count` — number of particles
/// * `state` — rigid body state with current CoM
/// * `alpha` — blend factor (0.0 = no constraint, 1.0 = fully constrained)
pub fn apply_shape_matching(
    particles: &mut [f32],
    particle_count: u32,
    state: &RigidBodyState,
    alpha: f32,
) {
    let floats_per_particle = 24;

    for i in 0..particle_count.min(state.rest_positions.len() as u32) {
        let base = i as usize * floats_per_particle;
        let rest = &state.rest_positions[i as usize];

        // Target position = CoM + rest offset (translation only, no rotation)
        let target_x = state.com[0] + rest[0];
        let target_y = state.com[1] + rest[1];
        let target_z = state.com[2] + rest[2];

        // Lerp current position toward target
        let cur_x = particles[base];
        let cur_y = particles[base + 1];
        let cur_z = particles[base + 2];

        particles[base] = cur_x + (target_x - cur_x) * alpha;
        particles[base + 1] = cur_y + (target_y - cur_y) * alpha;
        particles[base + 2] = cur_z + (target_z - cur_z) * alpha;

        // Set velocity to match rigid body velocity
        particles[base + 4] = state.velocity[0];
        particles[base + 5] = state.velocity[1];
        particles[base + 6] = state.velocity[2];
    }
}

/// Check if a rigid body should fracture based on velocity change.
///
/// Returns `true` if the velocity magnitude change exceeds [`FRACTURE_THRESHOLD`].
pub fn check_fracture(state: &RigidBodyState) -> bool {
    let prev_mag_sq = state.prev_velocity[0] * state.prev_velocity[0]
        + state.prev_velocity[1] * state.prev_velocity[1]
        + state.prev_velocity[2] * state.prev_velocity[2];
    let cur_mag_sq = state.velocity[0] * state.velocity[0]
        + state.velocity[1] * state.velocity[1]
        + state.velocity[2] * state.velocity[2];
    let delta = (prev_mag_sq.sqrt() - cur_mag_sq.sqrt()).abs();
    delta > FRACTURE_THRESHOLD
}

/// Compute the maximum squared velocity from particle data.
///
/// `particles` is the raw f32 buffer (24 f32s per particle).
pub fn compute_max_velocity_sq(particles: &[f32], particle_count: u32) -> f32 {
    let floats_per_particle = 24;
    let mut max_vel_sq = 0.0f32;

    for i in 0..particle_count {
        let base = i as usize * floats_per_particle;
        if base + 6 >= particles.len() {
            break;
        }
        let vx = particles[base + 4];
        let vy = particles[base + 5];
        let vz = particles[base + 6];
        let vel_sq = vx * vx + vy * vy + vz * vz;
        if vel_sq > max_vel_sq {
            max_vel_sq = vel_sq;
        }
    }

    max_vel_sq
}

/// Compute center of mass from downloaded particle data.
///
/// Returns the average position of all particles.
pub fn compute_com_from_particles(particles: &[f32], particle_count: u32) -> [f32; 3] {
    let floats_per_particle = 24;
    let mut sum = [0.0f32; 3];
    let mut count = 0u32;

    for i in 0..particle_count {
        let base = i as usize * floats_per_particle;
        if base + 2 >= particles.len() {
            break;
        }
        let px = particles[base];
        let py = particles[base + 1];
        let pz = particles[base + 2];
        if px.is_finite() && py.is_finite() && pz.is_finite() {
            sum[0] += px;
            sum[1] += py;
            sum[2] += pz;
            count += 1;
        }
    }

    if count == 0 {
        return [0.0; 3];
    }

    [
        sum[0] / count as f32,
        sum[1] / count as f32,
        sum[2] / count as f32,
    ]
}

/// Convert rigid body particles to liquid debris after fracture.
///
/// Sets phase to 1 (liquid) and adds random impulse.
pub fn fracture_to_debris(particles: &mut [f32], particle_count: u32, base_impulse: f32) {
    let floats_per_particle = 24;

    for i in 0..particle_count {
        let base = i as usize * floats_per_particle;
        if base + 23 >= particles.len() {
            break;
        }

        // Set phase to liquid (1)
        particles[base + 21] = f32::from_bits(1u32);

        // Add random impulse based on position hash
        let px = particles[base].to_bits();
        let py = particles[base + 1].to_bits();
        let pz = particles[base + 2].to_bits();
        let hash = px.wrapping_mul(73856093)
            ^ py.wrapping_mul(19349663)
            ^ pz.wrapping_mul(83492791);
        let rand_x = ((hash & 0xFF) as f32 / 127.5) - 1.0;
        let rand_y = ((hash >> 8) & 0xFF) as f32 / 255.0;
        let rand_z = (((hash >> 16) & 0xFF) as f32 / 127.5) - 1.0;

        particles[base + 4] += rand_x * base_impulse;
        particles[base + 5] += rand_y * base_impulse * 0.5; // upward bias
        particles[base + 6] += rand_z * base_impulse;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigid_body_state_new() {
        let positions = vec![[10.0, 5.0, 10.0], [11.0, 5.0, 10.0], [12.0, 5.0, 10.0]];
        let com = [11.0, 5.0, 10.0];
        let state = RigidBodyState::new(&positions, com);

        assert_eq!(state.rest_positions.len(), 3);
        assert!((state.rest_positions[0][0] - (-1.0)).abs() < 0.001);
        assert!((state.rest_positions[1][0] - 0.0).abs() < 0.001);
        assert!((state.rest_positions[2][0] - 1.0).abs() < 0.001);
        assert!(state.active);
    }

    #[test]
    fn test_integrate() {
        let mut state = RigidBodyState {
            com: [0.0, 10.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            prev_velocity: [0.0, 0.0, 0.0],
            rest_positions: Vec::new(),
            active: true,
            sleep_count: 0,
        };

        let dt = 0.016;
        let gravity = -196.0;
        integrate(&mut state, dt, gravity);

        // Velocity should have been updated by gravity
        assert!((state.velocity[1] - gravity * dt).abs() < 0.001);
        // Position should have moved
        assert!(state.com[1] < 10.0);
    }

    #[test]
    fn test_apply_shape_matching() {
        let mut particles = vec![0.0f32; 24 * 3];
        // Set up 3 particles at known positions
        particles[0] = 10.0; // p0 x
        particles[1] = 5.0; // p0 y
        particles[2] = 10.0; // p0 z
        particles[3] = 1.0; // p0 mass
        particles[24] = 11.0; // p1 x
        particles[25] = 5.0; // p1 y
        particles[26] = 10.0; // p1 z
        particles[27] = 1.0; // p1 mass
        particles[48] = 12.0; // p2 x
        particles[49] = 5.0; // p2 y
        particles[50] = 10.0; // p2 z
        particles[51] = 1.0; // p2 mass

        let state = RigidBodyState {
            com: [11.0, 4.0, 10.0], // CoM moved down by 1
            velocity: [0.0, -1.0, 0.0],
            prev_velocity: [0.0, 0.0, 0.0],
            rest_positions: vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            active: true,
            sleep_count: 0,
        };

        apply_shape_matching(&mut particles, 3, &state, 1.0);

        // With alpha=1.0, particles should be exactly at target positions
        assert!((particles[0] - 10.0).abs() < 0.001); // p0 x
        assert!((particles[1] - 4.0).abs() < 0.001); // p0 y (moved to CoM.y)
        assert!((particles[24] - 11.0).abs() < 0.001); // p1 x
        assert!((particles[25] - 4.0).abs() < 0.001); // p1 y
    }

    #[test]
    fn test_check_fracture_no_impact() {
        let state = RigidBodyState {
            com: [0.0; 3],
            velocity: [0.0, -2.0, 0.0],
            prev_velocity: [0.0, -1.5, 0.0],
            rest_positions: Vec::new(),
            active: true,
            sleep_count: 0,
        };
        assert!(!check_fracture(&state));
    }

    #[test]
    fn test_check_fracture_on_impact() {
        let state = RigidBodyState {
            com: [0.0; 3],
            velocity: [0.0, 0.0, 0.0], // stopped
            prev_velocity: [0.0, -10.0, 0.0], // was falling fast
            rest_positions: Vec::new(),
            active: true,
            sleep_count: 0,
        };
        assert!(check_fracture(&state));
    }

    #[test]
    fn test_compute_max_velocity_sq() {
        let mut particles = vec![0.0f32; 24 * 2];
        // Particle 0: vel = (1, 2, 3) → sq = 14
        particles[4] = 1.0;
        particles[5] = 2.0;
        particles[6] = 3.0;
        // Particle 1: vel = (0, 5, 0) → sq = 25
        particles[28] = 0.0;
        particles[29] = 5.0;
        particles[30] = 0.0;

        let max = compute_max_velocity_sq(&particles, 2);
        assert!((max - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_com_from_particles() {
        let mut particles = vec![0.0f32; 24 * 2];
        particles[0] = 10.0;
        particles[1] = 5.0;
        particles[2] = 10.0;
        particles[24] = 12.0;
        particles[25] = 5.0;
        particles[26] = 10.0;

        let com = compute_com_from_particles(&particles, 2);
        assert!((com[0] - 11.0).abs() < 0.001);
        assert!((com[1] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_fracture_to_debris() {
        let mut particles = vec![0.0f32; 24 * 2];
        particles[0] = 10.0;
        particles[1] = 5.0;
        particles[2] = 10.0;
        particles[3] = 1.0;
        // ids.y (phase) at offset 21
        particles[21] = f32::from_bits(0u32); // solid

        particles[24] = 11.0;
        particles[25] = 5.0;
        particles[26] = 10.0;
        particles[27] = 1.0;
        particles[45] = f32::from_bits(0u32); // solid

        fracture_to_debris(&mut particles, 2, 5.0);

        // Phase should be liquid (1)
        assert_eq!(particles[21].to_bits(), 1u32);
        assert_eq!(particles[45].to_bits(), 1u32);
    }
}
