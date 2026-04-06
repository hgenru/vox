//! Chemical reaction entries for the CA substrate.
//!
//! Each [`ReactionEntry`] describes a pairwise material reaction checked
//! during the CA step when two adjacent voxels meet the conditions.

use bytemuck::{Pod, Zeroable};

/// Chemical reaction entry for CA substrate. 32 bytes.
///
/// When voxel A (material `input_a`) is adjacent to voxel B (material `input_b`)
/// and the `condition` is met, both are replaced with `output_a` / `output_b`
/// and `heat_delta` is applied. `probability` controls stochastic firing (0-255).
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ReactionEntry {
    /// Material ID of the first input.
    pub input_a: u32,
    /// Material ID of the second input.
    pub input_b: u32,
    /// Material ID that replaces `input_a`.
    pub output_a: u32,
    /// Material ID that replaces `input_b`.
    pub output_b: u32,
    /// Temperature change applied on reaction (can be negative).
    pub heat_delta: i32,
    /// Condition: 0=none, 1=needs_oxygen, 2=needs_high_temp.
    pub condition: u32,
    /// Probability of firing per tick (0-255).
    pub probability: u32,
    /// Impulse strength: 0=none, 1=weak, 2=medium, 3=explosion.
    pub impulse: u32,
}

// Compile-time size check
const _: () = assert!(core::mem::size_of::<ReactionEntry>() == 32);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_size() {
        assert_eq!(size_of::<ReactionEntry>(), 32);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let r: ReactionEntry = ReactionEntry::zeroed();
        assert_eq!(r.input_a, 0);
        assert_eq!(r.input_b, 0);
        assert_eq!(r.output_a, 0);
        assert_eq!(r.output_b, 0);
        assert_eq!(r.heat_delta, 0);
        assert_eq!(r.condition, 0);
        assert_eq!(r.probability, 0);
        assert_eq!(r.impulse, 0);
    }
}
