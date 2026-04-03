//! Types for GPU indirect dispatch.

use bytemuck::{Pod, Zeroable};

/// Arguments for `vkCmdDispatchIndirect`.
///
/// Matches `VkDispatchIndirectCommand` layout with padding to 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectDispatchArgs {
    /// Number of workgroups in X dimension.
    pub group_x: u32,
    /// Number of workgroups in Y dimension.
    pub group_y: u32,
    /// Number of workgroups in Z dimension.
    pub group_z: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn indirect_dispatch_args_layout() {
        assert_eq!(size_of::<IndirectDispatchArgs>(), 16);
        assert!(align_of::<IndirectDispatchArgs>() >= 4);
    }

    #[test]
    fn indirect_dispatch_args_is_zeroable() {
        let args = IndirectDispatchArgs::zeroed();
        assert_eq!(args.group_x, 0);
        assert_eq!(args.group_y, 0);
        assert_eq!(args.group_z, 0);
        assert_eq!(args._pad, 0);
    }
}
