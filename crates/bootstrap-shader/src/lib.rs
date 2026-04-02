#![cfg_attr(target_arch = "spirv", no_std)]

use glam::UVec3;
use spirv_std::{glam, spirv};

/// Double the value - must be a separate function (rust-gpu linker requires at least one
/// function call in the entry point, otherwise the entry point gets dropped during SPIR-V linking).
pub fn double_val(n: u32) -> u32 {
    n * 2
}

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
) {
    let index = id.x as usize;
    data[index] = double_val(data[index]);
}
