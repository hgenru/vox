use spirv_builder::{Capability, SpirvBuilder};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shader_crate = PathBuf::from(manifest_dir).join("..").join("shaders");

    let result = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.3")
        .capability(Capability::AtomicFloat32AddEXT)
        .capability(Capability::VulkanMemoryModelDeviceScope)
        .extension("SPV_EXT_shader_atomic_float_add")
        .build()?;

    // All entry points are in a single SPIR-V module
    let spv_path = result.module.unwrap_single();
    println!("cargo:rustc-env=SHADERS_SPV_PATH={}", spv_path.display());

    Ok(())
}
