use std::path::PathBuf;

use spirv_builder::{Capability, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shader_crate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("shaders");

    let result = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.3")
        .capability(Capability::AtomicFloat32AddEXT)
        .capability(Capability::VulkanMemoryModelDeviceScope)
        .extension("SPV_EXT_shader_atomic_float_add")
        .build()?;

    let spv_path = result.module.unwrap_single();
    println!("cargo:rustc-env=SHADERS_SPV_PATH={}", spv_path.display());

    Ok(())
}
