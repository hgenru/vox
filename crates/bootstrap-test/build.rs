use std::path::PathBuf;

use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shader_crate = PathBuf::from(manifest_dir)
        .join("..")
        .join("bootstrap-shader");

    let result = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.3").build()?;

    let spv_path = result.module.unwrap_single();
    println!("cargo:rustc-env=BOOTSTRAP_SPV_PATH={}", spv_path.display());

    Ok(())
}
