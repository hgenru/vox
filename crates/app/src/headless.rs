//! Headless (no-window) execution mode.
//!
//! Runs the simulation for a fixed number of frames and saves a screenshot.

use anyhow::Result;
use gpu_core::VulkanContext;
use server::GpuSimulation;
use shared::{GRID_SIZE, RENDER_HEIGHT, RENDER_WIDTH};

use crate::scene::create_island_particles;
use crate::Args;

/// Run the simulation without a window, producing a PNG screenshot.
pub(crate) fn run_headless(args: &Args) -> Result<()> {
    tracing::info!("Headless mode: {} frames, {} substeps/frame", args.frames, args.substeps);
    let ctx = VulkanContext::new()?;
    let mut sim = GpuSimulation::new(&ctx)?;
    let particles = create_island_particles();
    sim.init_particles(&ctx, &particles)?;

    for i in 0..args.frames {
        ctx.execute_one_shot(|cmd| {
            for _ in 0..args.substeps { sim.step(cmd); }
        })?;
        if i % 10 == 0 { tracing::info!("Frame {}/{}", i, args.frames); }
    }

    let gs = GRID_SIZE as f32;
    let eye = [gs * 0.75, gs * 0.4, gs * 0.75];
    let target = [gs * 0.5, gs * 0.3, gs * 0.5];
    ctx.execute_one_shot(|cmd| {
        sim.render(cmd, RENDER_WIDTH, RENDER_HEIGHT, eye, target);
        sim.finalize_render(cmd);
    })?;

    let pixel_count = (RENDER_WIDTH as usize) * (RENDER_HEIGHT as usize) * 4;
    let pixels: Vec<u8> = gpu_core::buffer::readback::<u8>(&ctx, sim.render_output_gpu_buffer(), pixel_count)?;
    let mut rgba = pixels;
    for pixel in rgba.chunks_exact_mut(4) { pixel.swap(0, 2); }

    let output_path = args.output.as_deref().unwrap_or("screenshot.png");
    let img = image::RgbaImage::from_raw(RENDER_WIDTH, RENDER_HEIGHT, rgba)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from pixel data"))?;
    img.save(output_path)?;
    tracing::info!("Screenshot saved to {}", output_path);
    sim.destroy(&ctx);
    Ok(())
}
