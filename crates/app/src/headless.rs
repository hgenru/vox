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
    let mut sim = match content::load_material_database("assets/materials.ron") {
        Ok(db) => {
            tracing::info!("Loaded material database from assets/materials.ron");
            let materials = db.material_params();
            let (transitions, count) = db.phase_transition_rules();
            GpuSimulation::new_with_materials(&ctx, &materials, &transitions, count)?
        }
        Err(e) => {
            tracing::warn!(
                "Failed to load assets/materials.ron, using hardcoded defaults: {}",
                e
            );
            GpuSimulation::new(&ctx)?
        }
    };
    let (mut particles, scene_camera) = if let Some(scene_path) = &args.scene {
        let scene = content::load_scene(scene_path)?;
        tracing::info!("Loaded scene '{}' from {}", scene.name, scene_path);
        let cam = scene.camera.clone();
        let p = scene.spawn_particles();
        (p, cam)
    } else if args.big {
        tracing::info!("Using mountain range scene (--big)");
        (crate::scene::create_mountain_particles(), None)
    } else {
        (create_island_particles(), None)
    };
    if let Some(path) = &args.model {
        let pos = args.model_pos.unwrap_or((32.0, 20.0, 32.0));
        let offset = glam::Vec3::new(pos.0, pos.1, pos.2);
        let palette = content::vox_loader::default_palette_mapping();
        match content::load_vox_model(path, offset, &palette) {
            Ok(model_particles) => {
                tracing::info!("Loaded {} particles from model '{}'", model_particles.len(), path);
                particles.extend(model_particles);
            }
            Err(e) => {
                tracing::warn!("Failed to load .vox model '{}': {}", path, e);
            }
        }
    }
    sim.init_particles(&ctx, &particles)?;

    for i in 0..args.frames {
        ctx.execute_one_shot(|cmd| {
            for _ in 0..args.substeps { sim.step(cmd); }
        })?;
        if i % 10 == 0 { tracing::info!("Frame {}/{}", i, args.frames); }
    }

    let (eye, target) = if let Some(cam) = &scene_camera {
        ([cam.eye.0, cam.eye.1, cam.eye.2], [cam.target.0, cam.target.1, cam.target.2])
    } else {
        let gs = GRID_SIZE as f32;
        ([gs * 0.75, gs * 0.4, gs * 0.75], [gs * 0.5, gs * 0.3, gs * 0.5])
    };
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
