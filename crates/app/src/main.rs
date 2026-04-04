//! # app
//!
//! Binary entry point for the VOX voxel engine.
//!
//! Features:
//! - Island scene with mountain, lake, and lava flow
//! - FPS camera with walk/fly mode toggle (F key)
//! - Material toolbar with number key selection (1-3)
//! - Left click spawns selected material, right click removes

mod app;
mod headless;
mod scene;

use anyhow::Result;
use winit::event_loop::EventLoop;

/// Default number of physics substeps per frame.
const DEFAULT_SUBSTEPS: u32 = 1;

struct Args {
    headless: bool,
    frames: u32,
    output: Option<String>,
    substeps: u32,
    /// Path to a RON scene file. If `None`, the default island scene is used.
    scene: Option<String>,
    /// If true, use the mountain range scene instead of the island.
    big: bool,
    /// Path to a `.vox` (MagicaVoxel) model to load into the scene.
    model: Option<String>,
    /// World-space offset (x,y,z) at which to place the loaded model.
    model_pos: Option<(f32, f32, f32)>,
    /// If true, use WorldManager for chunk-streamed procedural terrain.
    world: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let headless = args.contains(&"--headless".to_string());
    let frames = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1).cloned());
    let substeps = args
        .iter()
        .position(|a| a == "--substeps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SUBSTEPS);
    let scene = args
        .iter()
        .position(|a| a == "--scene")
        .and_then(|i| args.get(i + 1).cloned());
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());
    let big = args.contains(&"--big".to_string());
    let world = args.contains(&"--world".to_string());
    let model_pos = args
        .iter()
        .position(|a| a == "--model-pos")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 3 {
                Some((
                    parts[0].trim().parse::<f32>().ok()?,
                    parts[1].trim().parse::<f32>().ok()?,
                    parts[2].trim().parse::<f32>().ok()?,
                ))
            } else {
                None
            }
        });
    Args { headless, frames, output, substeps, scene, big, model, model_pos, world }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    tracing::info!("VOX Engine starting");
    let args = parse_args();
    if args.headless { return headless::run_headless(&args); }

    let event_loop = EventLoop::new()?;
    let mut application = app::App::new(args.substeps, args.scene.as_deref(), args.big, args.model.as_deref(), args.model_pos, args.world);
    event_loop.run_app(&mut application)?;
    tracing::info!("VOX Engine shut down");
    Ok(())
}
