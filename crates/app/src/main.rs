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
const DEFAULT_SUBSTEPS: u32 = 2;

struct Args {
    headless: bool,
    frames: u32,
    output: Option<String>,
    substeps: u32,
    /// Path to a RON scene file. If `None`, the default island scene is used.
    scene: Option<String>,
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
    Args { headless, frames, output, substeps, scene }
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
    let mut application = app::App::new(args.substeps, args.scene.as_deref());
    event_loop.run_app(&mut application)?;
    tracing::info!("VOX Engine shut down");
    Ok(())
}
