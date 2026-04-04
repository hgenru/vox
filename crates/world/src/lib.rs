//! # world
//!
//! Chunk-based infinite world management for the VOX engine.
//!
//! Provides procedural terrain generation, chunk streaming (load/unload
//! around the camera), and on-disk persistence. The simulation grid is
//! always a fixed-size window into the infinite world; this crate decides
//! which chunks are active and translates particles between local chunk
//! coordinates and the global GPU grid.

pub mod chunk;
pub mod manager;
pub mod storage;
pub mod terrain;

pub use chunk::{ChunkCoord, ChunkData};
pub use manager::WorldManager;
