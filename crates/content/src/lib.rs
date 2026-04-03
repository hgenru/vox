//! # content
//!
//! Data-driven material, phase transition, and reaction loading from RON files.
//!
//! This crate bridges the gap between human-readable RON definition files in
//! `assets/` and the GPU-friendly `#[repr(C)]` structs defined in `shared`.
//! It is CPU-only (not `no_std`) and depends on `serde` + `ron`.

mod convert;
pub mod scene;
mod types;
pub mod vox_loader;

pub use convert::MaterialDatabase;
pub use scene::{CameraDef, SceneDef, SceneObject};
pub use types::*;
pub use vox_loader::load_vox_model;

/// Errors that can occur when loading content files.
#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    /// Failed to read the file from disk.
    #[error("failed to read content file: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the RON content.
    #[error("failed to parse RON: {0}")]
    Ron(#[from] ron::error::SpannedError),

    /// A material definition has a duplicate id.
    #[error("duplicate material id {0}")]
    DuplicateMaterialId(u32),

    /// A phase transition references an unknown material id.
    #[error("phase transition references unknown material id {0}")]
    UnknownTransitionMaterial(u32),

    /// A reaction references an unknown material id.
    #[error("reaction references unknown material id {0}")]
    UnknownReactionMaterial(u32),

    /// Too many phase transition rules for the GPU buffer.
    #[error("too many phase transition rules: {count} exceeds maximum {max}")]
    TooManyPhaseRules { count: usize, max: usize },

    /// Failed to load a `.vox` (MagicaVoxel) file.
    #[error("failed to load .vox file: {0}")]
    VoxLoad(String),
}

/// Load a [`SceneDef`] from a RON file at the given path.
///
/// Parses the file into a scene definition whose [`SceneDef::spawn_particles`]
/// method can then generate the initial particle list.
///
/// # Errors
///
/// Returns [`ContentError`] if the file cannot be read or parsed.
pub fn load_scene(path: &str) -> std::result::Result<SceneDef, ContentError> {
    let content = std::fs::read_to_string(path)?;
    parse_scene(&content)
}

/// Parse a [`SceneDef`] from a RON string.
///
/// Same as [`load_scene`] but operates on an in-memory string.
/// Useful for testing without filesystem access.
///
/// # Errors
///
/// Returns [`ContentError`] if parsing fails.
pub fn parse_scene(ron_str: &str) -> std::result::Result<SceneDef, ContentError> {
    Ok(ron::from_str(ron_str)?)
}

/// Load a [`MaterialDatabase`] from a RON file at the given path.
///
/// Parses the file, validates all cross-references, and returns a database
/// ready to be converted into GPU-friendly structs via its accessor methods.
///
/// # Errors
///
/// Returns [`ContentError`] if the file cannot be read, parsed, or if
/// validation fails (duplicate ids, unknown material references, etc.).
pub fn load_material_database(path: &str) -> std::result::Result<MaterialDatabase, ContentError> {
    let content = std::fs::read_to_string(path)?;
    parse_material_database(&content)
}

/// Parse a [`MaterialDatabase`] from a RON string.
///
/// Same as [`load_material_database`] but operates on an in-memory string.
/// Useful for testing without filesystem access.
///
/// # Errors
///
/// Returns [`ContentError`] if parsing or validation fails.
pub fn parse_material_database(
    ron_str: &str,
) -> std::result::Result<MaterialDatabase, ContentError> {
    let raw: types::MaterialDatabaseDef = ron::from_str(ron_str)?;
    MaterialDatabase::from_def(raw)
}
