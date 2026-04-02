//! # sim-cpu
//!
//! CPU reference implementation of MPM simulation.
//! Used for correctness testing against GPU compute shaders.
//! Implements: Grid operations, P2G, grid update, G2P, simulation orchestrator.

pub mod grid;
pub mod world;
