//! # protocol
//!
//! Server↔Client communication contract.
//! Defines `WorldSnapshot` (server→client) and `PlayerInput` (client→server).
//! Uses `bitcode` for fast binary serialization.
//! Designed so channel can be swapped for `quinn` QUIC later.
