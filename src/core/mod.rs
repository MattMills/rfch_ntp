//! Core types and traits for the RFCH NTP protocol
//! 
//! This module contains the fundamental building blocks used throughout the library.

pub mod error;
pub mod types;
pub mod serde;

pub use self::error::{Error, Result};
pub use self::types::{
    Config,
    FrequencyComponent,
    NodeId,
    Peer,
    Precision,
    Tier,
};

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Default port for RFCH NTP protocol
pub const DEFAULT_PORT: u16 = 4444;

/// Maximum packet size in bytes
pub const MAX_PACKET_SIZE: usize = 1024;

/// Minimum peers required for tier promotion
pub const MIN_PEERS_FOR_PROMOTION: usize = 3;

/// Maximum number of tiers supported
pub const MAX_TIERS: u8 = 8;
