//! Protocol implementation module
//! 
//! This module defines the RFCH NTP protocol messages, encoding/decoding,
//! and protocol state machines.

pub mod codec;
pub mod message;
pub mod state;

pub use self::codec::MessageCodec;
pub use self::message::{Message, TierInfo, FrequencyBand, TierChangeReason};
pub use self::state::{ProtocolState, ProtocolConfig, NodeState};

use crate::core::{Error, Result, FrequencyComponent, Tier};

// Constants
/// Maximum message size in bytes
pub const MAX_MESSAGE_SIZE: usize = 65507; // Maximum UDP payload size

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Default port for protocol communication
pub const DEFAULT_PORT: u16 = 4444;
