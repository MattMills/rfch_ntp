//! RFCH NTP: Recursive Frequency Composition Heartbeat Network Timing Protocol
//! 
//! This library implements a novel distributed timing protocol that establishes precise timing
//! through emergent behavior in a peer network rather than relying on absolute time sources.
#![ allow(warnings)]
pub mod core;

// These modules will be implemented incrementally
mod network;
mod protocol;
mod sync;
pub mod time;
mod util;

// Re-export commonly used items
pub use core::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
