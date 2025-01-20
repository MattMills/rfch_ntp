use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use num_complex::Complex64;

use serde::{Serialize, Deserialize};

/// Represents a tier in the hierarchical network structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tier(pub u8);

impl Tier {
    /// Creates a new tier
    pub fn new(level: u8) -> Self {
        Tier(level.min(super::MAX_TIERS - 1))
    }

    /// Returns the base tier (level 0)
    pub fn base() -> Self {
        Tier(0)
    }

    /// Returns the next tier up
    pub fn next(&self) -> Option<Self> {
        if self.0 + 1 < super::MAX_TIERS {
            Some(Tier(self.0 + 1))
        } else {
            None
        }
    }

    /// Returns the tier level
    pub fn level(&self) -> u8 {
        self.0
    }

    /// Creates a tier from precision level
    pub fn from_precision(precision: &Precision) -> Self {
        Tier((precision.0 / 1000) as u8)
    }
}

/// Node identifier in the network
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 16]);

impl NodeId {
    /// Generates a new random node ID
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);
        NodeId(bytes)
    }
}

/// Represents a frequency component in the signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyComponent {
    /// The complex amplitude of this frequency component
    #[serde(serialize_with = "super::serde::serialize_complex")]
    #[serde(deserialize_with = "super::serde::deserialize_complex")]
    pub amplitude: Complex64,
    /// The frequency in Hz
    pub frequency: f64,
    /// The phase in radians
    pub phase: f64,
    /// The time this component was measured
    #[serde(serialize_with = "super::serde::serialize_time")]
    #[serde(deserialize_with = "super::serde::deserialize_time")]
    pub timestamp: SystemTime,
}

/// Represents a peer in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    /// The peer's node ID
    pub id: NodeId,
    /// The peer's network address
    pub addr: SocketAddr,
    /// The peer's current tier
    pub tier: Tier,
    /// Last seen timestamp
    #[serde(serialize_with = "super::serde::serialize_time")]
    #[serde(deserialize_with = "super::serde::deserialize_time")]
    pub last_seen: SystemTime,
    /// Round-trip time to this peer
    #[serde(serialize_with = "super::serde::serialize_duration")]
    #[serde(deserialize_with = "super::serde::deserialize_duration")]
    pub rtt: Duration,
    /// Current frequency components being broadcast by this peer
    pub components: Vec<FrequencyComponent>,
}

/// Represents the precision level of timing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Precision(pub u32);

impl Precision {
    /// Returns the minimum precision level
    pub fn min() -> Self {
        Precision(0)
    }

    /// Returns whether this precision level is sufficient for promotion
    pub fn is_promotion_worthy(&self) -> bool {
        self.0 >= 1000 // Example threshold
    }
}

/// Configuration for the RFCH NTP node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Local address to bind to
    pub bind_addr: SocketAddr,
    /// Initial peers to connect to
    pub initial_peers: Vec<SocketAddr>,
    /// Minimum number of peers required for operation
    pub min_peers: usize,
    /// Maximum number of peers to maintain
    pub max_peers: usize,
    /// Base heartbeat interval
    #[serde(serialize_with = "super::serde::serialize_duration")]
    #[serde(deserialize_with = "super::serde::deserialize_duration")]
    pub heartbeat_interval: Duration,
    /// Timeout for peer connections
    #[serde(serialize_with = "super::serde::serialize_duration")]
    #[serde(deserialize_with = "super::serde::deserialize_duration")]
    pub peer_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            bind_addr: format!("0.0.0.0:{}", super::DEFAULT_PORT).parse().unwrap(),
            initial_peers: Vec::new(),
            min_peers: super::MIN_PEERS_FOR_PROMOTION,
            max_peers: 10,
            heartbeat_interval: Duration::from_secs(1),
            peer_timeout: Duration::from_secs(5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_creation() {
        let tier = Tier::new(3);
        assert_eq!(tier.0, 3);
        
        // Test max tier limit
        let tier = Tier::new(255);
        assert_eq!(tier.0, super::super::MAX_TIERS - 1);
    }

    #[test]
    fn test_node_id_random() {
        let id1 = NodeId::random();
        let id2 = NodeId::random();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_precision_ordering() {
        let p1 = Precision(100);
        let p2 = Precision(200);
        assert!(p1 < p2);
    }
}
