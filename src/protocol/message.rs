use serde::{Serialize, Deserialize};
use crate::core::{FrequencyComponent, NodeId, Tier};
use std::time::SystemTime;

/// Protocol message types for node communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Initial handshake when joining network
    Hello {
        /// Sender's node ID
        node_id: NodeId,
        /// Sender's current tier
        tier: Tier,
        /// Protocol version
        version: u8,
    },

    /// Response to Hello message
    Welcome {
        /// Responder's node ID
        node_id: NodeId,
        /// Network time reference
        #[serde(serialize_with = "crate::core::serde::serialize_time")]
        #[serde(deserialize_with = "crate::core::serde::deserialize_time")]
        reference_time: SystemTime,
        /// Current network tier structure
        network_tiers: Vec<TierInfo>,
    },

    /// Regular heartbeat message containing frequency components
    Heartbeat {
        /// Sender's node ID
        node_id: NodeId,
        /// Sender's current tier
        tier: Tier,
        /// Timestamp of message creation
        #[serde(serialize_with = "crate::core::serde::serialize_time")]
        #[serde(deserialize_with = "crate::core::serde::deserialize_time")]
        timestamp: SystemTime,
        /// Current frequency components
        components: Vec<FrequencyComponent>,
    },

    /// Request for synchronization with specific frequency components
    SyncRequest {
        /// Requester's node ID
        node_id: NodeId,
        /// Requested frequency components
        requested_components: Vec<f64>,
    },

    /// Response to sync request
    SyncResponse {
        /// Responder's node ID
        node_id: NodeId,
        /// Timestamp of response
        #[serde(serialize_with = "crate::core::serde::serialize_time")]
        #[serde(deserialize_with = "crate::core::serde::deserialize_time")]
        timestamp: SystemTime,
        /// Provided frequency components
        components: Vec<FrequencyComponent>,
    },

    /// Notification of tier change
    TierUpdate {
        /// Node ID
        node_id: NodeId,
        /// New tier
        new_tier: Tier,
        /// Reason for tier change
        reason: TierChangeReason,
    },
}

/// Information about a network tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierInfo {
    /// The tier level
    pub tier: Tier,
    /// Number of nodes in this tier
    pub node_count: u32,
    /// Average precision of nodes in this tier
    pub avg_precision: f64,
    /// Frequency bands handled by this tier
    pub frequency_bands: Vec<FrequencyBand>,
}

/// Represents a frequency band handled by a tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBand {
    /// Lower bound frequency in Hz
    pub low_freq: f64,
    /// Upper bound frequency in Hz
    pub high_freq: f64,
    /// Current number of nodes processing this band
    pub node_count: u32,
}

/// Reason for tier change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TierChangeReason {
    /// Promotion due to achieved precision
    Promotion {
        /// Achieved precision level
        precision: f64,
        /// Number of consistent measurements
        measurement_count: u32,
    },
    /// Demotion due to precision loss
    Demotion {
        /// Current precision level
        precision: f64,
        /// Error margin
        error_margin: f64,
    },
    /// Network rebalancing
    Rebalancing {
        /// Target node distribution
        target_distribution: Vec<u32>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use num_complex::Complex64;

    #[test]
    fn test_message_creation() {
        let node_id = NodeId::random();
        let tier = Tier::new(0);
        let now = SystemTime::now();

        let hello = Message::Hello {
            node_id: node_id.clone(),
            tier,
            version: crate::core::PROTOCOL_VERSION,
        };

        let heartbeat = Message::Heartbeat {
            node_id: node_id.clone(),
            tier,
            timestamp: now,
            components: vec![],
        };

        assert!(matches!(hello, Message::Hello { .. }));
        assert!(matches!(heartbeat, Message::Heartbeat { .. }));
    }

    #[test]
    fn test_message_serialization() {
        let node_id = NodeId::random();
        let tier = Tier::new(0);
        let now = SystemTime::now();
        let component = FrequencyComponent {
            amplitude: Complex64::new(1.0, 2.0),
            frequency: 10.0,
            phase: 0.5,
            timestamp: now,
        };

        let heartbeat = Message::Heartbeat {
            node_id: node_id.clone(),
            tier,
            timestamp: now,
            components: vec![component],
        };

        // Test bincode serialization
        let encoded = bincode::serialize(&heartbeat).unwrap();
        let decoded: Message = bincode::deserialize(&encoded).unwrap();

        match decoded {
            Message::Heartbeat { node_id: id, tier: t, timestamp, components } => {
                assert_eq!(id, node_id);
                assert_eq!(t, tier);
                assert_eq!(components.len(), 1);
                assert_eq!(components[0].frequency, 10.0);
                assert_eq!(components[0].phase, 0.5);
                assert_eq!(components[0].amplitude, Complex64::new(1.0, 2.0));

                // SystemTime comparison within reasonable bounds
                let diff = timestamp
                    .duration_since(now)
                    .unwrap_or_else(|e| e.duration());
                assert!(diff < Duration::from_millis(1));
            },
            _ => panic!("Decoded wrong message type"),
        }
    }

    #[test]
    fn test_tier_info() {
        let tier_info = TierInfo {
            tier: Tier::new(1),
            node_count: 5,
            avg_precision: 0.95,
            frequency_bands: vec![
                FrequencyBand {
                    low_freq: 0.0,
                    high_freq: 1.0,
                    node_count: 3,
                }
            ],
        };

        assert_eq!(tier_info.tier.0, 1);
        assert_eq!(tier_info.node_count, 5);
        assert!((tier_info.avg_precision - 0.95).abs() < f64::EPSILON);

        // Test serialization
        let encoded = bincode::serialize(&tier_info).unwrap();
        let decoded: TierInfo = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.tier, tier_info.tier);
        assert_eq!(decoded.node_count, tier_info.node_count);
        assert_eq!(decoded.avg_precision, tier_info.avg_precision);
        assert_eq!(decoded.frequency_bands.len(), tier_info.frequency_bands.len());
    }
}
