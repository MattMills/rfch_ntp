use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

use crate::core::{Error, Result, FrequencyComponent, NodeId, Peer, Precision, Tier};
use super::message::{Message, TierInfo, FrequencyBand, TierChangeReason};

/// Represents the current state of a node in the network
#[derive(Debug)]
pub enum NodeState {
    /// Initial state when node starts up
    Initial,
    
    /// Connecting to network, sent Hello message
    Connecting {
        /// Time Hello message was sent
        hello_sent: SystemTime,
        /// Number of connection attempts
        attempts: u32,
    },

    /// Connected to network in specific tier
    Connected {
        /// Current tier
        tier: Tier,
        /// Known peers
        peers: HashMap<NodeId, Peer>,
        /// Current frequency components
        components: Vec<FrequencyComponent>,
        /// Current precision level
        precision: Precision,
        /// Last heartbeat sent
        last_heartbeat: SystemTime,
    },

    /// Synchronizing frequency components
    Synchronizing {
        /// Target frequency components
        target_components: Vec<FrequencyComponent>,
        /// Sync start time
        start_time: SystemTime,
        /// Collected measurements
        measurements: Vec<FrequencyComponent>,
    },

    /// Transitioning between tiers
    TierTransition {
        /// Current tier
        current_tier: Tier,
        /// Target tier
        target_tier: Tier,
        /// Transition start time
        start_time: SystemTime,
    },
}

/// Protocol state machine for managing node behavior
pub struct ProtocolState {
    /// Node's unique identifier
    node_id: NodeId,
    /// Current state
    state: NodeState,
    /// Channel for sending messages
    message_tx: mpsc::Sender<Message>,
    /// Configuration
    config: ProtocolConfig,
}

/// Protocol configuration
#[derive(Debug, Clone)]
pub struct ProtocolConfig {
    /// Maximum connection attempts
    pub max_connection_attempts: u32,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Sync timeout
    pub sync_timeout: Duration,
    /// Minimum measurements for promotion
    pub min_promotion_measurements: u32,
    /// Required precision for promotion
    pub promotion_precision: f64,
    /// Timeout for considering a peer stale
    pub peer_timeout: Duration,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        ProtocolConfig {
            max_connection_attempts: 3,
            connection_timeout: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(1),
            sync_timeout: Duration::from_secs(30),
            min_promotion_measurements: 10,
            promotion_precision: 0.99,
            peer_timeout: Duration::from_secs(10),
        }
    }
}

/// Information about the current protocol state
#[derive(Debug, Clone)]
pub struct StateInfo {
    /// Current state type
    pub state_type: &'static str,
    /// Current tier (if connected)
    pub tier: Option<Tier>,
    /// Number of known peers
    pub peer_count: usize,
    /// Current precision
    pub precision: Option<Precision>,
    /// Time in current state
    pub time_in_state: Duration,
}

impl ProtocolState {
    /// Creates a new protocol state machine
    pub fn new(
        node_id: NodeId,
        message_tx: mpsc::Sender<Message>,
        config: ProtocolConfig,
    ) -> Self {
        ProtocolState {
            node_id,
            state: NodeState::Initial,
            message_tx,
            config,
        }
    }

    /// Gets information about the current state
    pub fn get_state_info(&self) -> StateInfo {
        let now = SystemTime::now();
        match &self.state {
            NodeState::Initial => StateInfo {
                state_type: "Initial",
                tier: None,
                peer_count: 0,
                precision: None,
                time_in_state: Duration::from_secs(0),
            },
            NodeState::Connecting { hello_sent, .. } => StateInfo {
                state_type: "Connecting",
                tier: None,
                peer_count: 0,
                precision: None,
                time_in_state: now.duration_since(*hello_sent)
                    .unwrap_or(Duration::from_secs(0)),
            },
            NodeState::Connected { tier, peers, precision, last_heartbeat, .. } => StateInfo {
                state_type: "Connected",
                tier: Some(*tier),
                peer_count: peers.len(),
                precision: Some(*precision),
                time_in_state: now.duration_since(*last_heartbeat)
                    .unwrap_or(Duration::from_secs(0)),
            },
            NodeState::Synchronizing { start_time, measurements, .. } => StateInfo {
                state_type: "Synchronizing",
                tier: None,
                peer_count: 0,
                precision: None,
                time_in_state: now.duration_since(*start_time)
                    .unwrap_or(Duration::from_secs(0)),
            },
            NodeState::TierTransition { current_tier, target_tier, start_time } => StateInfo {
                state_type: "TierTransition",
                tier: Some(*current_tier),
                peer_count: 0,
                precision: None,
                time_in_state: now.duration_since(*start_time)
                    .unwrap_or(Duration::from_secs(0)),
            },
        }
    }

    /// Handles an incoming message
    pub async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::Welcome { network_tiers, .. } => {
                if !matches!(self.state, NodeState::Connecting { .. }) {
                    return Err(Error::protocol("Received Welcome message in invalid state"));
                }
                self.handle_welcome(network_tiers).await
            }

            Message::Heartbeat { node_id, components, .. } => {
                if !matches!(self.state, NodeState::Connected { .. }) {
                    return Err(Error::protocol("Received Heartbeat message in invalid state"));
                }
                self.handle_heartbeat(node_id, components).await
            }

            Message::SyncResponse { components, .. } => {
                if !matches!(self.state, NodeState::Synchronizing { .. }) {
                    return Err(Error::protocol("Received SyncResponse in invalid state"));
                }
                self.handle_sync_response(components).await
            }

            Message::TierUpdate { new_tier, reason, .. } => {
                if !matches!(self.state, NodeState::Connected { .. }) {
                    return Err(Error::protocol("Received TierUpdate in invalid state"));
                }
                self.handle_tier_update(new_tier, reason).await
            }

            _ => Err(Error::protocol("Unexpected message type for current state")),
        }
    }

    /// Initiates the connection process
    pub async fn connect(&mut self) -> Result<()> {
        if !matches!(self.state, NodeState::Initial) {
            return Err(Error::protocol("Can only connect from Initial state"));
        }

        self.send_hello().await
    }

    /// Retries connection if previous attempt failed
    pub async fn retry_connect(&mut self) -> Result<()> {
        match &self.state {
            NodeState::Connecting { attempts, .. } => {
                if *attempts >= self.config.max_connection_attempts {
                    return Err(Error::protocol("Maximum connection attempts exceeded"));
                }

                self.send_hello().await
            }
            _ => Err(Error::protocol("Can only retry from Connecting state")),
        }
    }

    /// Sends Hello message and updates state
    async fn send_hello(&mut self) -> Result<()> {
        let hello = Message::Hello {
            node_id: self.node_id.clone(),
            tier: Tier::base(),
            version: crate::core::PROTOCOL_VERSION,
        };

        self.message_tx.send(hello).await
            .map_err(|e| Error::protocol(format!("Failed to send Hello message: {}", e)))?;

        let attempts = match &self.state {
            NodeState::Connecting { attempts, .. } => attempts + 1,
            _ => 1,
        };

        self.state = NodeState::Connecting {
            hello_sent: SystemTime::now(),
            attempts,
        };

        Ok(())
    }

    /// Handles Welcome message response
    async fn handle_welcome(&mut self, network_tiers: Vec<TierInfo>) -> Result<()> {
        if let NodeState::Connecting { attempts, .. } = &self.state {
            // Start in base tier
            self.state = NodeState::Connected {
                tier: Tier::base(),
                peers: HashMap::new(),
                components: Vec::new(),
                precision: Precision::min(),
                last_heartbeat: SystemTime::now(),
            };
            Ok(())
        } else {
            Err(Error::protocol("Received Welcome message in invalid state"))
        }
    }

    /// Handles incoming heartbeat messages
    async fn handle_heartbeat(&mut self, peer_id: NodeId, components: Vec<FrequencyComponent>) -> Result<()> {
        if let NodeState::Connected { peers, .. } = &mut self.state {
            // Update peer's components
            if let Some(peer) = peers.get_mut(&peer_id) {
                peer.components = components;
                peer.last_seen = SystemTime::now();
            }
            Ok(())
        } else {
            Err(Error::protocol("Received Heartbeat message in invalid state"))
        }
    }

    /// Sends heartbeat and cleans up stale peers
    pub async fn send_heartbeat(&mut self) -> Result<()> {
        if let NodeState::Connected { tier, peers, components, last_heartbeat, .. } = &mut self.state {
            // Check if it's time for a new heartbeat
            let now = SystemTime::now();
            let elapsed = now.duration_since(*last_heartbeat)
                .map_err(|e| Error::timing(format!("Time went backwards: {}", e)))?;

            if elapsed >= self.config.heartbeat_interval {
                // Remove stale peers
                peers.retain(|_, peer| {
                    if let Ok(elapsed) = now.duration_since(peer.last_seen) {
                        elapsed < self.config.peer_timeout
                    } else {
                        false
                    }
                });

                // Send heartbeat
                let heartbeat = Message::Heartbeat {
                    node_id: self.node_id.clone(),
                    tier: *tier,
                    timestamp: now,
                    components: components.clone(),
                };

                self.message_tx.send(heartbeat).await
                    .map_err(|e| Error::protocol(format!("Failed to send heartbeat: {}", e)))?;

                *last_heartbeat = now;
            }
        }
        Ok(())
    }

    /// Handles sync response messages
    async fn handle_sync_response(&mut self, components: Vec<FrequencyComponent>) -> Result<()> {
        if let NodeState::Synchronizing { measurements, .. } = &mut self.state {
            measurements.extend(components);
            Ok(())
        } else {
            Err(Error::protocol("Received SyncResponse in invalid state"))
        }
    }

    /// Handles tier update messages
    async fn handle_tier_update(&mut self, new_tier: Tier, reason: TierChangeReason) -> Result<()> {
        match &self.state {
            NodeState::Connected { tier, peers, components, precision, .. } => {
                // Validate tier change
                match reason {
                    TierChangeReason::Promotion { precision: req_precision, .. } => {
                        if precision.0 as f64 / 1000.0 < req_precision {
                            return Err(Error::protocol("Insufficient precision for promotion"));
                        }
                    }
                    TierChangeReason::Demotion { error_margin, .. } => {
                        if error_margin < self.config.promotion_precision {
                            return Err(Error::protocol("Invalid demotion with low error margin"));
                        }
                    }
                    TierChangeReason::Rebalancing { .. } => {
                        // Always allow rebalancing
                    }
                }

                // Start transition
                self.state = NodeState::TierTransition {
                    current_tier: *tier,
                    target_tier: new_tier,
                    start_time: SystemTime::now(),
                };

                // Notify peers of transition
                let update = Message::TierUpdate {
                    node_id: self.node_id.clone(),
                    new_tier,
                    reason: reason.clone(),
                };

                self.message_tx.send(update).await
                    .map_err(|e| Error::protocol(format!("Failed to send tier update: {}", e)))?;

                Ok(())
            }
            _ => Err(Error::protocol("Received TierUpdate in invalid state")),
        }
    }

    /// Checks if ready to complete tier transition
    pub async fn check_tier_transition(&mut self) -> Result<()> {
        let (should_transition, target_tier) = match &self.state {
            NodeState::TierTransition { start_time, target_tier, .. } => {
                let elapsed = SystemTime::now()
                    .duration_since(start_time.clone())
                    .map_err(|e| Error::timing(format!("Time went backwards: {}", e)))?;
                
                if elapsed >= Duration::from_secs(5) {
                    (true, target_tier.clone())
                } else {
                    (false, Tier::base()) // Dummy value, won't be used
                }
            }
            _ => (false, Tier::base()), // Dummy value, won't be used
        };

        if should_transition {
            // Request sync with peers in new tier
            let sync_req = Message::SyncRequest {
                node_id: self.node_id.clone(),
                requested_components: vec![], // Request all components
            };

            self.message_tx.send(sync_req).await
                .map_err(|e| Error::protocol(format!("Failed to send sync request: {}", e)))?;

            // Complete transition
            self.state = NodeState::Connected {
                tier: target_tier,
                peers: HashMap::new(),
                components: Vec::new(),
                precision: Precision::min(),
                last_heartbeat: SystemTime::now(),
            };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    use tokio::time::sleep;
    use num_complex::Complex64;

    #[tokio::test]
    async fn test_connection_flow() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Start connection
        state.connect().await.unwrap();
        assert!(matches!(state.state, NodeState::Connecting { .. }));

        // Receive Hello message
        if let Some(Message::Hello { node_id: id, .. }) = rx.recv().await {
            assert_eq!(id, node_id);
        } else {
            panic!("Expected Hello message");
        }

        // Handle Welcome message
        let welcome = Message::Welcome {
            node_id: NodeId::random(),
            reference_time: SystemTime::now(),
            network_tiers: vec![],
        };
        state.handle_message(welcome).await.unwrap();
        assert!(matches!(state.state, NodeState::Connected { .. }));
    }

    #[tokio::test]
    async fn test_tier_transition() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Start in connected state
        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision(1000), // High enough for promotion
            last_heartbeat: SystemTime::now(),
        };

        // Handle tier update
        let update = Message::TierUpdate {
            node_id: NodeId::random(),
            new_tier: Tier::new(1),
            reason: TierChangeReason::Promotion {
                precision: 0.95,
                measurement_count: 10,
            },
        };

        state.handle_message(update).await.unwrap();
        assert!(matches!(state.state, NodeState::TierTransition { .. }));

        // Verify tier update message was sent
        match rx.recv().await {
            Some(Message::TierUpdate { node_id: id, new_tier, .. }) => {
                assert_eq!(id, node_id);
                assert_eq!(new_tier, Tier::new(1));
            }
            _ => panic!("Expected TierUpdate message"),
        }

        // Wait for transition
        sleep(Duration::from_secs(6)).await;
        state.check_tier_transition().await.unwrap();

        // Should have sent sync request
        match rx.recv().await {
            Some(Message::SyncRequest { node_id: id, .. }) => {
                assert_eq!(id, node_id);
            }
            _ => panic!("Expected SyncRequest message"),
        }

        // Should be in new tier
        if let NodeState::Connected { tier, .. } = state.state {
            assert_eq!(tier, Tier::new(1));
        } else {
            panic!("Expected Connected state");
        }
    }

    #[tokio::test]
    async fn test_sync_flow() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Start in synchronizing state
        let target = vec![FrequencyComponent {
            amplitude: Complex64::new(1.0, 0.0),
            frequency: 1.0,
            phase: 0.0,
            timestamp: SystemTime::now(),
        }];

        state.state = NodeState::Synchronizing {
            target_components: target.clone(),
            start_time: SystemTime::now(),
            measurements: vec![],
        };

        // Handle sync response
        let response = Message::SyncResponse {
            node_id: NodeId::random(),
            timestamp: SystemTime::now(),
            components: target,
        };

        state.handle_message(response).await.unwrap();

        // Verify measurements were collected
        if let NodeState::Synchronizing { measurements, .. } = &state.state {
            assert!(!measurements.is_empty());
        } else {
            panic!("Expected Synchronizing state");
        }
    }

    #[tokio::test]
    async fn test_invalid_transitions() {
        let (tx, _rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Can't connect when not in Initial state
        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now(),
        };
        assert!(state.connect().await.is_err());

        // Can't handle Welcome when not connecting
        let welcome = Message::Welcome {
            node_id: NodeId::random(),
            reference_time: SystemTime::now(),
            network_tiers: vec![],
        };
        assert!(state.handle_message(welcome).await.is_err());
    }

    #[tokio::test]
    async fn test_connection_retry() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut config = ProtocolConfig::default();
        config.max_connection_attempts = 2;
        let mut state = ProtocolState::new(node_id.clone(), tx, config);

        // Initial connection
        state.connect().await.unwrap();
        if let NodeState::Connecting { attempts, .. } = &state.state {
            assert_eq!(*attempts, 1);
        } else {
            panic!("Expected Connecting state");
        }

        // Verify Hello message
        match rx.recv().await {
            Some(Message::Hello { node_id: id, .. }) => {
                assert_eq!(id, node_id);
            }
            _ => panic!("Expected Hello message"),
        }

        // First retry
        state.retry_connect().await.unwrap();
        if let NodeState::Connecting { attempts, .. } = &state.state {
            assert_eq!(*attempts, 2);
        } else {
            panic!("Expected Connecting state");
        }

        // Verify second Hello message
        match rx.recv().await {
            Some(Message::Hello { node_id: id, .. }) => {
                assert_eq!(id, node_id);
            }
            _ => panic!("Expected Hello message"),
        }

        // Second retry should fail (exceeds max attempts)
        assert!(state.retry_connect().await.is_err());

        // Can't retry from Connected state
        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now(),
        };
        assert!(state.retry_connect().await.is_err());
    }

    #[tokio::test]
    async fn test_heartbeat_and_cleanup() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut config = ProtocolConfig::default();
        config.heartbeat_interval = Duration::from_millis(100);
        config.peer_timeout = Duration::from_millis(200);

        let mut state = ProtocolState::new(node_id.clone(), tx, config);

        // Start in connected state with a peer
        let peer_id = NodeId::random();
        let mut peers = HashMap::new();
        peers.insert(peer_id.clone(), Peer {
            id: peer_id.clone(),
            addr: "127.0.0.1:4444".parse().unwrap(),
            tier: Tier::base(),
            last_seen: SystemTime::now(),
            rtt: Duration::from_millis(50),
            components: vec![],
        });

        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers,
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now() - Duration::from_millis(200), // Due for heartbeat
        };

        // Should send heartbeat immediately
        state.send_heartbeat().await.unwrap();
        if let Some(Message::Heartbeat { node_id: id, .. }) = rx.recv().await {
            assert_eq!(id, node_id);
        } else {
            panic!("Expected Heartbeat message");
        }

        // Wait for peer to become stale
        sleep(Duration::from_millis(250)).await;
        state.send_heartbeat().await.unwrap();

        // Peer should be removed
        if let NodeState::Connected { peers, .. } = &state.state {
            assert!(peers.is_empty(), "Stale peer should have been removed");
        } else {
            panic!("Expected Connected state");
        }
    }

    #[tokio::test]
    async fn test_tier_demotion() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Start in connected state in tier 1
        state.state = NodeState::Connected {
            tier: Tier::new(1),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision(1000),
            last_heartbeat: SystemTime::now(),
        };

        // Handle demotion
        let update = Message::TierUpdate {
            node_id: NodeId::random(),
            new_tier: Tier::base(),
            reason: TierChangeReason::Demotion {
                precision: 0.5,
                error_margin: 1.0, // High error margin
            },
        };

        state.handle_message(update).await.unwrap();
        assert!(matches!(state.state, NodeState::TierTransition { .. }));

        // Verify tier update message was sent
        match rx.recv().await {
            Some(Message::TierUpdate { node_id: id, new_tier, .. }) => {
                assert_eq!(id, node_id);
                assert_eq!(new_tier, Tier::base());
            }
            _ => panic!("Expected TierUpdate message"),
        }

        // Wait for transition
        sleep(Duration::from_secs(6)).await;
        state.check_tier_transition().await.unwrap();

        // Should have sent sync request
        match rx.recv().await {
            Some(Message::SyncRequest { node_id: id, .. }) => {
                assert_eq!(id, node_id);
            }
            _ => panic!("Expected SyncRequest message"),
        }

        // Should be in base tier
        if let NodeState::Connected { tier, .. } = state.state {
            assert_eq!(tier, Tier::base());
        } else {
            panic!("Expected Connected state");
        }
    }

    #[tokio::test]
    async fn test_failed_tier_transition() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Start in connected state with insufficient precision
        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision(500), // Too low for promotion
            last_heartbeat: SystemTime::now(),
        };

        // Attempt tier update
        let update = Message::TierUpdate {
            node_id: NodeId::random(),
            new_tier: Tier::new(1),
            reason: TierChangeReason::Promotion {
                precision: 0.95,
                measurement_count: 10,
            },
        };

        // Should fail due to insufficient precision
        assert!(state.handle_message(update).await.is_err());
        assert!(matches!(state.state, NodeState::Connected { .. }));
    }

    #[tokio::test]
    async fn test_invalid_message_handling() {
        let (tx, _rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Test heartbeat in Initial state
        let heartbeat = Message::Heartbeat {
            node_id: NodeId::random(),
            tier: Tier::base(),
            timestamp: SystemTime::now(),
            components: vec![],
        };
        assert!(state.handle_message(heartbeat).await.is_err());

        // Test sync response in Connected state
        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers: HashMap::new(),
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now(),
        };

        let sync_resp = Message::SyncResponse {
            node_id: NodeId::random(),
            timestamp: SystemTime::now(),
            components: vec![],
        };
        assert!(state.handle_message(sync_resp).await.is_err());
    }

    #[tokio::test]
    async fn test_peer_management() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut config = ProtocolConfig::default();
        config.heartbeat_interval = Duration::from_millis(50);
        config.peer_timeout = Duration::from_millis(100);
        let mut state = ProtocolState::new(node_id.clone(), tx, config);

        // Add multiple peers
        let mut peers = HashMap::new();
        for i in 0..3 {
            let peer_id = NodeId::random();
            peers.insert(peer_id.clone(), Peer {
                id: peer_id.clone(),
                addr: format!("127.0.0.1:{}", 4444 + i).parse().unwrap(),
                tier: Tier::base(),
                last_seen: SystemTime::now(),
                rtt: Duration::from_millis(50),
                components: vec![],
            });
        }

        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers,
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now() - Duration::from_millis(100),
        };

        // First heartbeat should keep all peers
        state.send_heartbeat().await.unwrap();
        if let NodeState::Connected { peers, .. } = &state.state {
            assert_eq!(peers.len(), 3);
        }

        // Wait for peers to become stale
        sleep(Duration::from_millis(150)).await;
        state.send_heartbeat().await.unwrap();

        // All peers should be removed
        if let NodeState::Connected { peers, .. } = &state.state {
            assert!(peers.is_empty());
        }
    }

    #[test]
    fn test_state_info() {
        let (tx, _rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let mut state = ProtocolState::new(node_id.clone(), tx, ProtocolConfig::default());

        // Initial state
        let info = state.get_state_info();
        assert_eq!(info.state_type, "Initial");
        assert!(info.tier.is_none());
        assert_eq!(info.peer_count, 0);
        assert!(info.precision.is_none());

        // Connected state
        let mut peers = HashMap::new();
        peers.insert(NodeId::random(), Peer {
            id: NodeId::random(),
            addr: "127.0.0.1:4444".parse().unwrap(),
            tier: Tier::base(),
            last_seen: SystemTime::now(),
            rtt: Duration::from_millis(50),
            components: vec![],
        });

        state.state = NodeState::Connected {
            tier: Tier::base(),
            peers,
            components: vec![],
            precision: Precision::min(),
            last_heartbeat: SystemTime::now(),
        };

        let info = state.get_state_info();
        assert_eq!(info.state_type, "Connected");
        assert!(info.tier.is_some());
        assert_eq!(info.peer_count, 1);
        assert!(info.precision.is_some());

        // Synchronizing state
        state.state = NodeState::Synchronizing {
            target_components: vec![],
            start_time: SystemTime::now(),
            measurements: vec![],
        };

        let info = state.get_state_info();
        assert_eq!(info.state_type, "Synchronizing");
        assert!(info.tier.is_none());
        assert_eq!(info.peer_count, 0);
        assert!(info.precision.is_none());

        // Tier transition state
        state.state = NodeState::TierTransition {
            current_tier: Tier::base(),
            target_tier: Tier::new(1),
            start_time: SystemTime::now(),
        };

        let info = state.get_state_info();
        assert_eq!(info.state_type, "TierTransition");
        assert_eq!(info.tier, Some(Tier::base()));
        assert_eq!(info.peer_count, 0);
        assert!(info.precision.is_none());
    }
}
