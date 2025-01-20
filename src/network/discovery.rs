use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::mpsc;
use tokio::time::interval;

use crate::core::{Error, Result, NodeId, Peer, Tier};
use crate::protocol::Message;

/// Configuration for peer discovery
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Initial peers to connect to
    pub initial_peers: Vec<SocketAddr>,
    /// Discovery interval
    pub discovery_interval: Duration,
    /// Peer timeout duration
    pub peer_timeout: Duration,
    /// Maximum number of peers
    pub max_peers: usize,
    /// Minimum number of peers
    pub min_peers: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        DiscoveryConfig {
            initial_peers: Vec::new(),
            discovery_interval: Duration::from_secs(60),
            peer_timeout: Duration::from_secs(180),
            max_peers: 10,
            min_peers: 3,
        }
    }
}

/// Shared state between discovery instances
#[derive(Clone)]
struct SharedState {
    /// Local node ID
    node_id: NodeId,
    /// Current tier
    tier: Tier,
    /// Configuration
    config: DiscoveryConfig,
    /// Channel for sending messages
    message_tx: mpsc::Sender<(Message, SocketAddr)>,
}

/// Inner mutable state
struct Inner {
    /// Known peers
    peers: HashMap<NodeId, Peer>,
}

/// Manages peer discovery and maintenance
pub struct PeerDiscovery {
    /// Shared state
    shared: Arc<SharedState>,
    /// Inner mutable state
    inner: Inner,
}

impl Clone for PeerDiscovery {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            inner: Inner {
                peers: HashMap::new(),
            },
        }
    }
}

impl PeerDiscovery {
    /// Creates a new peer discovery manager
    pub fn new(
        node_id: NodeId,
        tier: Tier,
        config: DiscoveryConfig,
        message_tx: mpsc::Sender<(Message, SocketAddr)>,
    ) -> Self {
        let shared = Arc::new(SharedState {
            node_id,
            tier,
            config,
            message_tx,
        });

        let inner = Inner {
            peers: HashMap::new(),
        };

        Self { shared, inner }
    }

    /// Starts the discovery process
    pub async fn run(&mut self) -> Result<()> {
        // Connect to initial peers
        let initial_peers = self.shared.config.initial_peers.clone();
        for &addr in &initial_peers {
            self.try_connect(addr).await?;
        }

        let mut discovery_interval = interval(self.shared.config.discovery_interval);
        let min_peers = self.shared.config.min_peers;

        loop {
            discovery_interval.tick().await;
            
            // Perform maintenance
            self.maintain_peers().await?;
            
            // Check if we need more peers
            let current_peers = self.inner.peers.len();
            if current_peers < min_peers {
                self.discover_peers().await?;
            }
        }
    }

    /// Attempts to connect to a peer
    async fn try_connect(&mut self, addr: SocketAddr) -> Result<()> {
        let hello = Message::Hello {
            node_id: self.shared.node_id.clone(),
            tier: self.shared.tier,
            version: crate::protocol::PROTOCOL_VERSION,
        };

        self.shared.message_tx.send((hello, addr)).await
            .map_err(|e| Error::network(format!("Failed to send Hello message: {}", e)))?;

        Ok(())
    }

    /// Maintains the peer list by removing stale peers
    async fn maintain_peers(&mut self) -> Result<()> {
        let now = SystemTime::now();
        self.inner.peers.retain(|_, peer| {
            if let Ok(elapsed) = now.duration_since(peer.last_seen) {
                elapsed < self.shared.config.peer_timeout
            } else {
                false
            }
        });

        Ok(())
    }

    /// Discovers new peers by querying existing peers
    async fn discover_peers(&mut self) -> Result<()> {
        // Create a list of peer addresses to query
        let peer_addrs: Vec<SocketAddr> = self.inner.peers.values()
            .map(|p| p.addr)
            .collect();

        // Create the sync request once
        let sync_request = Message::SyncRequest {
            node_id: self.shared.node_id.clone(),
            requested_components: vec![], // Empty for peer discovery
        };
        
        // Send request to each peer
        for addr in peer_addrs {
            self.shared.message_tx.send((sync_request.clone(), addr)).await
                .map_err(|e| Error::network(format!("Failed to send SyncRequest: {}", e)))?;
        }

        Ok(())
    }

    /// Updates a peer's information
    pub fn update_peer(&mut self, peer: Peer) -> Result<()> {
        if self.inner.peers.len() >= self.shared.config.max_peers && !self.inner.peers.contains_key(&peer.id) {
            return Ok(());
        }

        self.inner.peers.insert(peer.id.clone(), peer);
        Ok(())
    }

    /// Returns a list of current peers
    pub fn get_peers(&self) -> Vec<Peer> {
        self.inner.peers.values().cloned().collect()
    }

    /// Returns the number of current peers
    pub fn peer_count(&self) -> usize {
        self.inner.peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_peer_discovery() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let addr: SocketAddr = "127.0.0.1:4444".parse().unwrap();

        let config = DiscoveryConfig {
            initial_peers: vec![addr],
            discovery_interval: Duration::from_millis(100),
            ..Default::default()
        };

        let mut discovery = PeerDiscovery::new(
            node_id.clone(),
            Tier::base(),
            config,
            tx,
        );

        // Start discovery in background
        let discovery_handle = tokio::spawn(async move {
            discovery.run().await.unwrap();
        });

        // Should receive Hello message for initial peer
        if let Ok(Some((Message::Hello { node_id: id, .. }, addr))) = 
            timeout(Duration::from_millis(200), rx.recv()).await {
            assert_eq!(id, node_id);
        } else {
            panic!("Expected Hello message");
        }

        // Clean up
        discovery_handle.abort();
    }
}
