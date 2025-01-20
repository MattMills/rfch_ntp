//! Network management and peer communication module
//! 
//! This module handles network connections, peer discovery, and message routing.

mod connection;
mod discovery;

pub use self::connection::ConnectionManager;
pub use self::discovery::{PeerDiscovery, DiscoveryConfig};

use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::core::{Error, Result, NodeId, Peer, Config};
use crate::protocol::Message;

/// Network manager that coordinates connection and discovery
pub struct NetworkManager {
    /// Connection manager
    connection: ConnectionManager,
    /// Peer discovery
    discovery: PeerDiscovery,
    /// Message channel for internal communication
    message_tx: mpsc::Sender<(Message, SocketAddr)>,
}

/// Handle for sending messages through the network
#[derive(Clone)]
pub struct NetworkHandle {
    message_tx: mpsc::Sender<(Message, SocketAddr)>,
}

impl NetworkHandle {
    /// Sends a message to a peer
    pub async fn send_message(&self, message: Message, addr: SocketAddr) -> Result<()> {
        self.message_tx.send((message, addr)).await
            .map_err(|e| Error::network(format!("Failed to send message: {}", e)))
    }
}

impl NetworkManager {
    /// Creates a new network manager
    pub async fn new(config: Config, node_id: NodeId) -> Result<Self> {
        let (tx, _rx) = mpsc::channel::<(Message, SocketAddr)>(100);

        let connection = ConnectionManager::new(config.bind_addr, node_id.clone()).await?;
        
        let discovery_config = DiscoveryConfig {
            initial_peers: config.initial_peers,
            discovery_interval: Duration::from_secs(60),
            peer_timeout: config.peer_timeout,
            max_peers: config.max_peers,
            min_peers: config.min_peers,
        };

        let discovery = PeerDiscovery::new(
            node_id,
            crate::core::Tier::base(),
            discovery_config,
            tx.clone(),
        );

        Ok(NetworkManager {
            connection,
            discovery,
            message_tx: tx,
        })
    }

    /// Starts the network manager
    pub async fn run(&mut self) -> Result<()> {
        // Take ownership of connection
        let mut connection = std::mem::replace(
            &mut self.connection,
            // We'll never use this temporary value
            unsafe { std::mem::zeroed() }
        );

        // Spawn discovery task
        let mut discovery = self.discovery.clone();
        let discovery_handle = tokio::spawn(async move {
            discovery.run().await
        });

        // Run connection manager in this task
        let connection_handle = tokio::spawn(async move {
            connection.run().await
        });

        // Wait for both tasks to complete
        tokio::try_join!(
            async {
                discovery_handle
                    .await
                    .map_err(|e| Error::network(format!("Discovery task failed: {}", e)))??;
                Ok::<_, Error>(())
            },
            async {
                connection_handle
                    .await
                    .map_err(|e| Error::network(format!("Connection task failed: {}", e)))??;
                Ok::<_, Error>(())
            }
        )?;

        Ok(())
    }

    /// Returns a handle for sending messages
    pub fn handle(&self) -> NetworkHandle {
        NetworkHandle {
            message_tx: self.message_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_network_manager() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let config = Config {
            bind_addr: addr,
            initial_peers: vec![],
            min_peers: 1,
            max_peers: 10,
            heartbeat_interval: Duration::from_secs(1),
            peer_timeout: Duration::from_secs(5),
        };

        let node_id = NodeId::random();
        let mut manager = NetworkManager::new(config, node_id.clone()).await.unwrap();

        // Start manager in background
        let manager_handle = tokio::spawn(async move {
            manager.run().await.unwrap();
        });

        // Allow time for setup
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Clean up
        manager_handle.abort();
    }
}
