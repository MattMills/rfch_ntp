use std::net::SocketAddr;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio_util::codec::{Decoder, Encoder};
use bytes::BytesMut;

use crate::core::{Error, Result, NodeId, Peer};
use crate::protocol::{
    codec::MessageCodec,
    message::Message,
};

/// Shared state between connection manager instances
#[derive(Clone)]
struct SharedState {
    /// Local node ID
    node_id: NodeId,
    /// UDP socket for communication
    socket: Arc<UdpSocket>,
    /// Channel for outgoing messages
    message_tx: mpsc::Sender<(Message, SocketAddr)>,
}

/// Inner state of the connection manager
struct Inner {
    /// Message codec
    codec: MessageCodec,
    /// Channel for incoming messages
    message_rx: mpsc::Receiver<(Message, SocketAddr)>,
    /// Buffer for receiving data
    recv_buffer: BytesMut,
}

/// Manages network connections and message routing
pub struct ConnectionManager {
    /// Shared state
    shared: Arc<SharedState>,
    /// Inner state
    inner: Inner,
}

impl ConnectionManager {
    /// Creates a new connection manager
    pub async fn new(bind_addr: SocketAddr, node_id: NodeId) -> Result<Self> {
        let socket = Arc::new(
            UdpSocket::bind(bind_addr)
                .await
                .map_err(|e| Error::network(format!("Failed to bind socket: {}", e)))?
        );

        let (tx, rx) = mpsc::channel(100);

        let shared = Arc::new(SharedState {
            node_id,
            socket,
            message_tx: tx,
        });

        let inner = Inner {
            codec: MessageCodec::new(),
            message_rx: rx,
            recv_buffer: BytesMut::with_capacity(8192),
        };

        Ok(ConnectionManager { shared, inner })
    }

    /// Starts the connection manager
    pub async fn run(&mut self) -> Result<()> {
        let mut send_buffer = BytesMut::new();
        let mut inner = Inner {
            codec: MessageCodec::new(),
            message_rx: std::mem::replace(&mut self.inner.message_rx, mpsc::channel(1).1),
            recv_buffer: std::mem::take(&mut self.inner.recv_buffer),
        };

        loop {
            tokio::select! {
                // Handle outgoing messages
                Some((message, addr)) = inner.message_rx.recv() => {
                    send_buffer.clear();
                    inner.codec.encode(message, &mut send_buffer)?;
                    self.shared.socket.send_to(&send_buffer, addr).await
                        .map_err(|e| Error::network(format!("Failed to send message: {}", e)))?;
                }

                // Handle incoming messages
                Ok((size, addr)) = {
                    let buf = &mut inner.recv_buffer;
                    self.shared.socket.recv_from(buf)
                } => {
                    let data = inner.recv_buffer.split_to(size);
                    if let Some(message) = inner.codec.decode(&mut data.freeze().into())? {
                        self.handle_message(message, addr).await?;
                    }
                }
            }
        }
    }

    /// Handles an incoming message
    async fn handle_message(&mut self, message: Message, addr: SocketAddr) -> Result<()> {
        match &message {
            Message::Hello { node_id, tier, version } => {
                // Respond with Welcome message
                let response = Message::Welcome {
                    node_id: self.shared.node_id.clone(),
                    reference_time: SystemTime::now(),
                    network_tiers: vec![], // TODO: Get current network tiers
                };
                self.shared.message_tx.send((response, addr)).await
                    .map_err(|e| Error::network(format!("Failed to send Welcome message: {}", e)))?;
            }

            Message::Heartbeat { node_id, components, .. } => {
                // Forward heartbeat to other peers
                // TODO: Implement selective forwarding based on tier
            }

            // Handle other message types...
            _ => {}
        }

        Ok(())
    }

    /// Sends a message to a peer
    pub async fn send_message(&self, message: Message, addr: SocketAddr) -> Result<()> {
        self.shared.message_tx.send((message, addr)).await
            .map_err(|e| Error::network(format!("Failed to queue message: {}", e)))
    }

    /// Returns the local socket address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.shared.socket.local_addr()
            .map_err(|e| Error::network(format!("Failed to get local address: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_connection_establishment() {
        let addr1: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let node1_id = NodeId::random();
        let node2_id = NodeId::random();

        let mut manager1 = ConnectionManager::new(addr1, node1_id.clone()).await.unwrap();
        let mut manager2 = ConnectionManager::new(addr2, node2_id.clone()).await.unwrap();

        let addr1 = manager1.local_addr().unwrap();
        let addr2 = manager2.local_addr().unwrap();

        // Start managers
        let manager1_handle = tokio::spawn(async move {
            manager1.run().await.unwrap();
        });

        let manager2_handle = tokio::spawn(async move {
            manager2.run().await.unwrap();
        });

        // Allow time for setup
        sleep(Duration::from_millis(100)).await;

        // Clean up
        manager1_handle.abort();
        manager2_handle.abort();
    }
}
