use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

use crate::core::{Error, Result, NodeId, Peer, Tier};
use crate::protocol::Message;
use super::frequency::FrequencyAnalyzer;

/// Configuration for synchronization
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// FFT window size
    pub window_size: usize,
    /// Minimum samples required for analysis
    pub min_samples: usize,
    /// Maximum age of samples to consider
    pub max_sample_age: Duration,
    /// Analysis interval
    pub analysis_interval: Duration,
}

impl Default for SyncConfig {
    fn default() -> Self {
        SyncConfig {
            sample_rate: 1000.0,
            window_size: 1024,
            min_samples: 512,
            max_sample_age: Duration::from_secs(60),
            analysis_interval: Duration::from_secs(1),
        }
    }
}

/// Manages frequency analysis and synchronization
pub struct SyncManager {
    /// Node ID
    node_id: NodeId,
    /// Current tier
    tier: Tier,
    /// Configuration
    config: SyncConfig,
    /// Frequency analyzer
    analyzer: FrequencyAnalyzer,
    /// Channel for sending messages
    message_tx: mpsc::Sender<(Message, SocketAddr)>,
    /// Collected time samples
    samples: Vec<(SystemTime, f64)>,
    /// Known peers and their frequency components
    peer_components: HashMap<NodeId, Vec<crate::core::FrequencyComponent>>,
}

impl SyncManager {
    /// Creates a new synchronization manager
    pub fn new(
        node_id: NodeId,
        tier: Tier,
        config: SyncConfig,
        message_tx: mpsc::Sender<(Message, SocketAddr)>,
    ) -> Self {
        SyncManager {
            node_id,
            tier,
            analyzer: FrequencyAnalyzer::new(config.sample_rate, config.window_size),
            config,
            message_tx,
            samples: Vec::new(),
            peer_components: HashMap::new(),
        }
    }

    /// Starts the synchronization process
    pub async fn run(&mut self) -> Result<()> {
        use tokio::time::interval;
        let mut analysis_interval = interval(self.config.analysis_interval);

        loop {
            analysis_interval.tick().await;

            // Prune old samples
            self.prune_old_samples();

            // Perform analysis if we have enough samples
            if self.samples.len() >= self.config.min_samples {
                self.analyze_samples().await?;
            }
        }
    }

    /// Adds a new time sample
    pub fn add_sample(&mut self, timestamp: SystemTime, value: f64) {
        self.samples.push((timestamp, value));
    }

    /// Updates components from a peer
    pub fn update_peer_components(
        &mut self,
        peer_id: NodeId,
        components: Vec<crate::core::FrequencyComponent>,
    ) {
        self.peer_components.insert(peer_id, components);
    }

    /// Removes samples older than max_sample_age
    fn prune_old_samples(&mut self) {
        let cutoff = SystemTime::now()
            .checked_sub(self.config.max_sample_age)
            .unwrap_or_else(SystemTime::now);

        self.samples.retain(|(timestamp, _)| *timestamp >= cutoff);
    }

    /// Analyzes collected samples and broadcasts results
    async fn analyze_samples(&mut self) -> Result<()> {
        // Extract sample values
        let values: Vec<f64> = self.samples.iter().map(|(_, v)| *v).collect();

        // Perform frequency analysis
        let components = self.analyzer.decompose(&values)?;

        // Broadcast results to peers
        let heartbeat = Message::Heartbeat {
            node_id: self.node_id.clone(),
            tier: self.tier,
            timestamp: SystemTime::now(),
            components: components.clone(),
        };

        // Send to all known peers
        for peer_id in self.peer_components.keys() {
            if let Some(peer) = self.get_peer_addr(peer_id) {
                self.message_tx.send((heartbeat.clone(), peer)).await
                    .map_err(|e| Error::sync(format!("Failed to send heartbeat: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Gets a peer's address (placeholder - would come from peer manager)
    fn get_peer_addr(&self, _peer_id: &NodeId) -> Option<SocketAddr> {
        // TODO: Implement peer address lookup
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_sync_manager() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let config = SyncConfig::default();

        let mut manager = SyncManager::new(
            node_id.clone(),
            Tier::base(),
            config,
            tx,
        );

        // Add some test samples
        let now = SystemTime::now();
        for i in 0..1000 {
            let value = (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 1000.0).sin();
            manager.add_sample(now, value);
        }

        // Start manager in background
        let manager_handle = tokio::spawn(async move {
            manager.run().await.unwrap();
        });

        // Allow time for analysis
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Clean up
        manager_handle.abort();
    }
}
