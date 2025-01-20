use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

use crate::core::{Error, Result, FrequencyComponent, NodeId, Peer, Precision, Tier};
use crate::protocol::message::Message;
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
    /// Frequency band allocation per tier
    pub tier_frequency_bands: HashMap<Tier, (f64, f64)>,
}

impl Default for SyncConfig {
    fn default() -> Self {
        let mut tier_bands = HashMap::new();
        // Base tier handles low frequencies (0.1-1 Hz)
        tier_bands.insert(Tier::new(0), (0.1, 1.0));
        // Higher tiers handle progressively higher frequencies
        tier_bands.insert(Tier::new(1), (1.0, 10.0));
        tier_bands.insert(Tier::new(2), (10.0, 100.0));

        SyncConfig {
            sample_rate: 1000.0,
            window_size: 1024,
            min_samples: 512,
            max_sample_age: Duration::from_secs(60),
            tier_frequency_bands: tier_bands,
        }
    }
}

/// Manages synchronization of frequency components across peers
pub struct SyncManager {
    /// Node's unique identifier
    node_id: NodeId,
    /// Current tier
    tier: Tier,
    /// Frequency analyzer
    analyzer: FrequencyAnalyzer,
    /// Configuration
    config: SyncConfig,
    /// Channel for sending messages
    message_tx: mpsc::Sender<Message>,
    /// Collected time samples
    samples: Vec<(SystemTime, f64)>,
    /// Current frequency components
    components: Vec<FrequencyComponent>,
    /// Known peers and their components
    peer_components: HashMap<NodeId, Vec<FrequencyComponent>>,
}

impl SyncManager {
    /// Creates a new synchronization manager
    pub fn new(
        node_id: NodeId,
        tier: Tier,
        config: SyncConfig,
        message_tx: mpsc::Sender<Message>,
    ) -> Self {
        let analyzer = FrequencyAnalyzer::new(config.sample_rate, config.window_size);
        SyncManager {
            node_id,
            tier,
            analyzer,
            config,
            message_tx,
            samples: Vec::new(),
            components: Vec::new(),
            peer_components: HashMap::new(),
        }
    }

    /// Starts the synchronization process
    pub async fn run(&mut self) -> Result<()> {
        use tokio::time::interval;
        let mut analysis_interval = interval(Duration::from_secs(1));

        loop {
            analysis_interval.tick().await;

            // Prune old samples
            self.prune_old_samples();

            // Perform analysis if we have enough samples
            if self.samples.len() >= self.config.min_samples {
                self.analyze().await?;
            }
        }
    }

    /// Adds a new time sample
    pub fn add_sample(&mut self, timestamp: SystemTime, value: f64) {
        self.samples.push((timestamp, value));
        self.prune_old_samples();
    }

    /// Updates components from a peer
    pub fn update_peer_components(
        &mut self,
        peer_id: NodeId,
        components: Vec<FrequencyComponent>,
    ) {
        self.peer_components.insert(peer_id, components);
    }

    /// Performs frequency analysis on collected samples
    pub async fn analyze(&mut self) -> Result<()> {
        // Ensure we have enough samples
        if self.samples.len() < self.config.min_samples {
            return Ok(());
        }

        // Extract values in chronological order
        let values: Vec<f64> = self.samples
            .iter()
            .map(|(_, v)| *v)
            .collect();

        // Perform frequency decomposition
        let components = self.analyzer.decompose(&values)?;

        // Filter components based on tier's frequency band
        if let Some((low_freq, high_freq)) = self.config.tier_frequency_bands.get(&self.tier) {
            self.components = components
                .into_iter()
                .filter(|c| c.frequency >= *low_freq && c.frequency <= *high_freq)
                .collect();
        }

        // Broadcast new components to peers
        let heartbeat = Message::Heartbeat {
            node_id: self.node_id.clone(),
            tier: self.tier,
            timestamp: SystemTime::now(),
            components: self.components.clone(),
        };

        self.message_tx.send(heartbeat).await
            .map_err(|e| Error::sync(format!("Failed to send heartbeat: {}", e)))?;

        Ok(())
    }

    /// Calculates current precision based on component stability
    pub fn calculate_precision(&self) -> Precision {
        self.analyzer.calculate_precision(&self.components)
    }

    /// Synthesizes timing signal from current components
    pub fn synthesize_signal(&self, num_samples: usize) -> Vec<f64> {
        // Combine local and peer components
        let mut all_components = self.components.clone();
        for components in self.peer_components.values() {
            all_components.extend(components.iter().cloned());
        }

        self.analyzer.synthesize(&all_components, num_samples)
    }

    /// Removes samples older than max_sample_age
    fn prune_old_samples(&mut self) {
        let cutoff = SystemTime::now()
            .checked_sub(self.config.max_sample_age)
            .unwrap_or_else(SystemTime::now);

        self.samples.retain(|(timestamp, _)| *timestamp >= cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_sample_analysis() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let config = SyncConfig::default();
        let mut manager = SyncManager::new(
            node_id.clone(),
            Tier::base(),
            config,
            tx,
        );

        // Add test samples (simple sine wave)
        let sample_period = Duration::from_secs_f64(1.0 / config.sample_rate);
        let freq = 0.5; // 0.5 Hz (within base tier band)
        
        for i in 0..config.min_samples {
            let time = SystemTime::now();
            let value = (2.0 * std::f64::consts::PI * freq * i as f64 
                / config.sample_rate).sin();
            manager.add_sample(time, value);
        }

        // Perform analysis
        manager.analyze().await.unwrap();

        // Should receive heartbeat with components
        if let Some(Message::Heartbeat { components, .. }) = rx.recv().await {
            assert!(!components.is_empty());
            // Should find the 0.5 Hz component
            let found_freq = components.iter()
                .find(|c| (c.frequency - freq).abs() < 0.1)
                .is_some();
            assert!(found_freq, "Failed to detect 0.5 Hz component");
        } else {
            panic!("Expected heartbeat message");
        }
    }
}
