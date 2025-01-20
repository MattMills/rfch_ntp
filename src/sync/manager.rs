use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

use crate::core::{Error, Result, FrequencyComponent, NodeId, Peer, Precision, Tier};
use crate::protocol::message::Message;
use super::frequency::{FrequencyAnalyzer, PhaseSpace, WaveletDecomposition};

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
    /// Collected time samples with timestamps
    samples: Vec<(SystemTime, f64)>,
    /// Current frequency components
    components: Vec<FrequencyComponent>,
    /// Current phase space state
    phase_space: Option<PhaseSpace>,
    /// Current wavelet decomposition
    wavelet_decomp: Option<WaveletDecomposition>,
    /// Known peers and their components
    peer_components: HashMap<NodeId, Vec<FrequencyComponent>>,
    /// Peer phase coherence metrics
    peer_coherence: HashMap<NodeId, f64>,
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
            phase_space: None,
            wavelet_decomp: None,
            peer_components: HashMap::new(),
            peer_coherence: HashMap::new(),
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

    /// Updates components and coherence metrics from a peer
    pub fn update_peer_components(
        &mut self,
        peer_id: NodeId,
        components: Vec<FrequencyComponent>,
        coherence: f64,
    ) {
        self.peer_components.insert(peer_id.clone(), components);
        self.peer_coherence.insert(peer_id, coherence);
    }

    /// Performs comprehensive signal analysis on collected samples
    pub async fn analyze(&mut self) -> Result<()> {
        // Ensure we have enough samples
        if self.samples.len() < self.config.min_samples {
            return Ok(());
        }

        // Extract values and timestamps
        let (timestamps, values): (Vec<_>, Vec<_>) = self.samples.iter()
            .cloned()
            .unzip();

        // Perform wavelet decomposition
        self.wavelet_decomp = Some(self.analyzer.wavelet_transform(&values)?);

        // Map to phase space
        self.phase_space = Some(self.analyzer.phase_space_mapping(&values, &timestamps)?);

        // Perform frequency decomposition
        let mut components = self.analyzer.decompose(&values)?;

        // Filter components based on tier's frequency band
        if let Some((low_freq, high_freq)) = self.config.tier_frequency_bands.get(&self.tier) {
            components.retain(|c| c.frequency >= *low_freq && c.frequency <= *high_freq);
        }

        // Refine components using wavelet information
        if let Some(ref wavelet) = self.wavelet_decomp {
            self.refine_components(&mut components, wavelet);
        }

        self.components = components;

        // Calculate current phase coherence
        let coherence = self.phase_space.as_ref().map_or(0.0, |ps| ps.coherence);

        // Broadcast state to peers
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

    /// Refines frequency components using wavelet information
    fn refine_components(&self, components: &mut Vec<FrequencyComponent>, wavelet: &WaveletDecomposition) {
        for component in components.iter_mut() {
            // Find closest wavelet scale
            if let Some(scale_idx) = wavelet.frequencies.iter()
                .position(|&f| (f - component.frequency).abs() < 0.1)
            {
                // Use wavelet coefficients to refine amplitude and phase
                let coeffs = &wavelet.coefficients[scale_idx];
                let mean_coeff = coeffs.iter()
                    .map(|c| c.norm())
                    .sum::<f64>() / coeffs.len() as f64;
                
                // Adjust amplitude based on wavelet energy
                component.amplitude *= Complex64::new(mean_coeff, 0.0);
            }
        }
    }

    /// Calculates current precision based on multiple metrics
    pub fn calculate_precision(&self) -> Precision {
        // Get phase space coherence
        let phase_coherence = self.phase_space
            .as_ref()
            .map(|ps| ps.coherence)
            .unwrap_or(0.0);

        // Calculate peer-averaged coherence
        let peer_coherence = if !self.peer_coherence.is_empty() {
            self.peer_coherence.values().sum::<f64>() / self.peer_coherence.len() as f64
        } else {
            0.0
        };

        // Calculate combined precision using all metrics
        self.analyzer.calculate_precision(
            &self.components,
            self.phase_space.as_ref().unwrap_or(&PhaseSpace {
                coordinates: Vec::new(),
                timestamps: Vec::new(),
                coherence: 0.0,
            })
        )
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
    use std::f64::consts::PI;

    #[tokio::test]
    async fn test_sample_analysis() {
        let (tx, mut rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let config = SyncConfig::default();
        let mut manager = SyncManager::new(
            node_id.clone(),
            Tier::base(),
            config.clone(),
            tx,
        );

        // Add test samples (simple sine wave)
        let sample_period = Duration::from_secs_f64(1.0 / config.sample_rate);
        let freq = 0.5; // 0.5 Hz (within base tier band)
        
        for i in 0..config.min_samples {
            let time = SystemTime::now();
            let value = (2.0 * PI * freq * i as f64 / config.sample_rate).sin();
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

    #[tokio::test]
    async fn test_phase_coherence() {
        let (tx, _rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let config = SyncConfig::default();
        let mut manager = SyncManager::new(
            node_id.clone(),
            Tier::base(),
            config.clone(),
            tx,
        );

        // Add coherent samples (perfect sine wave)
        let freq = 0.5;
        for i in 0..config.min_samples {
            let time = SystemTime::now();
            let value = (2.0 * PI * freq * i as f64 / config.sample_rate).sin();
            manager.add_sample(time, value);
        }

        // Analyze and check phase coherence
        manager.analyze().await.unwrap();
        assert!(manager.phase_space.is_some());
        let coherence = manager.phase_space.as_ref().unwrap().coherence;
        assert!(coherence > 0.9, "Expected high phase coherence for perfect sine wave");
    }

    #[tokio::test]
    async fn test_wavelet_refinement() {
        let (tx, _rx) = mpsc::channel(32);
        let node_id = NodeId::random();
        let config = SyncConfig::default();
        let mut manager = SyncManager::new(
            node_id.clone(),
            Tier::base(),
            config.clone(),
            tx,
        );

        // Add multi-frequency signal
        let freq1 = 0.5;
        let freq2 = 0.7;
        for i in 0..config.min_samples {
            let time = SystemTime::now();
            let t = i as f64 / config.sample_rate;
            let value = (2.0 * PI * freq1 * t).sin() + 0.5 * (2.0 * PI * freq2 * t).sin();
            manager.add_sample(time, value);
        }

        // Analyze and check wavelet decomposition
        manager.analyze().await.unwrap();
        assert!(manager.wavelet_decomp.is_some());
        
        // Should detect both frequencies
        let components = &manager.components;
        assert!(components.iter().any(|c| (c.frequency - freq1).abs() < 0.1));
        assert!(components.iter().any(|c| (c.frequency - freq2).abs() < 0.1));
    }
}
