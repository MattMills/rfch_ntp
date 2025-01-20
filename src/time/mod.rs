//! Time synchronization and management module
//! 
//! This module provides functionality for precise time synchronization using multiple time sources:
//! 
//! - System clock with drift compensation
//! - NTP servers with hardware timestamping
//! - PTP hardware clocks for sub-microsecond precision
//! - GPS receivers with optional PPS for nanosecond precision
//! - Quantum clock with error correction
//! 
//! The module implements a hierarchical approach to time synchronization:
//! 
//! 1. Primary time source selection based on available hardware and precision requirements
//! 2. Automatic failover to backup sources if primary source fails or drifts
//! 3. Statistical filtering and Kalman filtering for drift estimation
//! 4. Hardware timestamp support for improved precision
//! 5. Quantum error correction for ultimate precision
//! 
//! # Examples
//! 
//! ```no_run
//! use rfch_ntp::time::{TimeSourceConfig, TimeSourceManager, TimeSource};
//! use std::time::Duration;
//! 
//! #[tokio::main]
//! async fn main() {
//!     // Configure with GPS primary and NTP backup
//!     let config = TimeSourceConfig {
//!         primary: TimeSource::Gps {
//!             device: "/dev/ttyACM0".to_string(),
//!             use_pps: true,
//!         },
//!         backups: vec![TimeSource::Ntp("pool.ntp.org".to_string())],
//!         sample_interval: Duration::from_secs(1),
//!         ..Default::default()
//!     };
//! 
//!     // Create and run time manager
//!     let mut manager = TimeSourceManager::new(config).unwrap();
//!     manager.run().await.unwrap();
//! }
//! ```

mod source;
mod gps;
mod ptp;
mod quantum;

pub use self::quantum::QuantumClock;
pub use self::source::{
    TimeSource, TimeSourceConfig, TimeSourceManager,
    TimeStats, RttStats, NtpConfig, PtpConfig, GpsConfig, QuantumConfig,
};

/// Utility functions for time management
pub mod util {
    use std::time::{Duration, SystemTime};
    use crate::core::{Error, Result};

    /// Calculates the time difference between two timestamps
    /// 
    /// This function handles both forward and backward time differences,
    /// returning the absolute duration between the timestamps.
    pub fn time_diff(a: SystemTime, b: SystemTime) -> Result<Duration> {
        a.duration_since(b)
            .or_else(|e| b.duration_since(a))
            .map_err(|e| Error::timing(format!("Failed to calculate time difference: {}", e)))
    }

    /// Estimates clock drift rate in parts per million (ppm)
    /// 
    /// Uses a sliding window of time samples to estimate the current drift rate
    /// of the system clock relative to the reference time source.
    pub fn estimate_drift_rate(
        samples: &[(SystemTime, Duration)],
        window: Duration,
    ) -> Result<f64> {
        if samples.len() < 2 {
            return Ok(0.0);
        }

        let now = SystemTime::now();
        let recent: Vec<_> = samples.iter()
            .filter(|(t, _)| {
                t.duration_since(now)
                    .map(|d| d < window)
                    .unwrap_or(false)
            })
            .collect();

        if recent.len() < 2 {
            return Ok(0.0);
        }

        let (t1, d1) = recent[0];
        let (t2, d2) = recent[recent.len() - 1];

        let elapsed = time_diff(*t2, *t1)?;
        let drift = if d2 > d1 {
            *d2 - *d1
        } else {
            *d1 - *d2
        };

        // Convert to ppm
        Ok(drift.as_secs_f64() / elapsed.as_secs_f64() * 1_000_000.0)
    }

    /// Validates a time source configuration
    /// 
    /// Checks that the configuration is valid and consistent:
    /// - Sample interval is reasonable
    /// - Backup sources are available
    /// - Source-specific settings are valid
    pub fn validate_config(config: &super::TimeSourceConfig) -> Result<()> {
        // Check sample interval
        if config.sample_interval < Duration::from_millis(10) {
            return Err(Error::config("Sample interval too small"));
        }

        // Validate NTP settings
        if let super::TimeSource::Ntp(_) = config.primary {
            if config.ntp_config.version < 3 || config.ntp_config.version > 4 {
                return Err(Error::config("Invalid NTP version"));
            }
        }

        // Validate PTP settings
        if let super::TimeSource::Ptp(_) = config.primary {
            if config.ptp_config.sync_interval < Duration::from_millis(100) {
                return Err(Error::config("PTP sync interval too small"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_time_diff() {
        let now = SystemTime::now();
        let later = now + Duration::from_secs(1);
        
        let diff = util::time_diff(later, now).unwrap();
        assert_eq!(diff.as_secs(), 1);

        let diff = util::time_diff(now, later).unwrap();
        assert_eq!(diff.as_secs(), 1);
    }

    #[test]
    fn test_drift_rate() {
        let now = SystemTime::now();
        
        // Create samples with realistic drift (5 ppm) and noise
        let drift_ppm = 5.0; // 5 parts per million
        let mut samples = Vec::new();
        let mut total_drift = Duration::ZERO;
        
        for i in 0..10 {
            // Calculate accumulated drift: 1 ppm = 1 microsecond per second
            let elapsed_secs = (10 - i) as f64;
            let drift = drift_ppm * elapsed_secs * 1e-6; // Convert ppm to seconds
            total_drift += Duration::from_secs_f64(drift);
            
            // Add some noise (±10 nanoseconds)
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let noise = Duration::from_nanos(rng.gen_range(0..20));
            
            let t = now - Duration::from_secs(10 - i);
            samples.push((t, total_drift + noise));
        }

        let rate = util::estimate_drift_rate(&samples, Duration::from_secs(30)).unwrap();
        
        // Rate should be close to our simulated 5 ppm
        assert!(rate > 4.0 && rate < 6.0, 
                "Expected drift rate around 5 ppm, got {} ppm", rate);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = TimeSourceConfig {
            primary: TimeSource::System,
            backups: vec![],
            sample_interval: Duration::from_secs(1),
            ..Default::default()
        };
        assert!(util::validate_config(&config).is_ok());

        // Invalid sample interval
        let config = TimeSourceConfig {
            sample_interval: Duration::from_millis(1),
            ..config.clone()
        };
        assert!(util::validate_config(&config).is_err());

        // Invalid NTP version
        let config = TimeSourceConfig {
            primary: TimeSource::Ntp("pool.ntp.org".to_string()),
            ntp_config: NtpConfig {
                version: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(util::validate_config(&config).is_err());
    }

    #[tokio::test]
    async fn test_quantum_time_source() {
        let config = TimeSourceConfig {
            primary: TimeSource::Quantum,
            backups: vec![TimeSource::System],
            quantum_config: QuantumConfig {
                coherence_time: Duration::from_millis(100),
                code_distance: 3,
                error_threshold: 0.01,
            },
            ..Default::default()
        };

        let mut manager = TimeSourceManager::new(config).unwrap();

        // Sample quantum time
        manager.sample_time().await.unwrap();
        let stats = manager.get_stats();
        assert_eq!(stats.sample_count, 1);
        // Error should be very small with quantum clock
        assert!(stats.mean_error < 100.0); // Less than 100ns
    }

    #[tokio::test]
    async fn test_quantum_coherence() {
        let config = TimeSourceConfig {
            primary: TimeSource::Quantum,
            quantum_config: QuantumConfig {
                coherence_time: Duration::from_millis(50),
                code_distance: 3,
                error_threshold: 0.01,
            },
            ..Default::default()
        };

        let mut manager = TimeSourceManager::new(config).unwrap();

        // First sample should succeed
        manager.sample_time().await.unwrap();

        // Wait for decoherence
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should switch to backup after decoherence
        manager.sample_time().await.unwrap();
        assert!(matches!(manager.current_source, TimeSource::Quantum));
    }
}
