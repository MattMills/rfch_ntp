use std::time::{Duration, SystemTime};
use tokio::time::interval;

use crate::core::{Error, Result};

/// Time source types
#[derive(Debug, Clone)]
pub enum TimeSource {
    /// System clock
    System,
    /// NTP server
    Ntp(String),
    /// PTP hardware clock
    Ptp(String),
    /// GPS receiver
    Gps(String),
}

/// Time source configuration
#[derive(Debug, Clone)]
pub struct TimeSourceConfig {
    /// Primary time source
    pub primary: TimeSource,
    /// Backup time sources
    pub backups: Vec<TimeSource>,
    /// Sampling interval
    pub sample_interval: Duration,
    /// Maximum allowed drift
    pub max_drift: Duration,
}

impl Default for TimeSourceConfig {
    fn default() -> Self {
        TimeSourceConfig {
            primary: TimeSource::System,
            backups: vec![],
            sample_interval: Duration::from_secs(1),
            max_drift: Duration::from_millis(100),
        }
    }
}

/// Manages time source sampling and drift compensation
pub struct TimeSourceManager {
    /// Configuration
    config: TimeSourceConfig,
    /// Current time source
    current_source: TimeSource,
    /// Last sample time
    last_sample: SystemTime,
    /// Accumulated drift
    drift: Duration,
}

impl TimeSourceManager {
    /// Creates a new time source manager
    pub fn new(config: TimeSourceConfig) -> Self {
        TimeSourceManager {
            current_source: config.primary.clone(),
            config,
            last_sample: SystemTime::now(),
            drift: Duration::from_secs(0),
        }
    }

    /// Starts the time source manager
    pub async fn run(&mut self) -> Result<()> {
        let mut sample_interval = interval(self.config.sample_interval);

        loop {
            sample_interval.tick().await;
            self.sample_time().await?;
        }
    }

    /// Takes a time sample from the current source
    async fn sample_time(&mut self) -> Result<()> {
        let now = match &self.current_source {
            TimeSource::System => SystemTime::now(),
            TimeSource::Ntp(server) => self.sample_ntp(server).await?,
            TimeSource::Ptp(device) => self.sample_ptp(device).await?,
            TimeSource::Gps(device) => self.sample_gps(device).await?,
        };

        // Calculate drift since last sample
        if let Ok(elapsed) = now.duration_since(self.last_sample) {
            let expected = self.config.sample_interval;
            if elapsed > expected {
                self.drift += elapsed - expected;
            } else {
                self.drift = self.drift.saturating_sub(expected - elapsed);
            }
        }

        // Switch to backup source if drift exceeds threshold
        if self.drift > self.config.max_drift {
            self.switch_to_backup()?;
        }

        self.last_sample = now;
        Ok(())
    }

    /// Samples time from an NTP server
    async fn sample_ntp(&self, server: &str) -> Result<SystemTime> {
        // TODO: Implement NTP client
        Err(Error::timing("NTP sampling not implemented"))
    }

    /// Samples time from a PTP hardware clock
    async fn sample_ptp(&self, device: &str) -> Result<SystemTime> {
        // TODO: Implement PTP client
        Err(Error::timing("PTP sampling not implemented"))
    }

    /// Samples time from a GPS receiver
    async fn sample_gps(&self, device: &str) -> Result<SystemTime> {
        // TODO: Implement GPS client
        Err(Error::timing("GPS sampling not implemented"))
    }

    /// Switches to a backup time source
    fn switch_to_backup(&mut self) -> Result<()> {
        // Find first working backup source
        for backup in &self.config.backups {
            match backup {
                TimeSource::System => {
                    self.current_source = TimeSource::System;
                    self.drift = Duration::from_secs(0);
                    return Ok(());
                }
                _ => continue, // Skip non-system sources for now
            }
        }

        Err(Error::timing("No available backup time sources"))
    }

    /// Returns the current drift from the time source
    pub fn get_drift(&self) -> Duration {
        self.drift
    }

    /// Returns the current time with drift compensation
    pub fn now(&self) -> SystemTime {
        SystemTime::now() + self.drift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_system_time_source() {
        let config = TimeSourceConfig::default();
        let mut manager = TimeSourceManager::new(config);

        // Sample time a few times
        for _ in 0..3 {
            manager.sample_time().await.unwrap();
            sleep(Duration::from_millis(100)).await;
        }

        // Drift should be small for system time
        assert!(manager.get_drift() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_backup_switching() {
        let config = TimeSourceConfig {
            primary: TimeSource::Ntp("pool.ntp.org".to_string()),
            backups: vec![TimeSource::System],
            ..Default::default()
        };

        let mut manager = TimeSourceManager::new(config);

        // NTP should fail and switch to system time
        manager.sample_time().await.unwrap();
        assert!(matches!(manager.current_source, TimeSource::System));
    }
}
