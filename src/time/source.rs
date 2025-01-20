use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, SystemTime};
use tokio::time::interval;
use nalgebra::DVector;

use crate::core::{Error, Result};
use super::gps::GpsReceiver;

/// Hardware timestamp information
#[derive(Debug, Clone)]
pub struct HardwareTimestamp {
    /// Software timestamp
    pub software: SystemTime,
    /// Hardware timestamp (nanoseconds since epoch)
    pub hardware: u64,
    /// Timestamp source
    pub source: TimeSource,
    /// Estimated error bounds (nanoseconds)
    pub error_bound: u64,
}

/// Time source types
#[derive(Debug, Clone)]
pub enum TimeSource {
    /// System clock
    System,
    /// NTP server
    Ntp(String),
    /// PTP hardware clock
    Ptp(String),
    /// GPS receiver with optional PPS
    Gps {
        /// Device path
        device: String,
        /// Use PPS if available
        use_pps: bool,
    },
    /// Quantum clock
    Quantum,
}

/// Time sample with statistical information
#[derive(Debug, Clone)]
struct TimeSample {
    /// Timestamp
    time: SystemTime,
    /// Hardware timestamp if available
    hw_timestamp: Option<HardwareTimestamp>,
    /// Round-trip time for network sources
    rtt: Option<Duration>,
    /// Estimated error (nanoseconds)
    error: u64,
    /// Source that provided the sample
    source: TimeSource,
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
    /// Sample window size for statistics
    pub window_size: usize,
    /// NTP specific settings
    pub ntp_config: NtpConfig,
    /// PTP specific settings
    pub ptp_config: PtpConfig,
    /// GPS specific settings
    pub gps_config: GpsConfig,
    /// Quantum specific settings
    pub quantum_config: QuantumConfig,
}

/// Quantum configuration
#[derive(Debug, Clone)]
pub struct QuantumConfig {
    /// Coherence time (how long quantum state remains stable)
    pub coherence_time: Duration,
    /// Error correction code distance
    pub code_distance: usize,
    /// Error threshold for correction
    pub error_threshold: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        QuantumConfig {
            coherence_time: Duration::from_millis(100),
            code_distance: 3,
            error_threshold: 0.01,
        }
    }
}

/// NTP configuration
#[derive(Debug, Clone)]
pub struct NtpConfig {
    /// NTP version (3 or 4)
    pub version: u8,
    /// Poll interval
    pub poll_interval: Duration,
    /// Minimum number of samples
    pub min_samples: usize,
}

impl Default for NtpConfig {
    fn default() -> Self {
        NtpConfig {
            version: 4,
            poll_interval: Duration::from_secs(16),
            min_samples: 8,
        }
    }
}

/// PTP configuration
#[derive(Debug, Clone)]
pub struct PtpConfig {
    /// Domain number
    pub domain: u8,
    /// Sync interval
    pub sync_interval: Duration,
    /// Hardware timestamp mode
    pub hw_timestamp: bool,
}

impl Default for PtpConfig {
    fn default() -> Self {
        PtpConfig {
            domain: 0,
            sync_interval: Duration::from_secs(1),
            hw_timestamp: true,
        }
    }
}

/// GPS configuration
#[derive(Debug, Clone)]
pub struct GpsConfig {
    /// Serial device path
    pub device: String,
    /// Baud rate
    pub baud_rate: u32,
    /// Enable PPS if available
    pub use_pps: bool,
}

impl Default for GpsConfig {
    fn default() -> Self {
        GpsConfig {
            device: "/dev/ttyACM0".to_string(),
            baud_rate: 9600,
            use_pps: true,
        }
    }
}

impl Default for TimeSourceConfig {
    fn default() -> Self {
        TimeSourceConfig {
            // Use system clock as primary source
            primary: TimeSource::System,
            // Use NTP pool as backup
            backups: vec![TimeSource::Ntp("pool.ntp.org".to_string())],
            // Sample more frequently for better frequency decomposition
            sample_interval: Duration::from_millis(100),
            // Allow larger drift before switching to backup
            max_drift: Duration::from_millis(500),
            // Use larger window for better frequency analysis
            window_size: 256,
            // Configure NTP for better precision
            ntp_config: NtpConfig {
                version: 4,
                poll_interval: Duration::from_secs(8), // Poll more frequently
                min_samples: 16, // More samples for better statistics
            },
            // Keep default PTP config
            ptp_config: PtpConfig::default(),
            // Keep default GPS config
            gps_config: GpsConfig::default(),
            // Keep default quantum config
            quantum_config: QuantumConfig::default(),
        }
    }
}

/// Statistics about time samples
#[derive(Debug, Clone)]
pub struct TimeStats {
    /// Number of samples in window
    pub sample_count: usize,
    /// Minimum error (nanoseconds)
    pub min_error: u64,
    /// Maximum error (nanoseconds)
    pub max_error: u64,
    /// Mean error (nanoseconds)
    pub mean_error: f64,
    /// RTT statistics if available
    pub rtt_stats: Option<RttStats>,
}

/// Round-trip time statistics
#[derive(Debug, Clone)]
pub struct RttStats {
    /// Minimum RTT
    pub min: Duration,
    /// Maximum RTT
    pub max: Duration,
    /// Mean RTT
    pub mean: Duration,
    /// Standard deviation of RTT
    pub stddev: Duration,
}

/// Manages time source sampling and drift compensation
pub struct TimeSourceManager {
    /// Configuration
    config: TimeSourceConfig,
    /// Current time source
    pub(crate) current_source: TimeSource,
    /// Recent time samples
    samples: VecDeque<TimeSample>,
    /// Kalman filter state
    drift_estimate: f64,
    /// Kalman filter covariance
    drift_covariance: f64,
    /// NTP socket for hardware timestamps
    ntp_socket: Option<Arc<Mutex<UdpSocket>>>,
    /// PTP device handle
    ptp_device: Option<String>,
    /// GPS receiver handle
    gps_receiver: Option<Arc<Mutex<GpsReceiver>>>,
    /// Quantum clock handle
    quantum_clock: Option<Arc<Mutex<crate::time::QuantumClock>>>,
}

impl TimeSourceManager {
    /// Creates a new time source manager
    pub fn new(config: TimeSourceConfig) -> Result<Self> {
        // Initialize NTP socket with hardware timestamping if available
        let ntp_socket = match &config.primary {
            TimeSource::Ntp(_) => {
                let socket = UdpSocket::bind("0.0.0.0:0")?;
                socket.set_nonblocking(true)?;
                Some(Arc::new(Mutex::new(socket)))
            }
            _ => None,
        };

        // Initialize GPS receiver if needed
        let gps_receiver = match &config.primary {
            TimeSource::Gps { device, use_pps } => {
                Some(Arc::new(Mutex::new(GpsReceiver::open(device, config.gps_config.baud_rate)?)))
            }
            _ => None,
        };

        // Initialize quantum clock if needed
        let quantum_clock = match &config.primary {
            TimeSource::Quantum => {
                Some(Arc::new(Mutex::new(crate::time::QuantumClock::new(config.quantum_config.coherence_time)?)))
            }
            _ => None,
        };

        Ok(TimeSourceManager {
            current_source: config.primary.clone(),
            config,
            samples: VecDeque::with_capacity(64),
            drift_estimate: 0.0,
            drift_covariance: 1e-6,
            ntp_socket,
            ptp_device: None,
            gps_receiver,
            quantum_clock,
        })
    }

    /// Starts the time source manager
    pub async fn run(&mut self) -> Result<()> {
        loop {
            self.sample_time().await?;
            self.update_drift().await;
            tokio::time::sleep(self.config.sample_interval).await;
        }
    }

    /// Takes a time sample from the current source
    pub async fn sample_time(&mut self) -> Result<()> {
        let sample = {
            match &self.current_source {
                TimeSource::System => self.sample_system()?,
                TimeSource::Ntp(server) => self.sample_ntp(server).await?,
                TimeSource::Ptp(device) => self.sample_ptp(device).await?,
                TimeSource::Gps { device, use_pps } => {
                    let device = device.clone();
                    self.sample_gps(&device, *use_pps).await?
                },
                TimeSource::Quantum => self.sample_quantum().await?,
            }
        };

        // Add sample to window
        self.samples.push_back(sample);
        if self.samples.len() > self.config.window_size {
            self.samples.pop_front();
        }

        Ok(())
    }

    /// Samples time from quantum clock
    async fn sample_quantum(&mut self) -> Result<TimeSample> {
        let clock = self.quantum_clock.as_ref()
            .ok_or_else(|| Error::timing("Quantum clock not initialized"))?;
        let mut clock = clock.lock().await;

        let hw_ts = clock.get_timestamp()?;

        Ok(TimeSample {
            time: hw_ts.software,
            hw_timestamp: Some(hw_ts.clone()),
            rtt: None,
            error: hw_ts.error_bound,
            source: TimeSource::Quantum,
        })
    }

    /// Samples system time
    fn sample_system(&self) -> Result<TimeSample> {
        Ok(TimeSample {
            time: SystemTime::now(),
            hw_timestamp: None,
            rtt: None,
            error: 1_000_000, // 1ms error assumed for system time
            source: TimeSource::System,
        })
    }

    /// Samples time from an NTP server
    async fn sample_ntp(&self, server: &str) -> Result<TimeSample> {
        let socket = self.ntp_socket.as_ref()
            .ok_or_else(|| Error::timing("NTP socket not initialized"))?;

        // NTP packet structure (48 bytes)
        let mut packet = [0u8; 48];
        packet[0] = (self.config.ntp_config.version << 3) | 3; // Version 4, client mode

        // Send request with hardware timestamp
        let t1 = SystemTime::now();
        let socket = socket.lock().await;
        socket.send_to(&packet, format!("{}:123", server))?;

        // Receive response with hardware timestamp
        let mut response = [0u8; 48];
        let (_, _) = socket.recv_from(&mut response)?;
        drop(socket); // Release lock
        let t4 = SystemTime::now();

        // Extract timestamps from packet
        let t2_secs = u32::from_be_bytes(response[32..36].try_into().unwrap());
        let t2_frac = u32::from_be_bytes(response[36..40].try_into().unwrap());
        let t3_secs = u32::from_be_bytes(response[40..44].try_into().unwrap());
        let t3_frac = u32::from_be_bytes(response[44..48].try_into().unwrap());

        // Convert NTP timestamps to Duration
        let t2 = Duration::new(t2_secs as u64, (t2_frac as f64 * 1e9) as u32);
        let t3 = Duration::new(t3_secs as u64, (t3_frac as f64 * 1e9) as u32);

        // Calculate offset and round-trip time
        let rtt = t4.duration_since(t1)?;
        let t1_unix = t1.duration_since(SystemTime::UNIX_EPOCH)?;
        let t4_unix = t4.duration_since(SystemTime::UNIX_EPOCH)?;
        
        let offset = ((t2 + t3) - (t1_unix + t4_unix)) / 2;

        Ok(TimeSample {
            time: SystemTime::now() + offset,
            hw_timestamp: None, // Hardware timestamps not implemented yet
            rtt: Some(rtt),
            error: (rtt.as_nanos() / 2) as u64,
            source: TimeSource::Ntp(server.to_string()),
        })
    }

    /// Samples time from a PTP hardware clock
    async fn sample_ptp(&self, device: &str) -> Result<TimeSample> {
        // TODO: Implement PTP hardware timestamping
        Err(Error::timing("PTP sampling not implemented"))
    }

    /// Samples time from a GPS receiver
    async fn sample_gps(&mut self, device: &str, use_pps: bool) -> Result<TimeSample> {
        let receiver = self.gps_receiver.as_ref()
            .ok_or_else(|| Error::timing("GPS receiver not initialized"))?;
        let mut receiver = receiver.lock().await;

        let hw_ts = receiver.get_timestamp()?;

        Ok(TimeSample {
            time: hw_ts.software,
            hw_timestamp: Some(hw_ts.clone()),
            rtt: None,
            error: hw_ts.error_bound,
            source: TimeSource::Gps {
                device: device.to_string(),
                use_pps,
            },
        })
    }

    /// Updates drift estimate using Kalman filter with adaptive noise
    async fn update_drift(&mut self) {
        if self.samples.len() < 2 {
            return;
        }

        // Calculate time differences and drifts
        let mut drifts = Vec::new();
        let mut weights = Vec::new();
        let mut total_elapsed = Duration::ZERO;

        for (s1, s2) in self.samples.iter().zip(self.samples.iter().skip(1)) {
            if let Ok(dt) = s2.time.duration_since(s1.time) {
                total_elapsed += dt;
                let expected = self.config.sample_interval;
                let drift = (dt.as_secs_f64() - expected.as_secs_f64()) / expected.as_secs_f64();
                
                // Weight by inverse of combined errors and time difference
                let weight = 1.0 / ((s1.error + s2.error) as f64 * dt.as_secs_f64());
                drifts.push(drift);
                weights.push(weight);
            }
        }

        if drifts.is_empty() {
            return;
        }

        // Normalize weights
        let total_weight: f64 = weights.iter().sum();
        let weights: Vec<_> = weights.iter().map(|w| w / total_weight).collect();

        // Calculate weighted average drift
        let avg_drift = drifts.iter()
            .zip(weights.iter())
            .map(|(d, w)| d * w)
            .sum::<f64>();

        // Kalman filter parameters
        let q = 1e-12; // Process noise (very small for stable clock)
        let r = 1e-8;  // Measurement noise (based on typical system clock stability)

        // Kalman filter update
        let p_pred = self.drift_covariance + q;
        let k = p_pred / (p_pred + r);
        
        // Update state
        self.drift_estimate = self.drift_estimate + k * (avg_drift - self.drift_estimate);
        self.drift_covariance = (1.0 - k) * p_pred;

        // Convert drift to PPM for threshold check
        let drift_ppm = self.drift_estimate * 1_000_000.0;
        let max_drift_ppm = self.config.max_drift.as_secs_f64() * 1_000_000.0;

        // Switch to backup if drift exceeds threshold
        if drift_ppm.abs() > max_drift_ppm {
            if let Err(e) = self.switch_to_backup().await {
                eprintln!("Failed to switch to backup: {}", e);
            }
        }
    }

    /// Switches to a backup time source with initialization
    async fn switch_to_backup(&mut self) -> Result<()> {
        for backup in &self.config.backups {
            match backup {
                TimeSource::System => {
                    self.current_source = TimeSource::System;
                    self.drift_estimate = 0.0;
                    self.drift_covariance = 1e-6;
                    return Ok(());
                }
                TimeSource::Ntp(server) => {
                    // Initialize NTP if needed
                    if self.ntp_socket.is_none() {
                        let socket = UdpSocket::bind("0.0.0.0:0")?;
                        socket.set_nonblocking(true)?;
                        self.ntp_socket = Some(Arc::new(Mutex::new(socket)));
                    }
                    self.current_source = backup.clone();
                    return Ok(());
                }
                TimeSource::Gps { device, use_pps } => {
                    // Try to initialize GPS
                    if self.gps_receiver.is_none() {
                        let receiver = GpsReceiver::open(device, self.config.gps_config.baud_rate)?;
                        self.gps_receiver = Some(Arc::new(Mutex::new(receiver)));
                    }
                    self.current_source = backup.clone();
                    return Ok(());
                }
                TimeSource::Quantum => {
                    // Try to initialize quantum clock
                    if self.quantum_clock.is_none() {
                        let clock = crate::time::QuantumClock::new(self.config.quantum_config.coherence_time)?;
                        self.quantum_clock = Some(Arc::new(Mutex::new(clock)));
                    }
                    // Check if quantum state is coherent
                    if let Some(clock) = &self.quantum_clock {
                        let clock = clock.lock().await;
                        if clock.is_coherent() {
                            self.current_source = TimeSource::Quantum;
                            return Ok(());
                        }
                    }
                    continue;
                }
                TimeSource::Ptp(device) => {
                    // PTP not implemented yet
                    continue;
                }
            }
        }

        // If all backups failed, try to stay with current source but reset drift
        self.drift_estimate = 0.0;
        self.drift_covariance = 1e-6;
        Err(Error::timing("No available backup time sources"))
    }

    /// Returns the current drift estimate
    pub fn get_drift(&self) -> Duration {
        Duration::from_secs_f64(self.drift_estimate.abs())
    }

    /// Returns the current time with drift compensation
    pub fn now(&self) -> SystemTime {
        SystemTime::now() + Duration::from_secs_f64(self.drift_estimate)
    }

    /// Returns statistics about recent samples
    pub fn get_stats(&self) -> TimeStats {
        let mut stats = TimeStats {
            sample_count: self.samples.len(),
            min_error: u64::MAX,
            max_error: 0,
            mean_error: 0.0,
            rtt_stats: None,
        };

        if self.samples.is_empty() {
            return stats;
        }

        // Calculate error statistics
        let errors: Vec<_> = self.samples.iter().map(|s| s.error).collect();
        stats.min_error = *errors.iter().min().unwrap();
        stats.max_error = *errors.iter().max().unwrap();
        stats.mean_error = errors.iter().sum::<u64>() as f64 / errors.len() as f64;

        // Calculate RTT statistics if available
        let rtts: Vec<_> = self.samples.iter()
            .filter_map(|s| s.rtt.map(|r| r.as_nanos() as f64))
            .collect();

        if !rtts.is_empty() {
            let mean_rtt = rtts.iter().sum::<f64>() / rtts.len() as f64;
            let variance = rtts.iter()
                .map(|&r| (r - mean_rtt).powi(2))
                .sum::<f64>() / rtts.len() as f64;

            stats.rtt_stats = Some(RttStats {
                min: Duration::from_nanos(*rtts.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as u64),
                max: Duration::from_nanos(*rtts.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as u64),
                mean: Duration::from_nanos(mean_rtt as u64),
                stddev: Duration::from_nanos((variance.sqrt()) as u64),
            });
        }

        stats
    }
}
