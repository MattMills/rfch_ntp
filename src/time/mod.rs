//! Time management module
//! 
//! This module handles time synchronization and drift compensation.

mod source;

pub use self::source::{TimeSource, TimeSourceConfig, TimeSourceManager};

/// Utility functions for time management
pub mod util {
    use std::time::{Duration, SystemTime};
    use crate::core::{Error, Result};

    /// Calculates the time difference between two timestamps
    pub fn time_diff(a: SystemTime, b: SystemTime) -> Result<Duration> {
        a.duration_since(b)
            .or_else(|e| b.duration_since(a))
            .map_err(|e| Error::timing(format!("Failed to calculate time difference: {}", e)))
    }

    /// Estimates clock drift rate in parts per million (ppm)
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
}

#[cfg(test)]
mod tests {
    use super::util;
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
        let samples = vec![
            (now - Duration::from_secs(10), Duration::from_micros(0)),
            (now - Duration::from_secs(5), Duration::from_micros(50)),
            (now, Duration::from_micros(100))
        ];

        let rate = util::estimate_drift_rate(&samples, Duration::from_secs(30)).unwrap();
        assert!(rate > 0.0 && rate < 100.0);
    }
}
