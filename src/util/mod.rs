//! Utility module
//! 
//! This module provides common utilities and helper functions used
//! throughout the library.

use crate::core::{Error, Result};

/// Converts a duration to a floating-point number of seconds
pub fn duration_to_secs(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64()
}

/// Converts a floating-point number of seconds to a duration
pub fn secs_to_duration(secs: f64) -> std::time::Duration {
    std::time::Duration::from_secs_f64(secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_duration_conversion() {
        let duration = Duration::from_secs_f64(1.5);
        let secs = duration_to_secs(duration);
        assert_eq!(secs, 1.5);
        let duration2 = secs_to_duration(secs);
        assert_eq!(duration, duration2);
    }
}
