use std::io;
use thiserror::Error;

/// Custom error types for RFCH NTP
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Peer error: {0}")]
    Peer(String),

    #[error("Timing error: {0}")]
    Timing(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Frequency analysis error: {0}")]
    FrequencyAnalysis(String),

    #[error("Tier management error: {0}")]
    TierManagement(String),

    #[error("Synchronization error: {0}")]
    Sync(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Result type alias using our custom Error type
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Creates a new protocol error
    pub fn protocol(msg: impl Into<String>) -> Self {
        Error::Protocol(msg.into())
    }

    /// Creates a new network error
    pub fn network(msg: impl Into<String>) -> Self {
        Error::Network(msg.into())
    }

    /// Creates a new peer error
    pub fn peer(msg: impl Into<String>) -> Self {
        Error::Peer(msg.into())
    }

    /// Creates a new timing error
    pub fn timing(msg: impl Into<String>) -> Self {
        Error::Timing(msg.into())
    }

    /// Creates a new configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Creates a new frequency analysis error
    pub fn frequency_analysis(msg: impl Into<String>) -> Self {
        Error::FrequencyAnalysis(msg.into())
    }

    /// Creates a new tier management error
    pub fn tier_management(msg: impl Into<String>) -> Self {
        Error::TierManagement(msg.into())
    }

    /// Creates a new synchronization error
    pub fn sync(msg: impl Into<String>) -> Self {
        Error::Sync(msg.into())
    }

    /// Creates a new invalid state error
    pub fn invalid_state(msg: impl Into<String>) -> Self {
        Error::InvalidState(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::protocol("test error");
        assert!(matches!(err, Error::Protocol(_)));
        assert_eq!(err.to_string(), "Protocol error: test error");
    }

    #[test]
    fn test_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::Other, "test");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
