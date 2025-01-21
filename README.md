# RFCH NTP: Recursive Frequency Composition Heartbeat Network Timing Protocol

A novel distributed timing protocol that establishes precise timing through emergent behavior in peer networks rather than relying on absolute time sources. RFCH NTP uses advanced signal processing and quantum-inspired mathematical techniques to achieve robust synchronization.

## Features

- **Distributed Time Synchronization**: Peer-to-peer network timing without central authority
- **Frequency Analysis**: FFT-based decomposition of timing signals
- **Phase Space Mapping**: Advanced geometric analysis of timing coherence
- **Multiple Time Sources**: Support for system clock, PTP, and GPS time sources
- **Drift Compensation**: Automatic detection and correction of timing drift
- **Robust Network Protocol**: Tokio-based async networking with custom codec

## Architecture

The library is organized into several key modules:

- **core**: Core types, error handling, and serialization
- **time**: Time source management and synchronization
  - Support for system clock, PTP, and GPS
  - Quantum-inspired timing analysis
  - Drift detection and compensation
- **sync**: Frequency analysis and synchronization
  - FFT-based signal decomposition
  - Phase space mapping
  - Frequency optimization
- **protocol**: Network protocol implementation
  - Custom message codec
  - State management
  - Discovery protocol
- **network**: Network communication layer
  - Connection management
  - Peer discovery
  - Async communication

## Mathematical Foundation

RFCH NTP implements advanced mathematical concepts:

- Phase space mapping using complex coordinates
- Fourier and wavelet transforms for frequency isolation
- Recursive recomposition of timing signals
- Topological protection of timing stability
- Golden ratio-based coupling mechanisms
- Cross-band coherence metrics

## Usage

Basic example of time source management and frequency analysis:

```rust
use rfch_ntp::time::{TimeSourceConfig, TimeSourceManager, TimeSource};
use rfch_ntp::sync::FrequencyAnalyzer;
use std::time::Duration;

#[tokio::main]
async fn main() {
    // Configure time source
    let mut config = TimeSourceConfig::default();
    config.primary = TimeSource::System;
    config.sample_interval = Duration::from_millis(100);
    config.window_size = 32;

    // Create analyzer
    let analyzer = FrequencyAnalyzer::new(10.0, 32);
    
    // Create manager
    let mut manager = TimeSourceManager::new(config).unwrap();

    // Sample and analyze
    manager.sample_time().await?;
    let stats = manager.get_stats();
    println!("Mean error: {:.2} µs", stats.mean_error / 1000.0);
}
```

## Installation

Add to your Cargo.toml:

```toml
[dependencies]
rfch_ntp = "0.1.0"
```

## Requirements

- Rust 2021 edition or later
- Tokio runtime for async operations
- System with precise timing capabilities (for optimal performance)

## Building and Testing

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run example
cargo run --example default_config
```

## Implementation Details

The protocol operates through several key mechanisms:

1. **Signal Decomposition**
   - Decomposes timing signals into phase space
   - Isolates frequency components using FFT
   - Analyzes energy distributions

2. **Peer Synchronization**
   - Each peer broadcasts selected frequency components
   - Includes timestamps and drift metrics
   - Maintains coherence through feedback loops

3. **Optimization**
   - Adaptive buffers track real-time coherence
   - Geometric validation ensures phase consistency
   - Topological protection against perturbations

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License.
