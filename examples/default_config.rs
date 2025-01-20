use rfch_ntp::time::{TimeSourceConfig, TimeSourceManager, TimeSource};
use rfch_ntp::sync::FrequencyAnalyzer;
use std::time::{Duration, SystemTime};
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    // Create time source configuration
    let sample_interval = Duration::from_millis(100); // 10Hz sampling
    let window_size = 32; // Match analyzer window

    let mut time_config = TimeSourceConfig::default();
    time_config.primary = TimeSource::System;
    time_config.backups = vec![]; // No backups for this example
    time_config.sample_interval = sample_interval;
    time_config.window_size = window_size;
    time_config.max_drift = Duration::from_micros(1000); // Allow more drift for testing

    // Create frequency analyzer with matching parameters
    let sample_rate = 10.0; // 10Hz to match time source
    let mut analyzer = FrequencyAnalyzer::new(sample_rate, window_size);
    
    println!("Starting time source manager and frequency analyzer:");
    println!("Time configuration:");
    println!("- Primary source: System clock");
    println!("- Sample interval: {:?} ({}Hz)", sample_interval, 
             1000 / sample_interval.as_millis());
    println!("- Window size: {} samples", window_size);
    println!("\nAnalyzer configuration:");
    println!("- Sample rate: {}Hz", sample_rate);
    println!("- FFT window: {} samples", window_size);

    // Create time manager
    let mut manager = TimeSourceManager::new(time_config).unwrap();
    println!("\nRunning for 10 seconds...");
    println!("Simulating drift with 1Hz and 0.1Hz components");
    
    // Monitor for 10 seconds
    let start = std::time::Instant::now();
    let run_duration = Duration::from_secs(10);
    let mut samples = Vec::new();
    let mut timestamps = Vec::new();
    
    while start.elapsed() < run_duration {
        // Sample time with artificial drift
        if let Err(e) = manager.sample_time().await {
            eprintln!("Sampling error: {}", e);
            continue;
        }

        // Get current state with simulated periodic drift
        let now = SystemTime::now();
        let base_drift = manager.get_drift().as_secs_f64();
        // Add 1Hz and 0.1Hz components with larger amplitudes
        let t = start.elapsed().as_secs_f64();
        let simulated_drift = base_drift + 
            0.005 * (2.0 * std::f64::consts::PI * 1.0 * t).sin() +  // 1 Hz, 5ms amplitude
            0.002 * (2.0 * std::f64::consts::PI * 0.1 * t).sin();   // 0.1 Hz, 2ms amplitude
        samples.push(simulated_drift);
        timestamps.push(now);

        // Display stats
        let stats = manager.get_stats();
        println!("\nTime stats at t={:.1}s:", t);
        println!("- Samples: {}", stats.sample_count);
        println!("- Mean error: {:.2} µs", stats.mean_error / 1000.0);
        println!("- Drift: {:.3} ppm", simulated_drift * 1_000_000.0);

        // Run frequency analysis when we have enough samples
        if samples.len() >= window_size {
            let window_samples = &samples[samples.len()-window_size..];
            let window_timestamps = &timestamps[timestamps.len()-window_size..];

            // Frequency decomposition
            match analyzer.decompose(window_samples) {
                Ok(components) => {
                    if !components.is_empty() {
                        println!("\nFrequency components:");
                        for comp in &components {
                            // Only show significant components
                            if comp.amplitude.norm() > 0.0001 {
                                println!("  {:.2} Hz: amplitude={:.6}, phase={:.2} rad",
                                    comp.frequency,
                                    comp.amplitude.norm(),
                                    comp.phase);
                            }
                        }
                    }
                }
                Err(e) => eprintln!("Frequency analysis error: {}", e),
            }

            // Phase space analysis
            match analyzer.phase_space_mapping(window_samples, window_timestamps) {
                Ok(phase_space) => {
                    println!("\nPhase space analysis:");
                    println!("  Coherence: {:.3}", phase_space.coherence);
                    println!("  Points: {}", phase_space.coordinates.len());
                }
                Err(e) => eprintln!("Phase space analysis error: {}", e),
            }
        }

        sleep(sample_interval).await;
    }

    println!("\nDone");
}
