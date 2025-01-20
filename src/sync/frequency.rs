use num_complex::Complex64;
use rustfft::FftPlanner;
use std::time::SystemTime;

use crate::core::{Error, Result, FrequencyComponent, Precision};

/// Frequency analyzer for decomposing and analyzing timing signals
pub struct FrequencyAnalyzer {
    /// FFT planner for efficient transforms
    fft_planner: FftPlanner<f64>,
    /// Sampling rate in Hz
    sample_rate: f64,
    /// Window size for FFT
    window_size: usize,
}

impl FrequencyAnalyzer {
    /// Creates a new frequency analyzer
    pub fn new(sample_rate: f64, window_size: usize) -> Self {
        FrequencyAnalyzer {
            fft_planner: FftPlanner::new(),
            sample_rate,
            window_size,
        }
    }

    /// Decomposes a time series into frequency components
    pub fn decompose(&mut self, samples: &[f64]) -> Result<Vec<FrequencyComponent>> {
        if samples.len() != self.window_size {
            return Err(Error::frequency_analysis(format!(
                "Expected {} samples, got {}",
                self.window_size,
                samples.len()
            )));
        }

        // Apply Hann window to reduce spectral leakage
        let windowed: Vec<Complex64> = samples
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 
                    / (self.window_size as f64 - 1.0)).cos());
                Complex64::new(x * window, 0.0)
            })
            .collect();

        // Perform FFT
        let mut buffer = windowed;
        let fft = self.fft_planner.plan_fft_forward(self.window_size);
        fft.process(&mut buffer);

        // Extract significant frequency components
        let components = self.extract_components(&buffer)?;

        Ok(components)
    }

    /// Extracts significant frequency components from FFT results
    fn extract_components(&self, buffer: &[Complex64]) -> Result<Vec<FrequencyComponent>> {
        let freq_resolution = self.sample_rate / self.window_size as f64;
        let nyquist = self.sample_rate / 2.0;
        let now = SystemTime::now();

        // Find peaks in frequency spectrum
        let mut components = Vec::new();
        for i in 1..buffer.len()/2 {
            let freq = i as f64 * freq_resolution;
            if freq >= nyquist {
                break;
            }

            let amplitude = buffer[i];
            let prev = buffer[i-1].norm();
            let current = amplitude.norm();
            let next = buffer[i+1].norm();

            // Peak detection
            if current > prev && current > next && current > 0.1 {
                components.push(FrequencyComponent {
                    amplitude,
                    frequency: freq,
                    phase: amplitude.arg(),
                    timestamp: now,
                });
            }
        }

        Ok(components)
    }

    /// Calculates precision based on frequency component stability
    pub fn calculate_precision(&self, components: &[FrequencyComponent]) -> Precision {
        // Simple precision metric based on component count and amplitude stability
        let component_count = components.len() as u32;
        let amplitude_sum: f64 = components.iter()
            .map(|c| c.amplitude.norm())
            .sum();

        // Scale precision based on component count and amplitude strength
        let precision_value = (component_count as f64 * amplitude_sum)
            .min(1000.0) as u32;

        Precision(precision_value)
    }

    /// Synthesizes a time series from frequency components
    pub fn synthesize(&self, components: &[FrequencyComponent], num_samples: usize) -> Vec<f64> {
        let mut samples = vec![0.0; num_samples];
        let sample_period = 1.0 / self.sample_rate;

        for t in 0..num_samples {
            let time = t as f64 * sample_period;
            for component in components {
                let angle = 2.0 * std::f64::consts::PI * component.frequency * time 
                    + component.phase;
                samples[t] += component.amplitude.norm() * angle.cos();
            }
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_frequency_decomposition() {
        let sample_rate = 1000.0; // 1 kHz
        let window_size = 1024;
        let mut analyzer = FrequencyAnalyzer::new(sample_rate, window_size);

        // Generate test signal: 10 Hz sine wave
        let freq = 10.0;
        let samples: Vec<f64> = (0..window_size)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect();

        let components = analyzer.decompose(&samples).unwrap();
        
        // Should find the 10 Hz component
        assert!(!components.is_empty());
        let found_freq = components.iter()
            .find(|c| (c.frequency - freq).abs() < 1.0)
            .is_some();
        assert!(found_freq, "Failed to detect 10 Hz component");
    }

    #[test]
    fn test_synthesis() {
        let sample_rate = 1000.0;
        let window_size = 1024;
        let analyzer = FrequencyAnalyzer::new(sample_rate, window_size);

        let components = vec![
            FrequencyComponent {
                amplitude: Complex64::new(1.0, 0.0),
                frequency: 10.0,
                phase: 0.0,
                timestamp: SystemTime::now(),
            }
        ];

        let samples = analyzer.synthesize(&components, window_size);
        assert_eq!(samples.len(), window_size);
    }
}
