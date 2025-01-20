use num_complex::Complex64;
use rustfft::FftPlanner;
use std::time::SystemTime;
use nalgebra::{Complex, DMatrix};

use crate::core::{Error, Result, FrequencyComponent, Precision};

/// Phase space representation of a signal
#[derive(Debug, Clone)]
pub struct PhaseSpace {
    /// Complex coordinates in phase space
    pub coordinates: Vec<Complex<f64>>,
    /// Time points
    pub timestamps: Vec<SystemTime>,
    /// Phase coherence metric
    pub coherence: f64,
}

/// Wavelet decomposition of a signal
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    /// Wavelet coefficients at different scales
    pub coefficients: Vec<Vec<Complex<f64>>>,
    /// Scale parameters
    pub scales: Vec<f64>,
    /// Center frequencies for each scale
    pub frequencies: Vec<f64>,
}

/// Frequency analyzer for decomposing and analyzing timing signals
pub struct FrequencyAnalyzer {
    /// FFT planner for efficient transforms
    fft_planner: FftPlanner<f64>,
    /// Sampling rate in Hz
    sample_rate: f64,
    /// Window size for FFT
    window_size: usize,
    /// Mother wavelet parameters
    wavelet_width: f64,
    /// Phase space dimension
    embedding_dim: usize,
    /// Phase space delay
    embedding_delay: usize,
}

impl FrequencyAnalyzer {
    /// Creates a new frequency analyzer
    pub fn new(sample_rate: f64, window_size: usize) -> Self {
        FrequencyAnalyzer {
            fft_planner: FftPlanner::new(),
            sample_rate,
            window_size,
            wavelet_width: 6.0,
            embedding_dim: 3,
            embedding_delay: 1,
        }
    }

    /// Performs wavelet transform on the input signal
    pub fn wavelet_transform(&self, samples: &[f64]) -> Result<WaveletDecomposition> {
        let n_samples = samples.len();
        let mut scales = Vec::new();
        let mut frequencies = Vec::new();
        let mut coefficients = Vec::new();

        // Generate logarithmically spaced scales
        let n_scales = 32;
        let min_scale = 2.0;
        let max_scale = n_samples as f64 / 6.0;
        let scale_step = (max_scale / min_scale).powf(1.0 / (n_scales as f64 - 1.0));

        let mut scale = min_scale;
        for _ in 0..n_scales {
            scales.push(scale);
            frequencies.push(self.sample_rate / (scale * 2.0 * std::f64::consts::PI));
            scale *= scale_step;
        }

        // Compute wavelet coefficients for each scale
        for scale in &scales {
            let mut scale_coeffs = Vec::with_capacity(n_samples);
            
            for t in 0..n_samples {
                let mut sum = Complex::new(0.0, 0.0);
                
                for i in 0..n_samples {
                    let t_scaled = (i as f64 - t as f64) / scale;
                    // Morlet wavelet
                    let wavelet = (-t_scaled * t_scaled / (2.0 * self.wavelet_width))
                        .exp() * (self.wavelet_width * t_scaled * Complex::i()).exp();
                    sum += Complex::new(samples[i], 0.0) * wavelet;
                }
                
                scale_coeffs.push(sum / scale.sqrt());
            }
            
            coefficients.push(scale_coeffs);
        }

        Ok(WaveletDecomposition {
            coefficients,
            scales,
            frequencies,
        })
    }

    /// Maps signal to phase space using delay embedding
    pub fn phase_space_mapping(&self, samples: &[f64], timestamps: &[SystemTime]) -> Result<PhaseSpace> {
        if samples.len() != timestamps.len() {
            return Err(Error::frequency_analysis(
                "Sample and timestamp counts must match"
            ));
        }

        let n_points = samples.len() - (self.embedding_dim - 1) * self.embedding_delay;
        let mut coordinates = Vec::with_capacity(n_points);
        let mut embedded_timestamps = Vec::with_capacity(n_points);

        // Construct delay vectors
        for i in 0..n_points {
            let mut coord = Complex::new(samples[i], 0.0);
            for d in 1..self.embedding_dim {
                let delay_idx = i + d * self.embedding_delay;
                coord += Complex::new(samples[delay_idx], 0.0) 
                    * Complex::i().powf(d as f64);
            }
            coordinates.push(coord);
            embedded_timestamps.push(timestamps[i]);
        }

        // Calculate phase coherence
        let coherence = self.calculate_phase_coherence(&coordinates);

        Ok(PhaseSpace {
            coordinates,
            timestamps: embedded_timestamps,
            coherence,
        })
    }

    /// Calculates phase coherence metric
    fn calculate_phase_coherence(&self, coordinates: &[Complex<f64>]) -> f64 {
        let n = coordinates.len();
        if n < 2 {
            return 0.0;
        }

        // Calculate phase differences
        let phase_diffs: Vec<f64> = coordinates.windows(2)
            .map(|w| (w[1] / w[0]).arg())
            .collect();

        // Calculate order parameter
        let sum = phase_diffs.iter()
            .map(|&phi| Complex::new(phi.cos(), phi.sin()))
            .sum::<Complex<f64>>();

        sum.norm() / n as f64
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

    /// Calculates precision based on frequency component stability and phase coherence
    pub fn calculate_precision(&self, components: &[FrequencyComponent], phase_space: &PhaseSpace) -> Precision {
        // Component-based precision
        let component_count = components.len() as f64;
        let amplitude_stability: f64 = components.iter()
            .map(|c| c.amplitude.norm())
            .sum();

        // Phase coherence contribution
        let coherence_factor = phase_space.coherence;

        // Combined precision metric
        let precision_value = (component_count * amplitude_stability * coherence_factor)
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
