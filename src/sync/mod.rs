//! Synchronization and optimization module
//! 
//! This module handles synchronization and optimization logic, including:
//! 
//! - Frequency analysis and decomposition
//! - Phase alignment and drift compensation
//! - Precision management and tier optimization
//! - Machine learning for network topology
//! 
//! The module implements a hierarchical approach to synchronization:
//! 
//! 1. Frequency component analysis using FFT and wavelets
//! 2. Statistical filtering and phase alignment
//! 3. Machine learning for tier optimization
//! 4. Dynamic network topology management

pub mod frequency;
pub mod sync_manager;
pub mod optimizer;

pub use self::frequency::FrequencyAnalyzer;
pub use self::sync_manager::{SyncManager, SyncConfig};
pub use self::optimizer::TierOptimizer;

use std::time::{Duration, SystemTime};
use crate::core::{Error, Result, FrequencyComponent, Precision, Tier};

/// Utility functions for synchronization
pub mod util {
    use super::*;

    /// Calculates the phase difference between two frequency components
    /// 
    /// Returns the normalized phase difference in the range [-π, π]
    pub fn phase_difference(a: &FrequencyComponent, b: &FrequencyComponent) -> f64 {
        let mut diff = a.phase - b.phase;
        // Normalize to [-π, π]
        while diff > std::f64::consts::PI {
            diff -= 2.0 * std::f64::consts::PI;
        }
        while diff < -std::f64::consts::PI {
            diff += 2.0 * std::f64::consts::PI;
        }
        diff
    }

    /// Estimates frequency drift between components
    /// 
    /// Calculates drift rate by combining:
    /// - Frequency difference
    /// - Phase evolution
    /// - Time difference
    pub fn estimate_drift(
        old: &FrequencyComponent,
        new: &FrequencyComponent
    ) -> Result<f64> {
        let time_diff = new.timestamp
            .duration_since(old.timestamp)
            .map_err(|e| Error::timing(format!("Invalid timestamp order: {}", e)))?;

        let phase_diff = phase_difference(new, old);
        let freq_diff = new.frequency - old.frequency;

        // Drift in Hz/s
        Ok(freq_diff / time_diff.as_secs_f64() + 
           phase_diff / (2.0 * std::f64::consts::PI * time_diff.as_secs_f64()))
    }

    /// Combines multiple frequency components with weighted averaging
    /// 
    /// Uses amplitude-weighted averaging to combine components with similar frequencies.
    /// Components are grouped by frequency within a small tolerance band.
    pub fn combine_components(components: &[FrequencyComponent]) -> Vec<FrequencyComponent> {
        use std::collections::HashMap;
        
        if components.is_empty() {
            return Vec::new();
        }

        // Sort components by frequency
        let mut sorted_components = components.to_vec();
        sorted_components.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());

        // Group components by frequency (within small tolerance)
        let mut groups = Vec::new();
        let mut current_group = vec![sorted_components[0].clone()];
        const FREQ_TOLERANCE: f64 = 0.1; // Hz

        for component in sorted_components.iter().skip(1) {
            let last_freq = current_group.last().unwrap().frequency;
            if (component.frequency - last_freq).abs() <= FREQ_TOLERANCE {
                current_group.push(component.clone());
            } else {
                if !current_group.is_empty() {
                    groups.push(std::mem::take(&mut current_group));
                }
                current_group.push(component.clone());
            }
        }
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        // Combine each group
        let mut result = Vec::new();
        for components in groups {
            // Use amplitude as weight
            let total_weight: f64 = components.iter()
                .map(|c| c.amplitude.norm())
                .sum();

            let weighted_freq = components.iter()
                .map(|c| c.frequency * c.amplitude.norm())
                .sum::<f64>() / total_weight;

            let weighted_phase = components.iter()
                .map(|c| c.phase * c.amplitude.norm())
                .sum::<f64>() / total_weight;

            let combined_amplitude = components.iter()
                .map(|c| c.amplitude)
                .sum();

            result.push(FrequencyComponent {
                frequency: weighted_freq,
                phase: weighted_phase,
                amplitude: combined_amplitude,
                timestamp: SystemTime::now(),
            });
        }

        result.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());
        result
    }

    /// Calculates stability metric for frequency components
    /// 
    /// Returns a value between 0 and 1 indicating the stability of the components:
    /// - 1.0 indicates perfect stability
    /// - 0.0 indicates complete instability
    pub fn calculate_stability(
        components: &[FrequencyComponent],
        window: Duration
    ) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Calculate variance in frequency and phase
        let now = SystemTime::now();
        let recent: Vec<_> = components.iter()
            .filter(|c| {
                now.duration_since(c.timestamp)
                    .map(|d| d < window)
                    .unwrap_or(false)
            })
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Calculate frequency stability using RMS
        let freq_mean = recent.iter()
            .map(|c| c.frequency)
            .sum::<f64>() / recent.len() as f64;

        let freq_rms = (recent.iter()
            .map(|c| ((c.frequency - freq_mean) / freq_mean).powi(2))
            .sum::<f64>() / recent.len() as f64)
            .sqrt();

        // Calculate phase stability using RMS
        let phase_diffs: Vec<_> = recent.windows(2)
            .map(|w| phase_difference(w[0], w[1]).abs())
            .collect();

        let phase_rms = if !phase_diffs.is_empty() {
            (phase_diffs.iter()
                .map(|&d| d.powi(2))
                .sum::<f64>() / phase_diffs.len() as f64)
                .sqrt()
        } else {
            std::f64::consts::PI
        };

        // Convert RMS values to stability metrics with adjusted thresholds
        let freq_stability = 1.0 / (1.0 + freq_rms.powi(2));  // More lenient frequency threshold
        let phase_stability = 1.0 / (1.0 + (phase_rms / (std::f64::consts::PI / 4.0)).powi(2));  // More lenient phase threshold

        // Combine metrics with weighted importance
        let freq_weight = 0.7;
        let phase_weight = 0.3;
        
        let stability = freq_weight * freq_stability + phase_weight * phase_stability;
        stability.min(1.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_phase_difference() {
        let now = SystemTime::now();
        let a = FrequencyComponent {
            frequency: 1.0,
            phase: 0.0,
            amplitude: Complex64::new(1.0, 0.0),
            timestamp: now,
        };
        let b = FrequencyComponent {
            frequency: 1.0,
            phase: std::f64::consts::PI / 2.0,
            amplitude: Complex64::new(1.0, 0.0),
            timestamp: now,
        };

        let diff = util::phase_difference(&a, &b);
        assert!((diff + std::f64::consts::PI / 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_components() {
        let now = SystemTime::now();
        let components = vec![
            FrequencyComponent {
                frequency: 1.0,
                phase: 0.0,
                amplitude: Complex64::new(1.0, 0.0),
                timestamp: now,
            },
            FrequencyComponent {
                frequency: 2.0, // Different frequency band
                phase: std::f64::consts::PI / 4.0,
                amplitude: Complex64::new(1.0, 0.0),
                timestamp: now,
            },
        ];

        let combined = util::combine_components(&components);
        assert_eq!(combined.len(), 2); // Should remain separate components
        assert!((combined[0].frequency - 1.0).abs() < 0.1);
        assert!((combined[1].frequency - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_stability_calculation() {
        let now = SystemTime::now();
        
        // Create stable components with very small variations
        let mut stable_components = Vec::new();
        for i in 0..5 {
            stable_components.push(FrequencyComponent {
                frequency: 1.0 + (i as f64 * 0.0001), // Tiny frequency drift
                phase: 0.01 * i as f64, // Small phase changes
                amplitude: Complex64::new(1.0, 0.0),
                timestamp: now + Duration::from_secs(i),
            });
        }

        // Create unstable components with large variations
        let mut unstable_components = Vec::new();
        for i in 0..5 {
            unstable_components.push(FrequencyComponent {
                frequency: 1.0 + (i as f64 * 0.5), // Large frequency changes
                phase: std::f64::consts::PI * i as f64 / 2.0, // Large phase changes
                amplitude: Complex64::new(1.0, 0.0),
                timestamp: now + Duration::from_secs(i),
            });
        }

        let stable = util::calculate_stability(&stable_components, Duration::from_secs(10));
        let unstable = util::calculate_stability(&unstable_components, Duration::from_secs(10));

        assert!(stable > 0.8, "Expected high stability (>0.8) for stable components, got {}", stable);
        assert!(unstable < 0.3, "Expected low stability (<0.3) for unstable components, got {}", unstable);
    }
}
