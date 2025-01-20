//! Synchronization module
//! 
//! This module handles the synchronization logic, including frequency analysis,
//! phase alignment, and precision management.

pub mod frequency;
pub mod sync_manager;

pub use self::frequency::FrequencyAnalyzer;
pub use self::sync_manager::{SyncManager, SyncConfig};

use std::time::{Duration, SystemTime};
use crate::core::{Error, Result, FrequencyComponent, Precision, Tier};

/// Utility functions for synchronization
pub mod util {
    use super::*;

    /// Calculates the phase difference between two frequency components
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
    pub fn combine_components(components: &[FrequencyComponent]) -> Vec<FrequencyComponent> {
        use std::collections::HashMap;
        
        // Group components by frequency (within small tolerance)
        let mut groups: HashMap<i64, Vec<&FrequencyComponent>> = HashMap::new();
        const FREQ_TOLERANCE: f64 = 0.1; // Hz

        for component in components {
            let freq_key = (component.frequency / FREQ_TOLERANCE).round() as i64;
            groups.entry(freq_key)
                .or_default()
                .push(component);
        }

        // Combine each group
        let mut result = Vec::new();
        for components in groups.values() {
            if components.is_empty() {
                continue;
            }

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

        result
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
                frequency: 1.1, // Close enough to combine with 1.0
                phase: std::f64::consts::PI / 4.0,
                amplitude: Complex64::new(1.0, 0.0),
                timestamp: now,
            },
        ];

        let combined = util::combine_components(&components);
        assert_eq!(combined.len(), 1); // Should combine into single component
        assert!((combined[0].frequency - 1.05).abs() < 0.1);
    }
}
