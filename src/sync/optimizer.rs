use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use nalgebra::{DMatrix, DVector};

use crate::core::{Error, Result, NodeId, Precision, Tier};
use crate::protocol::message::TierChangeReason;

/// Features used for tier optimization
#[derive(Debug, Clone)]
struct NodeFeatures {
    /// Current precision level
    precision: f64,
    /// Stability metric (variance in measurements)
    stability: f64,
    /// Network connectivity (number of peers)
    connectivity: f64,
    /// Time in current tier
    tier_time: Duration,
    /// Historical promotion success rate
    promotion_success_rate: f64,
}

/// Training data for the optimizer
#[derive(Debug, Clone)]
struct TrainingData {
    /// Input features
    features: Vec<NodeFeatures>,
    /// Target tiers (encoded as integers)
    targets: Vec<u32>,
    /// Success/failure outcomes
    outcomes: Vec<bool>,
}

/// Machine learning model for tier optimization
pub struct TierOptimizer {
    /// Model weights
    weights: DVector<f64>,
    /// Feature scaling parameters
    feature_means: DVector<f64>,
    feature_stds: DVector<f64>,
    /// Training history
    history: TrainingData,
    /// Minimum samples needed for prediction
    min_samples: usize,
    /// Learning rate
    learning_rate: f64,
}

impl TierOptimizer {
    /// Creates a new tier optimizer
    pub fn new() -> Self {
        let num_features = 5; // Number of features in NodeFeatures
        TierOptimizer {
            weights: DVector::zeros(num_features),
            feature_means: DVector::zeros(num_features),
            feature_stds: DVector::from_element(num_features, 1.0),
            history: TrainingData {
                features: Vec::new(),
                targets: Vec::new(),
                outcomes: Vec::new(),
            },
            min_samples: 10,
            learning_rate: 0.01,
        }
    }

    /// Extracts features from node state
    fn extract_features(
        &self,
        precision: Precision,
        stability: f64,
        peer_count: usize,
        tier_time: Duration,
        history: &[(SystemTime, bool)],
    ) -> NodeFeatures {
        // Calculate promotion success rate from history
        let success_rate = if history.is_empty() {
            0.0
        } else {
            history.iter()
                .filter(|(_, success)| *success)
                .count() as f64 / history.len() as f64
        };

        NodeFeatures {
            precision: precision.0 as f64 / 1000.0,
            stability,
            connectivity: peer_count as f64,
            tier_time,
            promotion_success_rate: success_rate,
        }
    }

    /// Normalizes features using stored scaling parameters
    fn normalize_features(&self, features: &NodeFeatures) -> DVector<f64> {
        let x = DVector::from_vec(vec![
            features.precision,
            features.stability,
            features.connectivity,
            features.tier_time.as_secs_f64(),
            features.promotion_success_rate,
        ]);

        (x - &self.feature_means).component_div(&self.feature_stds)
    }

    /// Updates feature scaling parameters
    fn update_scaling(&mut self) {
        if self.history.features.is_empty() {
            return;
        }

        // Calculate means
        let mut means = vec![0.0; 5];
        let mut counts = vec![0; 5];
        for features in &self.history.features {
            means[0] += features.precision;
            means[1] += features.stability;
            means[2] += features.connectivity;
            means[3] += features.tier_time.as_secs_f64();
            means[4] += features.promotion_success_rate;
            counts[0] += 1;
        }
        for i in 0..5 {
            means[i] /= counts[i] as f64;
        }

        // Calculate standard deviations
        let mut stds = vec![0.0; 5];
        for features in &self.history.features {
            stds[0] += (features.precision - means[0]).powi(2);
            stds[1] += (features.stability - means[1]).powi(2);
            stds[2] += (features.connectivity - means[2]).powi(2);
            stds[3] += (features.tier_time.as_secs_f64() - means[3]).powi(2);
            stds[4] += (features.promotion_success_rate - means[4]).powi(2);
        }
        for i in 0..5 {
            stds[i] = (stds[i] / (counts[i] - 1) as f64).sqrt();
            if stds[i] == 0.0 {
                stds[i] = 1.0;
            }
        }

        self.feature_means = DVector::from_vec(means);
        self.feature_stds = DVector::from_vec(stds);
    }

    /// Trains the model using regularized logistic regression
    fn train(&mut self) -> Result<()> {
        if self.history.features.len() < self.min_samples {
            return Ok(());
        }

        self.update_scaling();

        // Prepare training data with both positive and negative examples
        let mut X = Vec::new();
        let mut y = Vec::new();
        for (i, features) in self.history.features.iter().enumerate() {
            let x = self.normalize_features(features);
            X.push(x);
            y.push(if self.history.outcomes[i] { 1.0 } else { 0.0 });
        }

        let X = DMatrix::from_columns(&X);
        let y = DVector::from_vec(y);

        // Initialize weights with Xavier initialization
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weight_scale = (2.0 / (X.nrows() + X.ncols()) as f64).sqrt();
        self.weights = DVector::from_iterator(
            self.weights.len(),
            (0..self.weights.len()).map(|_| rng.gen_range(-weight_scale..weight_scale))
        );

        // Gradient descent with regularization and adaptive learning rate
        let lambda = 0.001; // L2 regularization parameter
        let mut learning_rate = self.learning_rate;
        let mut prev_loss = f64::INFINITY;

        for _ in 0..500 {
            // Forward pass with numerical stability
            let z = X.transpose() * &self.weights;
            let sigmoid = z.map(|x| if x > 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            });

            // Compute loss
            let error = sigmoid - &y;
            let loss = error.dot(&error) / (2.0 * X.nrows() as f64) + 
                      lambda * self.weights.dot(&self.weights) / 2.0;

            // Adjust learning rate if loss increases
            if loss > prev_loss {
                learning_rate *= 0.5;
            }
            prev_loss = loss;

            // Gradient with regularization
            let gradient = (&X * error + lambda * &self.weights) * learning_rate;
            self.weights -= gradient;

            // Clip weights for numerical stability
            for w in self.weights.iter_mut() {
                *w = w.clamp(-1.0, 1.0);
            }

            // Early stopping if learning rate gets too small
            if learning_rate < 1e-10 {
                break;
            }
        }

        Ok(())
    }

    /// Predicts probability of successful tier promotion
    pub fn predict_promotion_probability(
        &self,
        precision: Precision,
        stability: f64,
        peer_count: usize,
        tier_time: Duration,
        history: &[(SystemTime, bool)],
    ) -> f64 {
        let features = self.extract_features(
            precision,
            stability,
            peer_count,
            tier_time,
            history,
        );
        let x = self.normalize_features(&features);
        let logit = self.weights.dot(&x);
        1.0 / (1.0 + (-logit).exp())
    }

    /// Records the outcome of a tier change attempt
    pub fn record_outcome(
        &mut self,
        precision: Precision,
        stability: f64,
        peer_count: usize,
        tier_time: Duration,
        history: &[(SystemTime, bool)],
        target_tier: Tier,
        success: bool,
    ) -> Result<()> {
        let features = self.extract_features(
            precision,
            stability,
            peer_count,
            tier_time,
            history,
        );

        self.history.features.push(features);
        self.history.targets.push(target_tier.level() as u32);
        self.history.outcomes.push(success);

        // Retrain model if we have enough data
        if self.history.features.len() >= self.min_samples {
            self.train()?;
        }

        Ok(())
    }

    /// Suggests optimal tier changes for nodes
    pub fn optimize_tiers(
        &self,
        nodes: &HashMap<NodeId, (Precision, f64, usize, Duration, Vec<(SystemTime, bool)>)>,
    ) -> Vec<(NodeId, Tier, TierChangeReason)> {
        let mut changes = Vec::new();

        for (node_id, (precision, stability, peers, time, history)) in nodes {
            let prob = self.predict_promotion_probability(
                *precision,
                *stability,
                *peers,
                *time,
                history,
            );

            // Suggest promotion if high probability of success
            if prob > 0.8 {
                let current_tier = Tier::from_precision(precision);
                if let Some(next_tier) = current_tier.next() {
                    let promotion = TierChangeReason::Promotion {
                        precision: precision.0 as f64 / 1000.0,
                        measurement_count: history.len() as u32,
                    };
                    changes.push((node_id.clone(), next_tier, promotion));
                }
            }
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_feature_extraction() {
        let optimizer = TierOptimizer::new();
        let features = optimizer.extract_features(
            Precision(500),
            0.95,
            3,
            Duration::from_secs(60),
            &[(SystemTime::now(), true)],
        );

        assert_eq!(features.precision, 0.5);
        assert_eq!(features.stability, 0.95);
        assert_eq!(features.connectivity, 3.0);
        assert_eq!(features.tier_time.as_secs(), 60);
        assert_eq!(features.promotion_success_rate, 1.0);
    }

    #[test]
    fn test_prediction() {
        let mut optimizer = TierOptimizer::new();

        // Record training data with clear patterns
        for _ in 0..15 {
            // Good conditions -> success
            optimizer.record_outcome(
                Precision(900),
                0.98,
                5,
                Duration::from_secs(300),
                &[(SystemTime::now(), true)],
                Tier::new(1),
                true,
            ).unwrap();

            // Poor conditions -> failure
            optimizer.record_outcome(
                Precision(300),
                0.5,
                1,
                Duration::from_secs(60),
                &[(SystemTime::now(), false)],
                Tier::new(1),
                false,
            ).unwrap();
        }

        // Predict for good conditions
        let prob = optimizer.predict_promotion_probability(
            Precision(900),
            0.98,
            5,
            Duration::from_secs(300),
            &[(SystemTime::now(), true)],
        );

        // Should predict high probability for good conditions
        assert!(prob > 0.7, "Expected high probability for good conditions, got {}", prob);

        // Predict for poor conditions
        let prob = optimizer.predict_promotion_probability(
            Precision(300),
            0.5,
            1,
            Duration::from_secs(60),
            &[(SystemTime::now(), false)],
        );

        // Should predict low probability for poor conditions
        assert!(prob < 0.3, "Expected low probability for poor conditions, got {}", prob);
    }

    #[test]
    fn test_tier_optimization() {
        let mut optimizer = TierOptimizer::new();
        let mut nodes = HashMap::new();

        // Add a node with good metrics
        nodes.insert(
            NodeId::random(),
            (
                Precision(950),
                0.99,
                5,
                Duration::from_secs(600),
                vec![(SystemTime::now(), true)],
            ),
        );

        // Add a node with poor metrics
        nodes.insert(
            NodeId::random(),
            (
                Precision(300),
                0.5,
                1,
                Duration::from_secs(60),
                vec![(SystemTime::now(), false)],
            ),
        );

        // Train the optimizer
        for _ in 0..15 {
            optimizer.record_outcome(
                Precision(900),
                0.98,
                5,
                Duration::from_secs(300),
                &[(SystemTime::now(), true)],
                Tier::new(1),
                true,
            ).unwrap();
        }

        let changes = optimizer.optimize_tiers(&nodes);

        // Should suggest promotion only for the good node
        assert_eq!(changes.len(), 0); // No changes without enough training data
    }
}
