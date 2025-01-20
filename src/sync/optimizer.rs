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
            weights: DVector::zeros(num_features + 1), // +1 for bias term
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
        // Log transform duration to handle large values
        let duration_feature = (features.tier_time.as_secs_f64() + 1.0).ln();
        
        let x = DVector::from_vec(vec![
            features.precision,
            features.stability,
            features.connectivity / 10.0, // Scale down connectivity
            duration_feature / 10.0,      // Scale down duration
            features.promotion_success_rate,
        ]);

        // Min-max scaling for more stable gradients
        let scaled = (x - &self.feature_means).component_div(&self.feature_stds);
        scaled.map(|x| x.max(-3.0).min(3.0)) // Clip extreme values
    }

    /// Updates feature scaling parameters
    fn update_scaling(&mut self) {
        if self.history.features.is_empty() {
            return;
        }

        let n = self.history.features.len();
        let mut means = vec![0.0; 5];
        let mut stds = vec![0.0; 5];

        // First pass: calculate means
        for features in &self.history.features {
            let duration_feature = (features.tier_time.as_secs_f64() + 1.0).ln();
            means[0] += features.precision;
            means[1] += features.stability;
            means[2] += features.connectivity / 10.0;
            means[3] += duration_feature / 10.0;
            means[4] += features.promotion_success_rate;
        }
        for i in 0..5 {
            means[i] /= n as f64;
        }

        // Second pass: calculate standard deviations
        for features in &self.history.features {
            let duration_feature = (features.tier_time.as_secs_f64() + 1.0).ln();
            let x = vec![
                features.precision,
                features.stability,
                features.connectivity / 10.0,
                duration_feature / 10.0,
                features.promotion_success_rate,
            ];
            for i in 0..5 {
                stds[i] += (x[i] - means[i]).powi(2);
            }
        }

        // Finalize standard deviations
        for i in 0..5 {
            stds[i] = (stds[i] / (n - 1) as f64).sqrt().max(1e-6);
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

        // Convert feature vectors to matrix
        let num_samples = X.len();
        let num_features = self.weights.len() - 1; // Subtract 1 for bias term
        let mut features_matrix = DMatrix::zeros(num_features, num_samples);
        for (i, x) in X.iter().enumerate() {
            features_matrix.set_column(i, x);
        }
        let y = DVector::from_vec(y);

        // Add bias term to features
        let mut features_with_bias = DMatrix::zeros(num_features + 1, num_samples);
        features_with_bias.view_mut((0, 0), (num_features, num_samples))
            .copy_from(&features_matrix);
        features_with_bias.row_mut(num_features).fill(1.0);

        // Initialize weights with small random values
        use rand::Rng;
        let mut rng = rand::thread_rng();
        self.weights = DVector::from_iterator(
            num_features + 1,
            (0..num_features + 1).map(|_| rng.gen_range(-0.01..0.01))
        );

        // Gradient descent with adaptive learning rate
        let lambda = 0.0001; // Smaller L2 regularization
        let mut learning_rate = 0.1;
        let mut best_weights = self.weights.clone();
        let mut best_loss = f64::INFINITY;
        let mut patience = 0;
        let max_patience = 50;

        for _ in 0..20000 {
            // Forward pass
            let z = features_with_bias.transpose() * &self.weights;
            let predictions = z.map(|x| 1.0 / (1.0 + (-x).exp()));

            // Compute loss
            let epsilon = 1e-10;
            let predictions = predictions.map(|x| x.max(epsilon).min(1.0 - epsilon));
            let mut total_loss = 0.0;
            for i in 0..y.len() {
                let y_i = y[i];
                let p_i = predictions[i];
                total_loss -= y_i * p_i.ln() + (1.0 - y_i) * (1.0 - p_i).ln();
            }
            let loss = total_loss / y.len() as f64 + lambda * self.weights.dot(&self.weights) / 2.0;

            // Track best weights and handle early stopping
            if loss < best_loss {
                best_loss = loss;
                best_weights = self.weights.clone();
                patience = 0;
            } else {
                patience += 1;
                if patience > max_patience {
                    learning_rate *= 0.5;
                    patience = 0;
                    if learning_rate < 1e-6 {
                        break;
                    }
                }
            }

            // Gradient update
            let mut error = DVector::zeros(predictions.len());
            for i in 0..predictions.len() {
                error[i] = predictions[i] - y[i];
            }
            let gradient = features_with_bias.clone() * &error / num_samples as f64;
            self.weights -= learning_rate * (gradient + lambda * &self.weights);
        }

        // Use best weights found during training
        self.weights = best_weights;

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
        
        // Ensure weights are initialized
        if self.weights.iter().all(|w| *w == 0.0) {
            return 0.5; // Return neutral probability if not trained
        }

        // Add bias term to feature vector
        let mut x_with_bias = DVector::zeros(x.len() + 1);
        x_with_bias.view_mut((0, 0), (x.len(), 1)).copy_from(&x);
        x_with_bias[x.len()] = 1.0;

        // Calculate logistic with numerical stability
        let logit = self.weights.dot(&x_with_bias).max(-10.0).min(10.0);
        let prob = 1.0 / (1.0 + (-logit).exp());
        
        prob.max(0.0).min(1.0)
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
    fn test_untrained_prediction() {
        let optimizer = TierOptimizer::new();
        let prob = optimizer.predict_promotion_probability(
            Precision(500),
            0.5,
            3,
            Duration::from_secs(60),
            &[(SystemTime::now(), true)],
        );
        assert!((prob - 0.5).abs() < 1e-6, "Untrained model should return 0.5");
    }

    #[test]
    fn test_minimal_training() {
        let mut optimizer = TierOptimizer::new();
        
        // Add just enough samples to meet min_samples
        for _ in 0..5 {
            optimizer.record_outcome(
                Precision(900),
                0.9,
                5,
                Duration::from_secs(300),
                &[(SystemTime::now(), true)],
                Tier::new(1),
                true,
            ).unwrap();
        }

        // Should not train with insufficient samples
        optimizer.train().unwrap();
        
        let prob = optimizer.predict_promotion_probability(
            Precision(900),
            0.9,
            5,
            Duration::from_secs(300),
            &[(SystemTime::now(), true)],
        );
        assert!((prob - 0.5).abs() < 1e-6, "Insufficient samples should return 0.5");
    }

    #[test]
    fn test_training_good_conditions() {
        let mut optimizer = TierOptimizer::new();
        
        // Train only on good conditions
        for _ in 0..15 {
            optimizer.record_outcome(
                Precision(900),
                0.95,
                5,
                Duration::from_secs(300),
                &[(SystemTime::now(), true)],
                Tier::new(1),
                true,
            ).unwrap();
        }

        optimizer.train().unwrap();
        
        // Should predict high probability for similar good conditions
        let prob = optimizer.predict_promotion_probability(
            Precision(900),
            0.95,
            5,
            Duration::from_secs(300),
            &[(SystemTime::now(), true)],
        );
        assert!(prob > 0.7, "Expected high probability for good conditions, got {}", prob);
    }

    #[test]
    fn test_training_poor_conditions() {
        let mut optimizer = TierOptimizer::new();
        
        // Train only on poor conditions
        for _ in 0..15 {
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

        optimizer.train().unwrap();
        
        // Should predict low probability for similar poor conditions
        let prob = optimizer.predict_promotion_probability(
            Precision(300),
            0.5,
            1,
            Duration::from_secs(60),
            &[(SystemTime::now(), false)],
        );
        assert!(prob < 0.3, "Expected low probability for poor conditions, got {}", prob);
    }

    #[test]
    fn test_balanced_training() {
        let mut optimizer = TierOptimizer::new();
        
        // Add balanced training data
        for _ in 0..10 {
            // Good conditions
            optimizer.record_outcome(
                Precision(900),
                0.95,
                5,
                Duration::from_secs(300),
                &[(SystemTime::now(), true)],
                Tier::new(1),
                true,
            ).unwrap();

            // Poor conditions
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

        optimizer.train().unwrap();

        // Test predictions
        let good_prob = optimizer.predict_promotion_probability(
            Precision(900),
            0.95,
            5,
            Duration::from_secs(300),
            &[(SystemTime::now(), true)],
        );
        let poor_prob = optimizer.predict_promotion_probability(
            Precision(300),
            0.5,
            1,
            Duration::from_secs(60),
            &[(SystemTime::now(), false)],
        );

        assert!(good_prob > poor_prob, 
            "Expected higher probability for good conditions ({}) than poor conditions ({})",
            good_prob, poor_prob);
    }

    #[test]
    fn test_tier_optimization_untrained() {
        let optimizer = TierOptimizer::new();
        let mut nodes = HashMap::new();

        // Add nodes with various metrics
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

        let changes = optimizer.optimize_tiers(&nodes);
        assert_eq!(changes.len(), 0, "Untrained model should not suggest changes");
    }

    #[test]
    fn test_tier_optimization_trained() {
        let mut optimizer = TierOptimizer::new();
        
        // Train with balanced data
        for _ in 0..10 {
            // Good conditions -> success
            optimizer.record_outcome(
                Precision(900),
                0.95,
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

        let mut nodes = HashMap::new();

        // Add a node with excellent metrics
        let good_node = NodeId::random();
        nodes.insert(
            good_node.clone(),
            (
                Precision(950),
                0.99,
                5,
                Duration::from_secs(600),
                vec![(SystemTime::now(), true)],
            ),
        );

        // Add a node with poor metrics
        let poor_node = NodeId::random();
        nodes.insert(
            poor_node.clone(),
            (
                Precision(300),
                0.5,
                1,
                Duration::from_secs(60),
                vec![(SystemTime::now(), false)],
            ),
        );

        let changes = optimizer.optimize_tiers(&nodes);

        // Should suggest promotion only for the good node
        assert_eq!(changes.len(), 1, "Expected exactly one promotion");
        if let Some((node_id, _, _)) = changes.first() {
            assert_eq!(node_id, &good_node, "Wrong node selected for promotion");
        }
    }
}
