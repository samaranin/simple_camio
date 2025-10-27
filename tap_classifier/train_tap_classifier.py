"""
Training and evaluation script for TapClassifier.

This script provides utilities to:
- Generate synthetic training data based on tap detection parameters
- Train the classifier offline
- Evaluate classifier performance
- Visualize feature importance
- Export trained models

USAGE:
    # Generate synthetic training data and train
    python train_tap_classifier.py --train --samples 1000

    # Evaluate trained model
    python train_tap_classifier.py --evaluate

    # Show feature importance
    python train_tap_classifier.py --feature-importance

    # Train with custom parameters
    python train_tap_classifier.py --train --samples 2000 --learning-rate 0.02 --epochs 5
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from tap_classifier.tap_classifier import TapClassifier
from config import TapDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_tap_data(num_samples=500):
    """
    Generate synthetic training data for tap detection.

    Creates realistic feature vectors based on typical tap characteristics
    defined in TapDetectionConfig.

    Args:
        num_samples (int): Number of positive tap examples to generate

    Returns:
        tuple: (features_list, labels_list)
    """
    cfg = TapDetectionConfig
    features_list = []
    labels_list = []

    logger.info(f"Generating {num_samples} synthetic tap examples...")

    # Generate positive examples (actual taps)
    for i in range(num_samples):
        # Depth features (realistic tap depths)
        zrel_depth = np.random.uniform(cfg.ZREL_MIN_PRESS_DEPTH, 0.06)
        plane_depth = np.random.uniform(cfg.PLANE_MIN_PRESS_DEPTH, 0.05)
        ang_depth = np.random.uniform(cfg.ANG_MIN_PRESS_DEPTH, 40.0)
        z_depth = np.random.uniform(cfg.TAP_MIN_PRESS_DEPTH, 0.05)

        # Spatial features (low drift for good taps)
        drift = np.random.uniform(0, cfg.TAP_MAX_XY_DRIFT * 0.5)

        # Velocity features (inward motion)
        vzrel = np.random.uniform(cfg.TAP_MIN_VEL, 0.5)
        vplane = np.random.uniform(cfg.TAP_MIN_VEL, 0.5)
        vang = np.random.uniform(cfg.ANG_MIN_VEL, 300.0)
        vz = np.random.uniform(cfg.TAP_MIN_VEL, 0.5)
        ray_vel = np.random.uniform(cfg.RAY_MIN_IN_VEL, 0.3)

        # Temporal features (typical tap duration)
        duration = np.random.uniform(cfg.TAP_MIN_DURATION, cfg.TAP_MAX_DURATION * 0.7)

        # Hand size features
        scale_factor = np.random.uniform(cfg.MIN_SCALE_FACTOR, cfg.MAX_SCALE_FACTOR)
        palm_width = np.random.uniform(cfg.SMALL_HAND_THRESHOLD, cfg.REFERENCE_PALM_WIDTH)

        # Trigger count (taps typically trigger multiple detectors)
        trigger_count = np.random.randint(2, 5)

        # Engineered features
        depth_ratio = zrel_depth / (plane_depth + 1e-6)
        vel_consistency = np.random.uniform(0.6, 1.0)
        spatial_stability = 1.0 / (1.0 + np.exp((drift - 75.0) / 30.0))
        temporal_fitness = np.exp(-0.5 * ((duration - 0.25) / 0.125) ** 2)

        # Create feature vector
        features = np.array([
            zrel_depth, plane_depth, ang_depth, z_depth,
            drift, vzrel, vplane, vang, vz, ray_vel,
            duration, scale_factor, palm_width, trigger_count,
            depth_ratio, vel_consistency, spatial_stability, temporal_fitness
        ], dtype=float)

        features_list.append(features)
        labels_list.append(True)

    logger.info(f"Generating {num_samples} synthetic non-tap examples...")

    # Generate negative examples (not taps)
    for i in range(num_samples):
        # Non-taps can have various failure modes
        failure_mode = np.random.choice(['insufficient_depth', 'too_much_drift',
                                        'wrong_duration', 'low_velocity'])

        if failure_mode == 'insufficient_depth':
            # Too shallow press
            zrel_depth = np.random.uniform(0, cfg.ZREL_MIN_PRESS_DEPTH * 0.5)
            plane_depth = np.random.uniform(0, cfg.PLANE_MIN_PRESS_DEPTH * 0.5)
            ang_depth = np.random.uniform(0, cfg.ANG_MIN_PRESS_DEPTH * 0.5)
            z_depth = np.random.uniform(0, cfg.TAP_MIN_PRESS_DEPTH * 0.5)
            drift = np.random.uniform(0, cfg.TAP_MAX_XY_DRIFT * 0.7)
            duration = np.random.uniform(cfg.TAP_MIN_DURATION, cfg.TAP_MAX_DURATION)
        elif failure_mode == 'too_much_drift':
            # Hand moved too much
            zrel_depth = np.random.uniform(cfg.ZREL_MIN_PRESS_DEPTH, 0.04)
            plane_depth = np.random.uniform(cfg.PLANE_MIN_PRESS_DEPTH, 0.04)
            ang_depth = np.random.uniform(cfg.ANG_MIN_PRESS_DEPTH, 30.0)
            z_depth = np.random.uniform(cfg.TAP_MIN_PRESS_DEPTH, 0.04)
            drift = np.random.uniform(cfg.TAP_MAX_XY_DRIFT, cfg.TAP_MAX_XY_DRIFT * 2.0)
            duration = np.random.uniform(cfg.TAP_MIN_DURATION, cfg.TAP_MAX_DURATION)
        elif failure_mode == 'wrong_duration':
            # Too long or too short
            zrel_depth = np.random.uniform(cfg.ZREL_MIN_PRESS_DEPTH, 0.04)
            plane_depth = np.random.uniform(cfg.PLANE_MIN_PRESS_DEPTH, 0.04)
            ang_depth = np.random.uniform(cfg.ANG_MIN_PRESS_DEPTH, 30.0)
            z_depth = np.random.uniform(cfg.TAP_MIN_PRESS_DEPTH, 0.04)
            drift = np.random.uniform(0, cfg.TAP_MAX_XY_DRIFT * 0.7)
            duration_choice = np.random.choice(['too_short', 'too_long'])
            if duration_choice == 'too_short':
                duration = np.random.uniform(0.001, cfg.TAP_MIN_DURATION * 0.8)
            else:
                duration = np.random.uniform(cfg.TAP_MAX_DURATION, cfg.TAP_MAX_DURATION * 2.0)
        else:  # low_velocity
            # Not enough movement
            zrel_depth = np.random.uniform(0, cfg.ZREL_MIN_PRESS_DEPTH * 1.5)
            plane_depth = np.random.uniform(0, cfg.PLANE_MIN_PRESS_DEPTH * 1.5)
            ang_depth = np.random.uniform(0, cfg.ANG_MIN_PRESS_DEPTH * 1.5)
            z_depth = np.random.uniform(0, cfg.TAP_MIN_PRESS_DEPTH * 1.5)
            drift = np.random.uniform(0, cfg.TAP_MAX_XY_DRIFT * 0.5)
            duration = np.random.uniform(cfg.TAP_MIN_DURATION, cfg.TAP_MAX_DURATION)

        # Velocity features (varied)
        vzrel = np.random.uniform(0, cfg.TAP_MIN_VEL * 2.0)
        vplane = np.random.uniform(0, cfg.TAP_MIN_VEL * 2.0)
        vang = np.random.uniform(0, 200.0)
        vz = np.random.uniform(0, cfg.TAP_MIN_VEL * 2.0)
        ray_vel = np.random.uniform(0, cfg.RAY_MIN_IN_VEL * 1.5)

        # Hand size features
        scale_factor = np.random.uniform(cfg.MIN_SCALE_FACTOR, cfg.MAX_SCALE_FACTOR)
        palm_width = np.random.uniform(cfg.SMALL_HAND_THRESHOLD, cfg.REFERENCE_PALM_WIDTH)

        # Trigger count (non-taps typically trigger fewer detectors)
        trigger_count = np.random.randint(0, 3)

        # Engineered features
        depth_ratio = zrel_depth / (plane_depth + 1e-6) if plane_depth > 1e-6 else 0.0
        vel_consistency = np.random.uniform(0.0, 0.6)
        spatial_stability = 1.0 / (1.0 + np.exp((drift - 75.0) / 30.0))
        temporal_fitness = np.exp(-0.5 * ((duration - 0.25) / 0.125) ** 2)

        # Create feature vector
        features = np.array([
            zrel_depth, plane_depth, ang_depth, z_depth,
            drift, vzrel, vplane, vang, vz, ray_vel,
            duration, scale_factor, palm_width, trigger_count,
            depth_ratio, vel_consistency, spatial_stability, temporal_fitness
        ], dtype=float)

        features_list.append(features)
        labels_list.append(False)

    logger.info(f"Generated {len(features_list)} total examples "
               f"({sum(labels_list)} positive, {len(labels_list) - sum(labels_list)} negative)")

    return features_list, labels_list


def train_classifier(num_samples=1000, learning_rate=0.01, epochs=10,
                     model_path='models/tap_model.json'):
    """
    Train the tap classifier on synthetic data.

    Args:
        num_samples (int): Number of samples per class
        learning_rate (float): Learning rate for training
        epochs (int): Number of training epochs
        model_path (str): Path to save trained model
    """
    logger.info("Starting classifier training...")

    # Generate training data
    features_list, labels_list = generate_synthetic_tap_data(num_samples)

    # Initialize classifier
    classifier = TapClassifier(learning_rate=learning_rate)

    # Train
    logger.info(f"Training classifier for {epochs} epochs...")
    avg_loss = classifier.batch_train(features_list, labels_list, epochs=epochs)

    # Get performance metrics
    metrics = classifier.get_performance_metrics()
    logger.info(f"Training complete!")
    logger.info(f"  Average loss: {avg_loss:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")

    # Save model
    classifier.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    return classifier


def evaluate_classifier(model_path='models/tap_model.json', num_test_samples=500):
    """
    Evaluate a trained classifier on test data.

    Args:
        model_path (str): Path to trained model
        num_test_samples (int): Number of test samples per class
    """
    logger.info("Evaluating classifier...")

    # Load classifier
    classifier = TapClassifier(model_path=model_path)

    # Generate test data
    features_list, labels_list = generate_synthetic_tap_data(num_test_samples)

    # Reset metrics
    classifier.reset_metrics()

    # Evaluate
    correct = 0
    total = len(features_list)

    for features, label in zip(features_list, labels_list):
        prob = classifier.predict(features)
        prediction = prob >= 0.5

        if prediction == label:
            correct += 1

        # Update confusion matrix
        if label and prediction:
            classifier.true_positives += 1
        elif label and not prediction:
            classifier.false_negatives += 1
        elif not label and prediction:
            classifier.false_positives += 1
        else:
            classifier.true_negatives += 1

    # Get metrics
    metrics = classifier.get_performance_metrics()

    logger.info("Evaluation Results:")
    logger.info(f"  Test Samples: {total}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    True Positives: {metrics['true_positives']}")
    logger.info(f"    False Positives: {metrics['false_positives']}")
    logger.info(f"    True Negatives: {metrics['true_negatives']}")
    logger.info(f"    False Negatives: {metrics['false_negatives']}")


def show_feature_importance(model_path='models/tap_model.json'):
    """
    Display feature importance from trained model.

    Args:
        model_path (str): Path to trained model
    """
    logger.info("Loading model for feature importance analysis...")

    # Load classifier
    classifier = TapClassifier(model_path=model_path)

    # Get feature importance
    importance = classifier.get_feature_importance()

    logger.info("\nFeature Importance (normalized):")
    logger.info("=" * 50)

    for i, (feature_name, score) in enumerate(importance.items(), 1):
        bar = "█" * int(score * 50)
        logger.info(f"{i:2d}. {feature_name:20s} {bar} {score:.4f}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train and evaluate TapClassifier')

    parser.add_argument('--train', action='store_true',
                       help='Train the classifier')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the classifier')
    parser.add_argument('--feature-importance', action='store_true',
                       help='Show feature importance')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of training samples per class (default: 1000)')
    parser.add_argument('--test-samples', type=int, default=500,
                       help='Number of test samples per class (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--model-path', type=str, default='models/tap_model.json',
                       help='Path to model file (default: models/tap_model.json)')

    args = parser.parse_args()

    # Ensure models directory exists
    Path('../models').mkdir(exist_ok=True)

    # Execute requested operations
    if args.train:
        train_classifier(
            num_samples=args.samples,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            model_path=args.model_path
        )

    if args.evaluate:
        evaluate_classifier(
            model_path=args.model_path,
            num_test_samples=args.test_samples
        )

    if args.feature_importance:
        show_feature_importance(model_path=args.model_path)

    # If no action specified, show help
    if not (args.train or args.evaluate or args.feature_importance):
        parser.print_help()
        print("\nExample usage:")
        print("  python train_tap_classifier.py --train --samples 1000")
        print("  python train_tap_classifier.py --evaluate")
        print("  python train_tap_classifier.py --feature-importance")


if __name__ == '__main__':
    main()

