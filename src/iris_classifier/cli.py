"""
Command-line interface for Iris Classification.
"""

import argparse
import logging
import sys
from pathlib import Path

from .data_loader import IrisDataLoader
from .models import ModelFactory, ModelTrainer
from .evaluator import ModelEvaluator
from .visualizer import IrisVisualizer
from .utils import setup_logging, format_prediction_result, validate_sample_input
from .config import AVAILABLE_MODELS, TARGET_NAMES

logger = logging.getLogger(__name__)


def train_command(args):
    """Train a model."""
    setup_logging(args.log_level)

    logger.info(f"Training {args.model} model...")

    # Load data
    data_loader = IrisDataLoader(scale=args.scale)
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        test_size=args.test_size
    )

    # Create and train model
    model = ModelFactory.create_model(args.model)
    trainer = ModelTrainer(model, args.model)
    trainer.train(X_train, y_train)

    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, args.model)
    evaluator.print_evaluation_report(results)

    # Save model if requested
    if args.save:
        save_path = trainer.save()
        print(f"\nModel saved to: {save_path}")


def compare_command(args):
    """Compare multiple models."""
    setup_logging(args.log_level)

    logger.info("Comparing all available models...")

    # Load data
    data_loader = IrisDataLoader(scale=args.scale)
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        test_size=args.test_size
    )

    # Get all models
    models = ModelFactory.get_all_models()

    # Compare
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(
        models, X_train, y_train, X_test, y_test, cv=args.cv
    )

    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)

    # Get best model
    best_model = evaluator.get_best_model(comparison_df)
    print(f"\nBest Model: {best_model}")

    # Visualize if requested
    if args.plot:
        visualizer = IrisVisualizer()
        visualizer.plot_model_comparison(comparison_df)


def predict_command(args):
    """Make a prediction on a new sample."""
    setup_logging(args.log_level)

    # Validate input
    validate_sample_input(
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width
    )

    # Load data
    data_loader = IrisDataLoader(scale=args.scale)

    # Load or train model
    if args.model_file:
        logger.info(f"Loading model from {args.model_file}...")
        model = ModelFactory.load_model(args.model_file)
    else:
        logger.info(f"Training {args.model} model...")
        X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
        model = ModelFactory.create_model(args.model)
        model.fit(X_train, y_train)

    # Prepare sample
    sample = data_loader.predict_sample(
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width
    )

    # Predict
    prediction = model.predict(sample)[0]

    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(sample)[0]

    # Format and print result
    result = format_prediction_result(prediction, TARGET_NAMES, probabilities)
    print("\n" + "="*50)
    print(result)
    print("="*50)


def visualize_command(args):
    """Create visualizations."""
    setup_logging(args.log_level)

    logger.info("Creating visualizations...")

    # Load data
    data_loader = IrisDataLoader()
    X, y = data_loader.get_full_dataset()

    visualizer = IrisVisualizer()

    if args.type == 'all' or args.type == 'distribution':
        visualizer.plot_feature_distributions(X, y)

    if args.type == 'all' or args.type == 'pairplot':
        visualizer.plot_pairplot(X, y)

    if args.type == 'all' or args.type == 'correlation':
        visualizer.plot_correlation_matrix(X)

    if args.type == 'all' or args.type == 'pca':
        visualizer.plot_pca_visualization(X, y)

    if args.type == 'all' or args.type == 'classes':
        visualizer.plot_class_distribution(y)


def info_command(args):
    """Display dataset information."""
    setup_logging(args.log_level)

    data_loader = IrisDataLoader()
    info = data_loader.get_dataset_info()

    print("\n" + "="*60)
    print("IRIS DATASET INFORMATION")
    print("="*60)
    print(f"Number of samples:  {info['n_samples']}")
    print(f"Number of features: {info['n_features']}")
    print(f"Number of classes:  {info['n_classes']}")
    print(f"\nFeatures: {', '.join(info['feature_names'])}")
    print(f"\nClasses: {', '.join(info['target_names'])}")
    print(f"\nClass distribution:")
    for class_idx, count in info['class_distribution'].items():
        print(f"  {info['target_names'][class_idx]:12s}: {count}")

    if args.stats:
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        print(data_loader.get_feature_statistics().to_string())

    print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iris Flower Classification - Machine Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--model',
        choices=AVAILABLE_MODELS,
        default='decision_tree',
        help='Model to train'
    )
    train_parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set size (0.0 to 1.0)'
    )
    train_parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features using StandardScaler'
    )
    train_parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model'
    )
    train_parser.set_defaults(func=train_command)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all models')
    compare_parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set size (0.0 to 1.0)'
    )
    compare_parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features using StandardScaler'
    )
    compare_parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    compare_parser.add_argument(
        '--plot',
        action='store_true',
        help='Show comparison plot'
    )
    compare_parser.set_defaults(func=compare_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict iris species')
    predict_parser.add_argument(
        'sepal_length',
        type=float,
        help='Sepal length in cm'
    )
    predict_parser.add_argument(
        'sepal_width',
        type=float,
        help='Sepal width in cm'
    )
    predict_parser.add_argument(
        'petal_length',
        type=float,
        help='Petal length in cm'
    )
    predict_parser.add_argument(
        'petal_width',
        type=float,
        help='Petal width in cm'
    )
    predict_parser.add_argument(
        '--model',
        choices=AVAILABLE_MODELS,
        default='decision_tree',
        help='Model to use for prediction'
    )
    predict_parser.add_argument(
        '--model-file',
        type=str,
        help='Load model from file'
    )
    predict_parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features using StandardScaler'
    )
    predict_parser.set_defaults(func=predict_command)

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument(
        '--type',
        choices=['all', 'distribution', 'pairplot', 'correlation', 'pca', 'classes'],
        default='all',
        help='Type of visualization'
    )
    viz_parser.set_defaults(func=visualize_command)

    # Info command
    info_parser = subparsers.add_parser('info', help='Display dataset information')
    info_parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed feature statistics'
    )
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
