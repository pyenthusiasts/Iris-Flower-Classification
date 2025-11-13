#!/usr/bin/env python3
"""
Main script for Iris Flower Classification.

This provides a simple interface to the iris_classifier package,
demonstrating basic usage and model comparison.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iris_classifier import IrisDataLoader, ModelFactory, ModelEvaluator, IrisVisualizer
from iris_classifier.utils import setup_logging

def main():
    """Main function demonstrating the package usage."""
    # Setup logging
    setup_logging('INFO')

    print("\n" + "="*70)
    print("IRIS FLOWER CLASSIFICATION - COMPREHENSIVE ANALYSIS")
    print("="*70)

    # Load data
    print("\n[1/5] Loading Iris dataset...")
    data_loader = IrisDataLoader()
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split()

    # Display dataset info
    info = data_loader.get_dataset_info()
    print(f"  ✓ Loaded {info['n_samples']} samples with {info['n_features']} features")
    print(f"  ✓ Classes: {', '.join(info['target_names'])}")

    # Compare all models
    print("\n[2/5] Training and comparing all models...")
    models = ModelFactory.get_all_models()
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(
        models, X_train, y_train, X_test, y_test, cv=5
    )

    print("\n" + "-"*70)
    print("MODEL COMPARISON RESULTS")
    print("-"*70)
    print(comparison_df.to_string(index=False))
    print("-"*70)

    best_model_name = evaluator.get_best_model(comparison_df)
    best_accuracy = comparison_df[comparison_df['Model'] == best_model_name]['Accuracy'].values[0]
    print(f"\n  ✓ Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Train best model
    print(f"\n[3/5] Training best model ({best_model_name})...")
    best_model = ModelFactory.create_model(best_model_name)
    best_model.fit(X_train, y_train)
    print(f"  ✓ Model trained successfully")

    # Detailed evaluation
    print(f"\n[4/5] Detailed evaluation of {best_model_name}...")
    results = evaluator.evaluate_model(best_model, X_test, y_test, best_model_name)
    evaluator.print_evaluation_report(results)

    # Make a sample prediction
    print("\n[5/5] Making sample prediction...")
    sample = data_loader.predict_sample(5.0, 3.6, 1.4, 0.2)
    prediction = best_model.predict(sample)[0]

    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(sample)[0]
        print(f"\n  Sample: [Sepal Length: 5.0, Sepal Width: 3.6, Petal Length: 1.4, Petal Width: 0.2]")
        print(f"  Predicted Species: {info['target_names'][prediction]}")
        print(f"\n  Probabilities:")
        for i, (name, prob) in enumerate(zip(info['target_names'], probabilities)):
            print(f"    {name:12s}: {prob:6.2%}")
    else:
        print(f"\n  Sample: [5.0, 3.6, 1.4, 0.2]")
        print(f"  Predicted Species: {info['target_names'][prediction]}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nTo explore more features:")
    print("  • Use the CLI: python -m iris_classifier.cli --help")
    print("  • Check out the notebooks in the notebooks/ directory")
    print("  • Run tests: pytest tests/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
