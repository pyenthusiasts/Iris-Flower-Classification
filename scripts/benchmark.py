#!/usr/bin/env python3
"""
Benchmark script for Iris Classification models.

This script measures the performance of different models including:
- Training time
- Prediction time
- Memory usage
- Accuracy
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iris_classifier import IrisDataLoader, ModelFactory, ModelEvaluator


class ModelBenchmark:
    """Benchmark different ML models."""

    def __init__(self):
        """Initialize benchmark."""
        self.data_loader = IrisDataLoader()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_loader.get_train_test_split()
        self.results = []

    def benchmark_model(self, model_name: str, n_iterations: int = 100) -> Dict:
        """
        Benchmark a single model.

        Args:
            model_name: Name of the model to benchmark
            n_iterations: Number of prediction iterations for timing

        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking {model_name}...")

        # Measure training time and memory
        tracemalloc.start()
        start_time = time.time()

        model = ModelFactory.create_model(model_name)
        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure prediction time
        prediction_times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = model.predict(self.X_test)
            prediction_times.append(time.time() - start)

        avg_prediction_time = np.mean(prediction_times)
        std_prediction_time = np.std(prediction_times)

        # Evaluate accuracy
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, self.X_test, self.y_test, model_name)

        benchmark_results = {
            "model": model_name,
            "training_time_s": training_time,
            "avg_prediction_time_ms": avg_prediction_time * 1000,
            "std_prediction_time_ms": std_prediction_time * 1000,
            "predictions_per_second": 1 / avg_prediction_time if avg_prediction_time > 0 else 0,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "accuracy": results["accuracy"],
            "f1_score": results["f1_score"]
        }

        print(f"  Training time: {training_time:.4f}s")
        print(f"  Avg prediction time: {avg_prediction_time * 1000:.4f}ms")
        print(f"  Predictions/sec: {benchmark_results['predictions_per_second']:.2f}")
        print(f"  Peak memory: {benchmark_results['peak_memory_mb']:.2f}MB")
        print(f"  Accuracy: {results['accuracy']:.4f}")

        return benchmark_results

    def run_all_benchmarks(self) -> pd.DataFrame:
        """
        Run benchmarks for all available models.

        Returns:
            DataFrame with benchmark results
        """
        print("="*70)
        print("IRIS CLASSIFICATION MODEL BENCHMARKS")
        print("="*70)

        models = ModelFactory.list_available_models()

        for model_name in models.keys():
            try:
                result = self.benchmark_model(model_name)
                self.results.append(result)
            except Exception as e:
                print(f"  ERROR: Failed to benchmark {model_name}: {str(e)}")

        df = pd.DataFrame(self.results)
        df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

        return df

    def print_summary(self, df: pd.DataFrame):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print("="*70)

        # Print best performers
        print("\nBEST PERFORMERS:")
        print(f"  Fastest Training: {df.loc[df['training_time_s'].idxmin(), 'model']}")
        print(f"  Fastest Prediction: {df.loc[df['avg_prediction_time_ms'].idxmin(), 'model']}")
        print(f"  Lowest Memory: {df.loc[df['peak_memory_mb'].idxmin(), 'model']}")
        print(f"  Highest Accuracy: {df.loc[df['accuracy'].idxmax(), 'model']}")

        # Print trade-offs
        print("\nTRADE-OFFS:")
        speed_accuracy = df.nsmallest(3, 'avg_prediction_time_ms')[['model', 'avg_prediction_time_ms', 'accuracy']]
        print("Top 3 Fastest Models:")
        print(speed_accuracy.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def main():
    """Main function."""
    benchmark = ModelBenchmark()
    results_df = benchmark.run_all_benchmarks()
    benchmark.print_summary(results_df)

    # Save results
    output_file = Path("benchmark_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
