"""
Model evaluation and comparison module.
"""

from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

from .config import TARGET_NAMES, RANDOM_STATE

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and compare machine learning models.
    """

    def __init__(self, target_names: List[str] = None):
        """
        Initialize the evaluator.

        Args:
            target_names: Names of target classes
        """
        self.target_names = target_names or TARGET_NAMES

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test,
        y_test,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: True test labels
            model_name: Name of the model

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=self.target_names,
            zero_division=0
        )

        # Try to get probability predictions for ROC AUC
        try:
            y_pred_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test,
                y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
        except (AttributeError, ValueError):
            roc_auc = None

        results = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "predictions": y_pred
        }

        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")

        return results

    def cross_validate_model(
        self,
        model: BaseEstimator,
        X,
        y,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max()
        }

        logger.info(
            f"Cross-validation - Mean {scoring}: {results['mean_score']:.4f} "
            f"(+/- {results['std_score']:.4f})"
        )

        return results

    def compare_models(
        self,
        models: Dict[str, BaseEstimator],
        X_train,
        y_train,
        X_test,
        y_test,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dictionary mapping model names to model instances
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            cv: Number of cross-validation folds

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")

        results = []

        for model_name, model in models.items():
            try:
                # Train model
                logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)

                # Evaluate on test set
                eval_results = self.evaluate_model(model, X_test, y_test, model_name)

                # Cross-validation
                cv_results = self.cross_validate_model(model, X_train, y_train, cv=cv)

                # Combine results
                results.append({
                    "Model": model_name,
                    "Accuracy": eval_results["accuracy"],
                    "Precision": eval_results["precision"],
                    "Recall": eval_results["recall"],
                    "F1-Score": eval_results["f1_score"],
                    "ROC-AUC": eval_results["roc_auc"] if eval_results["roc_auc"] else np.nan,
                    "CV Mean": cv_results["mean_score"],
                    "CV Std": cv_results["std_score"]
                })

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results.append({
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-Score": np.nan,
                    "ROC-AUC": np.nan,
                    "CV Mean": np.nan,
                    "CV Std": np.nan
                })

        # Create DataFrame and sort by accuracy
        df = pd.DataFrame(results)
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        logger.info("Model comparison complete")

        return df

    def get_best_model(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "Accuracy"
    ) -> str:
        """
        Get the best performing model from comparison results.

        Args:
            comparison_df: DataFrame from compare_models
            metric: Metric to use for comparison

        Returns:
            Name of best model
        """
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison results")

        best_idx = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]

        logger.info(
            f"Best model by {metric}: {best_model} "
            f"({comparison_df.loc[best_idx, metric]:.4f})"
        )

        return best_model

    @staticmethod
    def print_evaluation_report(evaluation_results: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report.

        Args:
            evaluation_results: Results from evaluate_model
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {evaluation_results['model_name']}")
        print(f"{'='*60}")
        print(f"\nAccuracy:  {evaluation_results['accuracy']:.4f}")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall:    {evaluation_results['recall']:.4f}")
        print(f"F1-Score:  {evaluation_results['f1_score']:.4f}")

        if evaluation_results['roc_auc'] is not None:
            print(f"ROC-AUC:   {evaluation_results['roc_auc']:.4f}")

        print(f"\n{'-'*60}")
        print("Classification Report:")
        print(f"{'-'*60}")
        print(evaluation_results['classification_report'])

        print(f"{'-'*60}")
        print("Confusion Matrix:")
        print(f"{'-'*60}")
        print(evaluation_results['confusion_matrix'])
        print(f"{'='*60}\n")
