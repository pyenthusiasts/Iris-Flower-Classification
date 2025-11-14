"""Tests for evaluator module."""

import pytest
import pandas as pd
import numpy as np
from iris_classifier.evaluator import ModelEvaluator
from iris_classifier.models import ModelFactory
from iris_classifier.data_loader import IrisDataLoader


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    @pytest.fixture
    def trained_model(self):
        """Fixture providing a trained model and test data."""
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test = loader.get_train_test_split()

        model = ModelFactory.create_model('decision_tree')
        model.fit(X_train, y_train)

        return model, X_test, y_test

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.target_names is not None

    def test_evaluate_model(self, trained_model):
        """Test model evaluation."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        results = evaluator.evaluate_model(model, X_test, y_test, 'test_model')

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'confusion_matrix' in results
        assert results['accuracy'] >= 0.0
        assert results['accuracy'] <= 1.0

    def test_cross_validate_model(self):
        """Test cross-validation."""
        loader = IrisDataLoader()
        X, y = loader.get_full_dataset()
        model = ModelFactory.create_model('decision_tree')

        evaluator = ModelEvaluator()
        cv_results = evaluator.cross_validate_model(model, X, y, cv=5)

        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert cv_results['mean_score'] >= 0.0
        assert cv_results['mean_score'] <= 1.0

    def test_compare_models(self):
        """Test model comparison."""
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test = loader.get_train_test_split()

        models = {
            'decision_tree': ModelFactory.create_model('decision_tree'),
            'knn': ModelFactory.create_model('knn')
        }

        evaluator = ModelEvaluator()
        comparison_df = evaluator.compare_models(models, X_train, y_train, X_test, y_test, cv=3)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns

    def test_get_best_model(self):
        """Test getting best model."""
        comparison_df = pd.DataFrame({
            'Model': ['model1', 'model2', 'model3'],
            'Accuracy': [0.8, 0.9, 0.85]
        })

        evaluator = ModelEvaluator()
        best_model = evaluator.get_best_model(comparison_df, 'Accuracy')

        assert best_model == 'model2'

    def test_get_best_model_invalid_metric(self):
        """Test getting best model with invalid metric."""
        comparison_df = pd.DataFrame({
            'Model': ['model1', 'model2'],
            'Accuracy': [0.8, 0.9]
        })

        evaluator = ModelEvaluator()

        with pytest.raises(ValueError):
            evaluator.get_best_model(comparison_df, 'invalid_metric')
