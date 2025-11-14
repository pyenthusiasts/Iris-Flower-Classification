"""Tests for models module."""

import pytest
from sklearn.base import BaseEstimator
from iris_classifier.models import ModelFactory, ModelTrainer
from iris_classifier.data_loader import IrisDataLoader


class TestModelFactory:
    """Test cases for ModelFactory class."""

    def test_create_decision_tree(self):
        """Test creating decision tree model."""
        model = ModelFactory.create_model('decision_tree')
        assert model is not None
        assert isinstance(model, BaseEstimator)

    def test_create_random_forest(self):
        """Test creating random forest model."""
        model = ModelFactory.create_model('random_forest')
        assert model is not None

    def test_create_svm(self):
        """Test creating SVM model."""
        model = ModelFactory.create_model('svm')
        assert model is not None

    def test_create_knn(self):
        """Test creating KNN model."""
        model = ModelFactory.create_model('knn')
        assert model is not None

    def test_create_invalid_model(self):
        """Test creating invalid model raises error."""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model')

    def test_get_all_models(self):
        """Test getting all models."""
        models = ModelFactory.get_all_models()
        assert len(models) == 8
        assert 'decision_tree' in models
        assert 'random_forest' in models

    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelFactory.list_available_models()
        assert isinstance(models, dict)
        assert len(models) == 8

    def test_get_model_description(self):
        """Test getting model description."""
        desc = ModelFactory.get_model_description('decision_tree')
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_custom_params(self):
        """Test creating model with custom parameters."""
        model = ModelFactory.create_model('decision_tree', {'max_depth': 3})
        assert model.max_depth == 3


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    @pytest.fixture
    def data(self):
        """Fixture providing train/test data."""
        loader = IrisDataLoader()
        return loader.get_train_test_split()

    def test_initialization(self):
        """Test trainer initialization."""
        model = ModelFactory.create_model('decision_tree')
        trainer = ModelTrainer(model, 'decision_tree')
        assert trainer.model is not None
        assert trainer.model_name == 'decision_tree'
        assert not trainer.is_trained

    def test_training(self, data):
        """Test model training."""
        X_train, X_test, y_train, y_test = data
        model = ModelFactory.create_model('decision_tree')
        trainer = ModelTrainer(model, 'decision_tree')

        trainer.train(X_train, y_train)
        assert trainer.is_trained

    def test_prediction(self, data):
        """Test making predictions."""
        X_train, X_test, y_train, y_test = data
        model = ModelFactory.create_model('decision_tree')
        trainer = ModelTrainer(model, 'decision_tree')

        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_prediction_without_training(self, data):
        """Test prediction without training raises error."""
        X_train, X_test, y_train, y_test = data
        model = ModelFactory.create_model('decision_tree')
        trainer = ModelTrainer(model, 'decision_tree')

        with pytest.raises(ValueError):
            trainer.predict(X_test)

    def test_predict_proba(self, data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = data
        model = ModelFactory.create_model('decision_tree')
        trainer = ModelTrainer(model, 'decision_tree')

        trainer.train(X_train, y_train)
        probas = trainer.predict_proba(X_test)

        assert probas.shape == (len(X_test), 3)
        assert all(probas.sum(axis=1) - 1.0 < 1e-6)  # Probabilities sum to 1
