"""Tests for data_loader module."""

import pytest
import pandas as pd
import numpy as np
from iris_classifier.data_loader import IrisDataLoader


class TestIrisDataLoader:
    """Test cases for IrisDataLoader class."""

    def test_initialization(self):
        """Test data loader initialization."""
        loader = IrisDataLoader()
        assert loader.data is not None
        assert loader.target is not None
        assert isinstance(loader.data, pd.DataFrame)
        assert isinstance(loader.target, pd.Series)

    def test_data_shape(self):
        """Test data dimensions."""
        loader = IrisDataLoader()
        assert loader.data.shape == (150, 4)
        assert len(loader.target) == 150

    def test_feature_names(self):
        """Test feature names."""
        loader = IrisDataLoader()
        expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        assert loader.feature_names == expected_features

    def test_target_names(self):
        """Test target names."""
        loader = IrisDataLoader()
        assert len(loader.target_names) == 3
        assert 'setosa' in loader.target_names

    def test_train_test_split(self):
        """Test train-test split."""
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.3)

        assert len(X_train) == 105
        assert len(X_test) == 45
        assert len(y_train) == 105
        assert len(y_test) == 45

    def test_train_test_split_custom_size(self):
        """Test train-test split with custom size."""
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)

        assert len(X_train) == 120
        assert len(X_test) == 30

    def test_scaling(self):
        """Test feature scaling."""
        loader = IrisDataLoader(scale=True)
        X_train, X_test, y_train, y_test = loader.get_train_test_split()

        # Check that scaled data has mean close to 0 and std close to 1
        assert np.abs(X_train.mean().mean()) < 0.1
        assert np.abs(X_train.std().mean() - 1.0) < 0.5

    def test_get_full_dataset(self):
        """Test getting full dataset."""
        loader = IrisDataLoader()
        X, y = loader.get_full_dataset()

        assert len(X) == 150
        assert len(y) == 150

    def test_dataset_info(self):
        """Test dataset information retrieval."""
        loader = IrisDataLoader()
        info = loader.get_dataset_info()

        assert info['n_samples'] == 150
        assert info['n_features'] == 4
        assert info['n_classes'] == 3

    def test_feature_statistics(self):
        """Test feature statistics."""
        loader = IrisDataLoader()
        stats = loader.get_feature_statistics()

        assert isinstance(stats, pd.DataFrame)
        assert 'mean' in stats.index
        assert 'std' in stats.index

    def test_predict_sample(self):
        """Test creating prediction sample."""
        loader = IrisDataLoader()
        sample = loader.predict_sample(5.0, 3.6, 1.4, 0.2)

        assert isinstance(sample, pd.DataFrame)
        assert sample.shape == (1, 4)
