"""Tests for utils module."""

import pytest
from iris_classifier.utils import format_prediction_result, validate_sample_input


class TestUtils:
    """Test cases for utility functions."""

    def test_format_prediction_result_without_probabilities(self):
        """Test formatting prediction result without probabilities."""
        result = format_prediction_result(0, ['setosa', 'versicolor', 'virginica'])
        assert 'setosa' in result
        assert 'Predicted Species' in result

    def test_format_prediction_result_with_probabilities(self):
        """Test formatting prediction result with probabilities."""
        result = format_prediction_result(
            0,
            ['setosa', 'versicolor', 'virginica'],
            [0.9, 0.05, 0.05]
        )
        assert 'setosa' in result
        assert 'Probabilities' in result
        assert '90.00%' in result

    def test_validate_sample_input_valid(self):
        """Test validating valid sample input."""
        assert validate_sample_input(5.0, 3.5, 1.4, 0.2) is True

    def test_validate_sample_input_negative_values(self):
        """Test validating negative values raises error."""
        with pytest.raises(ValueError, match="positive values"):
            validate_sample_input(-5.0, 3.5, 1.4, 0.2)

    def test_validate_sample_input_non_numeric(self):
        """Test validating non-numeric values raises error."""
        with pytest.raises(ValueError, match="numeric values"):
            validate_sample_input("5.0", 3.5, 1.4, 0.2)

    def test_validate_sample_input_zero_values(self):
        """Test validating zero values raises error."""
        with pytest.raises(ValueError, match="positive values"):
            validate_sample_input(0, 3.5, 1.4, 0.2)
