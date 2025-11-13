"""
Custom exceptions for the Iris Classifier package.
"""


class IrisClassifierError(Exception):
    """Base exception for iris_classifier package."""
    pass


class DataLoadError(IrisClassifierError):
    """Exception raised when data loading fails."""
    pass


class ModelNotFoundError(IrisClassifierError):
    """Exception raised when a model is not found."""
    pass


class ModelNotTrainedError(IrisClassifierError):
    """Exception raised when attempting to use an untrained model."""
    pass


class InvalidModelError(IrisClassifierError):
    """Exception raised when an invalid model name is provided."""
    pass


class InvalidInputError(IrisClassifierError):
    """Exception raised when invalid input is provided."""
    pass


class PredictionError(IrisClassifierError):
    """Exception raised when prediction fails."""
    pass


class EvaluationError(IrisClassifierError):
    """Exception raised when model evaluation fails."""
    pass


class ConfigurationError(IrisClassifierError):
    """Exception raised when configuration is invalid."""
    pass


class ModelSaveError(IrisClassifierError):
    """Exception raised when model saving fails."""
    pass


class ModelLoadError(IrisClassifierError):
    """Exception raised when model loading fails."""
    pass
