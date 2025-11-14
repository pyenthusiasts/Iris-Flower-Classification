"""
Utility functions for the Iris Classification package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LOG_FORMAT, LOG_LEVEL


def setup_logging(
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = None
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=handlers
    )


def format_prediction_result(
    prediction: int,
    target_names: list,
    probabilities: Optional[list] = None
) -> str:
    """
    Format prediction result as a readable string.

    Args:
        prediction: Predicted class index
        target_names: Names of target classes
        probabilities: Optional prediction probabilities

    Returns:
        Formatted string
    """
    result = f"Predicted Species: {target_names[prediction]}"

    if probabilities is not None:
        result += "\n\nPrediction Probabilities:"
        for i, (name, prob) in enumerate(zip(target_names, probabilities)):
            result += f"\n  {name:12s}: {prob:6.2%}"

    return result


def validate_sample_input(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> bool:
    """
    Validate sample input values.

    Args:
        sepal_length: Sepal length in cm
        sepal_width: Sepal width in cm
        petal_length: Petal length in cm
        petal_width: Petal width in cm

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not all(isinstance(x, (int, float)) for x in [sepal_length, sepal_width, petal_length, petal_width]):
        raise ValueError("All measurements must be numeric values")

    if not all(x > 0 for x in [sepal_length, sepal_width, petal_length, petal_width]):
        raise ValueError("All measurements must be positive values")

    # Reasonable ranges based on Iris dataset
    if not (4.0 <= sepal_length <= 8.0):
        logging.warning(f"Sepal length {sepal_length} is outside typical range [4.0, 8.0]")

    if not (2.0 <= sepal_width <= 4.5):
        logging.warning(f"Sepal width {sepal_width} is outside typical range [2.0, 4.5]")

    if not (1.0 <= petal_length <= 7.0):
        logging.warning(f"Petal length {petal_length} is outside typical range [1.0, 7.0]")

    if not (0.1 <= petal_width <= 2.5):
        logging.warning(f"Petal width {petal_width} is outside typical range [0.1, 2.5]")

    return True
