"""
Configuration settings for the Iris Classification package.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.2

# Available models
AVAILABLE_MODELS = [
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "logistic_regression",
    "naive_bayes",
    "gradient_boosting",
    "mlp"
]

# Feature names
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Target names
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = "seaborn-v0_8-darkgrid"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
