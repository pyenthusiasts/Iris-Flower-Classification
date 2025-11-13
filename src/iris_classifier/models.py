"""
Machine learning models for Iris classification.
"""

from typing import Dict, Any, Optional
import logging
import joblib
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

from .config import RANDOM_STATE, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating and managing machine learning models.
    """

    MODEL_CONFIGS = {
        "decision_tree": {
            "class": DecisionTreeClassifier,
            "params": {
                "random_state": RANDOM_STATE,
                "max_depth": 5,
                "min_samples_split": 2
            },
            "description": "Decision Tree Classifier"
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "random_state": RANDOM_STATE,
                "max_depth": 5,
                "min_samples_split": 2
            },
            "description": "Random Forest Classifier"
        },
        "svm": {
            "class": SVC,
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "random_state": RANDOM_STATE
            },
            "description": "Support Vector Machine"
        },
        "knn": {
            "class": KNeighborsClassifier,
            "params": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski"
            },
            "description": "K-Nearest Neighbors"
        },
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "random_state": RANDOM_STATE,
                "max_iter": 1000,
                "solver": "lbfgs"
            },
            "description": "Logistic Regression"
        },
        "naive_bayes": {
            "class": GaussianNB,
            "params": {},
            "description": "Gaussian Naive Bayes"
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": RANDOM_STATE
            },
            "description": "Gradient Boosting Classifier"
        },
        "mlp": {
            "class": MLPClassifier,
            "params": {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "early_stopping": True
            },
            "description": "Multi-Layer Perceptron"
        }
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """
        Create a machine learning model by name.

        Args:
            model_name: Name of the model to create
            custom_params: Optional custom parameters to override defaults

        Returns:
            Instantiated model

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in cls.MODEL_CONFIGS:
            available = ", ".join(cls.MODEL_CONFIGS.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available}"
            )

        config = cls.MODEL_CONFIGS[model_name]
        params = config["params"].copy()

        # Override with custom parameters
        if custom_params:
            params.update(custom_params)

        logger.info(f"Creating {config['description']} with params: {params}")

        return config["class"](**params)

    @classmethod
    def get_all_models(cls) -> Dict[str, BaseEstimator]:
        """
        Create all available models with default parameters.

        Returns:
            Dictionary mapping model names to instantiated models
        """
        logger.info("Creating all available models...")
        return {
            name: cls.create_model(name)
            for name in cls.MODEL_CONFIGS.keys()
        }

    @classmethod
    def get_model_description(cls, model_name: str) -> str:
        """
        Get the description of a model.

        Args:
            model_name: Name of the model

        Returns:
            Model description

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        return cls.MODEL_CONFIGS[model_name]["description"]

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        List all available models and their descriptions.

        Returns:
            Dictionary mapping model names to descriptions
        """
        return {
            name: config["description"]
            for name, config in cls.MODEL_CONFIGS.items()
        }

    @staticmethod
    def save_model(model: BaseEstimator, model_name: str, filename: Optional[str] = None) -> Path:
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save
            model_name: Name of the model
            filename: Optional custom filename

        Returns:
            Path to saved model file
        """
        if filename is None:
            filename = f"{model_name}_model.pkl"

        filepath = MODELS_DIR / filename
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    @staticmethod
    def load_model(filename: str) -> BaseEstimator:
        """
        Load a trained model from disk.

        Args:
            filename: Name of the model file

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = MODELS_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

        return model


class ModelTrainer:
    """
    Train and manage machine learning models.
    """

    def __init__(self, model: BaseEstimator, model_name: str):
        """
        Initialize the trainer.

        Args:
            model: Model instance to train
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        self.is_trained = False

    def train(self, X_train, y_train) -> "ModelTrainer":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.model_name} training complete")

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predicted labels

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Predicted probabilities

        Raises:
            ValueError: If model is not trained or doesn't support predict_proba
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"{self.model_name} doesn't support probability predictions")

        return self.model.predict_proba(X)

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save the trained model.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved model

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        return ModelFactory.save_model(self.model, self.model_name, filename)
