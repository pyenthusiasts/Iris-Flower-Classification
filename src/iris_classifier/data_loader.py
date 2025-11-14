"""
Data loading and preprocessing module for Iris dataset.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

from .config import RANDOM_STATE, TEST_SIZE, FEATURE_NAMES, TARGET_NAMES

logger = logging.getLogger(__name__)


class IrisDataLoader:
    """
    Load and preprocess the Iris dataset.

    Attributes:
        data (pd.DataFrame): Feature data
        target (pd.Series): Target labels
        target_names (np.ndarray): Names of target classes
        feature_names (list): Names of features
    """

    def __init__(self, scale: bool = False):
        """
        Initialize the data loader.

        Args:
            scale (bool): Whether to scale features using StandardScaler
        """
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.data = None
        self.target = None
        self.target_names = None
        self.feature_names = None
        self._load_data()

    def _load_data(self) -> None:
        """Load the Iris dataset from sklearn."""
        logger.info("Loading Iris dataset...")
        iris = load_iris()

        # Create DataFrame for features
        self.data = pd.DataFrame(
            iris.data,
            columns=FEATURE_NAMES
        )

        # Create Series for target
        self.target = pd.Series(iris.target, name="species")

        # Store metadata
        self.target_names = iris.target_names
        self.feature_names = FEATURE_NAMES

        logger.info(f"Dataset loaded: {self.data.shape[0]} samples, "
                   f"{self.data.shape[1]} features")

    def get_train_test_split(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
            stratify (bool): Whether to stratify split by target

        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={test_size}")

        stratify_by = self.target if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by
        )

        # Scale if needed
        if self.scale:
            logger.info("Scaling features...")
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=self.feature_names,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=self.feature_names,
                index=X_test.index
            )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def get_full_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the full dataset.

        Returns:
            Tuple containing features and target
        """
        return self.data, self.target

    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        return {
            "n_samples": len(self.data),
            "n_features": len(self.feature_names),
            "n_classes": len(self.target_names),
            "feature_names": self.feature_names,
            "target_names": self.target_names.tolist(),
            "class_distribution": self.target.value_counts().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict()
        }

    def get_feature_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of features.

        Returns:
            DataFrame with feature statistics
        """
        return self.data.describe()

    def predict_sample(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> pd.DataFrame:
        """
        Create a DataFrame for a new sample to predict.

        Args:
            sepal_length: Sepal length in cm
            sepal_width: Sepal width in cm
            petal_length: Petal length in cm
            petal_width: Petal width in cm

        Returns:
            DataFrame containing the sample features
        """
        sample = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=self.feature_names
        )

        if self.scale and self.scaler is not None:
            sample = pd.DataFrame(
                self.scaler.transform(sample),
                columns=self.feature_names
            )

        return sample
