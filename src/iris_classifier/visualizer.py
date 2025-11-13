"""
Visualization module for Iris dataset and model results.
"""

from typing import Optional, List, Tuple
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import FIGURE_SIZE, DPI, TARGET_NAMES, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class IrisVisualizer:
    """
    Visualization tools for Iris dataset and model results.
    """

    def __init__(self, figsize: Tuple[int, int] = FIGURE_SIZE, dpi: int = DPI):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size
            dpi: Default DPI for figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_feature_distributions(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        target_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distributions of all features.

        Args:
            data: Feature data
            target: Target labels
            target_names: Names of target classes
            save_path: Optional path to save figure
        """
        logger.info("Plotting feature distributions...")

        target_names = target_names or TARGET_NAMES
        n_features = data.shape[1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, column in enumerate(data.columns):
            for class_idx, class_name in enumerate(target_names):
                class_data = data[target == class_idx][column]
                axes[idx].hist(
                    class_data,
                    alpha=0.6,
                    label=class_name,
                    bins=20
                )

            axes[idx].set_xlabel(column.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {column.replace("_", " ").title()}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_pairplot(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        target_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a pairplot of features colored by target.

        Args:
            data: Feature data
            target: Target labels
            target_names: Names of target classes
            save_path: Optional path to save figure
        """
        logger.info("Creating pairplot...")

        target_names = target_names or TARGET_NAMES

        # Create a copy with species names
        df = data.copy()
        df['species'] = target.map(lambda x: target_names[x])

        # Create pairplot
        pairplot = sns.pairplot(
            df,
            hue='species',
            diag_kind='kde',
            markers=['o', 's', 'D'],
            plot_kws={'alpha': 0.6},
            height=2.5
        )

        pairplot.fig.suptitle('Iris Dataset Pairplot', y=1.02, fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot correlation matrix of features.

        Args:
            data: Feature data
            save_path: Optional path to save figure
        """
        logger.info("Plotting correlation matrix...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate correlation
        corr = data.corr()

        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        target_names: List[str] = None,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            model_name: Name of the model
            save_path: Optional path to save figure
        """
        logger.info(f"Plotting confusion matrix for {model_name}...")

        target_names = target_names or TARGET_NAMES

        fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_names
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')

        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "Accuracy",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of multiple models.

        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            save_path: Optional path to save figure
        """
        logger.info(f"Plotting model comparison by {metric}...")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by metric
        df_sorted = comparison_df.sort_values(metric, ascending=True)

        # Create horizontal bar plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color='skyblue', edgecolor='navy')

        # Color the best model differently
        bars[-1].set_color('lightcoral')

        # Add value labels
        for i, (model, value) in enumerate(zip(df_sorted['Model'], df_sorted[metric])):
            if not np.isnan(value):
                ax.text(value + 0.01, i, f'{value:.4f}', va='center')

        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'Model Comparison by {metric}', fontsize=14, pad=20)
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_pca_visualization(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        target_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot PCA visualization of the dataset.

        Args:
            data: Feature data
            target: Target labels
            target_names: Names of target classes
            save_path: Optional path to save figure
        """
        logger.info("Creating PCA visualization...")

        target_names = target_names or TARGET_NAMES

        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each class
        for class_idx, class_name in enumerate(target_names):
            mask = target == class_idx
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=class_name,
                alpha=0.7,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax.set_title('PCA Visualization of Iris Dataset', fontsize=14, pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_decision_tree(
        self,
        model: DecisionTreeClassifier,
        feature_names: List[str] = None,
        target_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a decision tree.

        Args:
            model: Trained decision tree model
            feature_names: Names of features
            target_names: Names of target classes
            save_path: Optional path to save figure
        """
        logger.info("Plotting decision tree...")

        feature_names = feature_names or FEATURE_NAMES
        target_names = target_names or TARGET_NAMES

        fig, ax = plt.subplots(figsize=(20, 10))

        plot_tree(
            model,
            feature_names=feature_names,
            class_names=target_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        ax.set_title('Decision Tree Visualization', fontsize=16, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        model,
        feature_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance (for tree-based models).

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Names of features
            save_path: Optional path to save figure
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute")
            return

        logger.info("Plotting feature importance...")

        feature_names = feature_names or FEATURE_NAMES

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(
            range(len(importances)),
            importances[indices],
            color='skyblue',
            edgecolor='navy'
        )

        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def plot_class_distribution(
        self,
        target: pd.Series,
        target_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of target classes.

        Args:
            target: Target labels
            target_names: Names of target classes
            save_path: Optional path to save figure
        """
        logger.info("Plotting class distribution...")

        target_names = target_names or TARGET_NAMES

        fig, ax = plt.subplots(figsize=(8, 6))

        counts = target.value_counts().sort_index()
        labels = [target_names[i] for i in counts.index]

        ax.bar(labels, counts.values, color='skyblue', edgecolor='navy')

        ax.set_xlabel('Species', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution in Iris Dataset', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (label, count) in enumerate(zip(labels, counts.values)):
            ax.text(i, count + 1, str(count), ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()
