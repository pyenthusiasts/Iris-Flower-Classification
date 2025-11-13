"""
Iris Flower Classification Package

A comprehensive machine learning package for classifying iris flowers
using multiple algorithms and providing detailed analysis and visualization.
"""

__version__ = "2.0.0"
__author__ = "Your Name"

from .data_loader import IrisDataLoader
from .models import ModelFactory
from .evaluator import ModelEvaluator
from .visualizer import IrisVisualizer

__all__ = [
    "IrisDataLoader",
    "ModelFactory",
    "ModelEvaluator",
    "IrisVisualizer",
]
