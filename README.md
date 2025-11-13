# Iris Flower Classification

[![CI](https://github.com/pyenthusiasts/Iris-Flower-Classification/workflows/CI/badge.svg)](https://github.com/pyenthusiasts/Iris-Flower-Classification/actions)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready machine learning package for classifying iris flowers using multiple algorithms with detailed analysis and visualization capabilities.

## Features

- **Multiple ML Algorithms**: 8 different classification algorithms including Decision Trees, Random Forest, SVM, KNN, and Neural Networks
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Data Visualization**: Rich visualizations for EDA, model comparison, and result analysis
- **Command-Line Interface**: Easy-to-use CLI for training, evaluation, and prediction
- **Python API**: Clean, well-documented API for programmatic access
- **Interactive Notebooks**: Jupyter notebooks for exploratory analysis
- **Extensive Testing**: Comprehensive test suite with pytest
- **CI/CD Pipeline**: Automated testing and quality checks with GitHub Actions

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Project Structure](#project-structure)
- [Available Models](#available-models)
- [Examples](#examples)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Using pip (recommended)

```bash
pip install -r requirements.txt
pip install -e .
```

### From source

```bash
git clone https://github.com/pyenthusiasts/Iris-Flower-Classification.git
cd Iris-Flower-Classification
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Quick Start

### Using the Main Script

```bash
python main.py
```

This will run a complete analysis including:
- Loading the Iris dataset
- Training all available models
- Comparing model performance
- Detailed evaluation of the best model
- Sample predictions

### Using the CLI

```bash
# Train a specific model
iris-classifier train --model random_forest --save

# Compare all models
iris-classifier compare --plot

# Make a prediction
iris-classifier predict 5.0 3.6 1.4 0.2 --model random_forest

# Display dataset information
iris-classifier info --stats
```

### Using the Python API

```python
from iris_classifier import IrisDataLoader, ModelFactory, ModelEvaluator

# Load data
loader = IrisDataLoader()
X_train, X_test, y_train, y_test = loader.get_train_test_split()

# Train a model
model = ModelFactory.create_model('random_forest')
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
evaluator.print_evaluation_report(results)
```

## Usage

### Command-Line Interface

The package includes a comprehensive CLI with the following commands:

#### Train a Model

```bash
iris-classifier train [OPTIONS]

Options:
  --model TEXT            Model to train (default: decision_tree)
  --test-size FLOAT       Test set size (default: 0.3)
  --scale                 Scale features using StandardScaler
  --save                  Save trained model
```

#### Compare Models

```bash
iris-classifier compare [OPTIONS]

Options:
  --test-size FLOAT       Test set size (default: 0.3)
  --scale                 Scale features
  --cv INTEGER            Number of CV folds (default: 5)
  --plot                  Show comparison plot
```

#### Make Predictions

```bash
iris-classifier predict SEPAL_LENGTH SEPAL_WIDTH PETAL_LENGTH PETAL_WIDTH [OPTIONS]

Options:
  --model TEXT            Model to use (default: decision_tree)
  --model-file TEXT       Load model from file
  --scale                 Scale features
```

#### Visualize Data

```bash
iris-classifier visualize [OPTIONS]

Options:
  --type TEXT             Visualization type:
                         all, distribution, pairplot, correlation, pca, classes
```

#### Dataset Information

```bash
iris-classifier info [OPTIONS]

Options:
  --stats                Show detailed feature statistics
```

### Python API

#### Data Loading

```python
from iris_classifier import IrisDataLoader

# Basic usage
loader = IrisDataLoader()
X, y = loader.get_full_dataset()

# With feature scaling
loader = IrisDataLoader(scale=True)
X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.3)

# Get dataset information
info = loader.get_dataset_info()
stats = loader.get_feature_statistics()
```

#### Model Training

```python
from iris_classifier.models import ModelFactory, ModelTrainer

# Create a model
model = ModelFactory.create_model('random_forest')

# With custom parameters
model = ModelFactory.create_model('decision_tree', {'max_depth': 5})

# Train with ModelTrainer
trainer = ModelTrainer(model, 'random_forest')
trainer.train(X_train, y_train)

# Make predictions
predictions = trainer.predict(X_test)
probabilities = trainer.predict_proba(X_test)

# Save model
trainer.save('my_model.pkl')
```

#### Model Evaluation

```python
from iris_classifier import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate a single model
results = evaluator.evaluate_model(model, X_test, y_test, 'random_forest')
evaluator.print_evaluation_report(results)

# Cross-validation
cv_results = evaluator.cross_validate_model(model, X_train, y_train, cv=5)

# Compare multiple models
models = ModelFactory.get_all_models()
comparison_df = evaluator.compare_models(models, X_train, y_train, X_test, y_test)

# Get best model
best_model_name = evaluator.get_best_model(comparison_df)
```

#### Visualization

```python
from iris_classifier import IrisVisualizer

visualizer = IrisVisualizer()

# Data visualizations
visualizer.plot_feature_distributions(X, y)
visualizer.plot_pairplot(X, y)
visualizer.plot_correlation_matrix(X)
visualizer.plot_pca_visualization(X, y)
visualizer.plot_class_distribution(y)

# Model visualizations
visualizer.plot_confusion_matrix(y_test, predictions, model_name='Random Forest')
visualizer.plot_model_comparison(comparison_df)
visualizer.plot_feature_importance(model)
```

### Jupyter Notebooks

The `notebooks/` directory contains interactive notebooks:

1. **01_exploratory_data_analysis.ipynb**: Comprehensive EDA of the Iris dataset
2. **02_model_training_and_comparison.ipynb**: Training and comparing multiple models

To run notebooks:

```bash
jupyter notebook notebooks/
```

## Project Structure

```
Iris-Flower-Classification/
├── src/
│   └── iris_classifier/
│       ├── __init__.py           # Package initialization
│       ├── config.py              # Configuration settings
│       ├── data_loader.py         # Data loading and preprocessing
│       ├── models.py              # ML model factory and trainer
│       ├── evaluator.py           # Model evaluation and comparison
│       ├── visualizer.py          # Visualization tools
│       ├── utils.py               # Utility functions
│       └── cli.py                 # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── test_data_loader.py       # Data loader tests
│   ├── test_models.py             # Model tests
│   ├── test_evaluator.py         # Evaluator tests
│   └── test_utils.py              # Utility tests
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training_and_comparison.ipynb
├── data/                          # Data directory
│   └── README.md
├── models/                        # Saved models directory
│   └── README.md
├── docs/                          # Documentation
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── main.py                        # Main script
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── pyproject.toml                 # Build configuration
├── README.md                      # This file
├── CONTRIBUTING.md                # Contribution guidelines
├── CODE_OF_CONDUCT.md            # Code of conduct
├── LICENSE                        # MIT license
└── .gitignore                     # Git ignore rules
```

## Available Models

The package supports 8 different classification algorithms:

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `decision_tree` | Decision Tree Classifier | max_depth, min_samples_split |
| `random_forest` | Random Forest Classifier | n_estimators, max_depth |
| `svm` | Support Vector Machine | kernel, C, gamma |
| `knn` | K-Nearest Neighbors | n_neighbors, weights |
| `logistic_regression` | Logistic Regression | C, solver |
| `naive_bayes` | Gaussian Naive Bayes | - |
| `gradient_boosting` | Gradient Boosting | n_estimators, learning_rate |
| `mlp` | Multi-Layer Perceptron | hidden_layer_sizes, max_iter |

## Examples

### Example 1: Quick Model Training

```python
from iris_classifier import IrisDataLoader, ModelFactory, ModelEvaluator

# Load and split data
loader = IrisDataLoader()
X_train, X_test, y_train, y_test = loader.get_train_test_split()

# Train model
model = ModelFactory.create_model('random_forest')
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Example 2: Model Comparison

```python
from iris_classifier import IrisDataLoader, ModelFactory, ModelEvaluator, IrisVisualizer

# Setup
loader = IrisDataLoader()
X_train, X_test, y_train, y_test = loader.get_train_test_split()

# Compare all models
models = ModelFactory.get_all_models()
evaluator = ModelEvaluator()
comparison = evaluator.compare_models(models, X_train, y_train, X_test, y_test)

# Visualize results
visualizer = IrisVisualizer()
visualizer.plot_model_comparison(comparison)

# Get best model
best = evaluator.get_best_model(comparison)
print(f"Best model: {best}")
```

### Example 3: Making Predictions

```python
from iris_classifier import IrisDataLoader, ModelFactory

# Load data and train model
loader = IrisDataLoader()
X_train, X_test, y_train, y_test = loader.get_train_test_split()
model = ModelFactory.create_model('random_forest')
model.fit(X_train, y_train)

# Prepare new sample
sample = loader.predict_sample(5.0, 3.6, 1.4, 0.2)

# Predict
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print(f"Predicted species: {loader.target_names[prediction]}")
for name, prob in zip(loader.target_names, probabilities):
    print(f"  {name}: {prob:.2%}")
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/pyenthusiasts/Iris-Flower-Classification.git
cd Iris-Flower-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]
```

### Code Quality

```bash
# Format code
black src/iris_classifier

# Lint code
flake8 src/iris_classifier --max-line-length=100

# Type checking
mypy src/iris_classifier
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=iris_classifier --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestModelFactory::test_create_decision_tree
```

### Test Coverage

The project maintains >80% test coverage across all modules.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Code of conduct
- Development setup
- Coding standards
- Testing guidelines
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- Scikit-learn for machine learning algorithms
- The open-source community for various tools and libraries

## Citation

If you use this package in your research, please cite:

```bibtex
@software{iris_classifier,
  author = {Your Name},
  title = {Iris Flower Classification},
  year = {2024},
  url = {https://github.com/pyenthusiasts/Iris-Flower-Classification}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/pyenthusiasts/Iris-Flower-Classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pyenthusiasts/Iris-Flower-Classification/discussions)

---

**Happy Classifying!** 🌸
