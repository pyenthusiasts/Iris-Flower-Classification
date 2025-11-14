# Iris Flower Classification

[![CI](https://github.com/pyenthusiasts/Iris-Flower-Classification/workflows/CI/badge.svg)](https://github.com/pyenthusiasts/Iris-Flower-Classification/actions)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, **production-ready** machine learning package for classifying iris flowers using multiple algorithms with detailed analysis, visualization, and enterprise-grade deployment capabilities.

## Features

### Core ML Capabilities
- **Multiple ML Algorithms**: 8 different classification algorithms including Decision Trees, Random Forest, SVM, KNN, and Neural Networks
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Data Visualization**: Rich visualizations for EDA, model comparison, and result analysis
- **Batch Processing**: Efficient batch prediction support

### Interfaces
- **REST API**: Production-grade FastAPI server with OpenAPI documentation
- **Command-Line Interface**: Full-featured CLI for training, evaluation, and prediction
- **Python SDK**: Clean, well-documented API for programmatic access
- **Interactive Notebooks**: Jupyter notebooks for exploratory analysis

### Production Features
- **Docker Support**: Multi-stage Dockerfile with security best practices
- **Docker Compose**: Complete stack with API, Prometheus, and Grafana
- **Kubernetes Ready**: Full K8s manifests with HPA, ingress, and monitoring
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Health Checks**: Liveness and readiness probes
- **Load Testing**: Locust-based performance testing
- **CI/CD Pipeline**: Automated testing, linting, and security scanning

### Code Quality
- **Extensive Testing**: Comprehensive test suite with >80% coverage
- **Pre-commit Hooks**: Automated code quality checks
- **Type Hints**: Full type annotation support
- **Documentation**: Complete API documentation and deployment guides
- **Security**: Bandit security scanning, dependency checks

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [REST API](#rest-api)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Production Deployment](#production-deployment)
  - [Docker](#docker)
  - [Docker Compose](#docker-compose)
  - [Kubernetes](#kubernetes)
- [Project Structure](#project-structure)
- [Available Models](#available-models)
- [Examples](#examples)
- [Monitoring](#monitoring)
- [Performance](#performance)
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

### Using the REST API

```bash
# Start the API server
make api
# or
uvicorn iris_classifier.api:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

Make predictions via HTTP:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sample": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    "model_name": "random_forest",
    "include_probabilities": true
  }'
```

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

## Production Deployment

### Docker

Build and run with Docker:

```bash
# Build image
make docker-build

# Run container
make docker-run
```

Or manually:

```bash
docker build -t iris-classifier:latest .
docker run -d -p 8000:8000 --name iris-api iris-classifier:latest
```

Access the API at http://localhost:8000

### Docker Compose

Run the complete stack with monitoring:

```bash
make docker-compose-up
```

This starts:
- **API Server** → http://localhost:8000
- **API Documentation** → http://localhost:8000/docs
- **Prometheus** → http://localhost:9090
- **Grafana** → http://localhost:3000 (admin/admin)

### Kubernetes

Deploy to Kubernetes cluster:

```bash
# Apply all configurations
kubectl apply -f k8s/ -n iris-classifier

# Check status
kubectl get pods -n iris-classifier
kubectl get svc -n iris-classifier
```

Features:
- Horizontal Pod Autoscaling (2-10 replicas)
- Health checks and readiness probes
- Resource limits and requests
- Ingress configuration
- ConfigMaps and Secrets management

**Detailed deployment guide:** See [DEPLOYMENT.md](DEPLOYMENT.md)

**API documentation:** See [API.md](API.md)

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

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

```bash
# View metrics
curl http://localhost:8000/metrics
```

Key metrics:
- `iris_predictions_total`: Total number of predictions
- `iris_prediction_duration_seconds`: Prediction latency histogram
- `iris_errors_total`: Error count by type

### Grafana Dashboards

Access Grafana at http://localhost:3000 (when using Docker Compose):

1. Login with admin/admin
2. Prometheus datasource is pre-configured
3. Import dashboards from `monitoring/grafana/dashboards/`

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "version": "2.0.0",
  "models_loaded": 3,
  "uptime_seconds": 3600.5
}
```

## Performance

### Benchmarking

Run performance benchmarks:

```bash
make benchmark
# or
python scripts/benchmark.py
```

Results include:
- Training time per model
- Prediction latency (avg, std, percentiles)
- Memory usage
- Throughput (predictions/second)
- Accuracy metrics

### Load Testing

Run load tests with Locust:

```bash
make load-test
# or
locust -f tests/load_test.py --host=http://localhost:8000
```

Access Locust UI at http://localhost:8089

### Expected Performance

With default configuration (4 workers):
- **Single prediction latency**: 1-5ms
- **Throughput**: 200-500 req/s
- **Batch (100 samples)**: 50-100ms
- **Memory per worker**: ~250MB

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
