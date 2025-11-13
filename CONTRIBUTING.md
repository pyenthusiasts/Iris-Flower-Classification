# Contributing to Iris Flower Classification

Thank you for your interest in contributing to the Iris Flower Classification project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Run tests
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.7 or higher
- pip
- virtualenv (recommended)

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Iris-Flower-Classification.git
cd Iris-Flower-Classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Install the package in editable mode
pip install -e .
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues reported in the issue tracker
- **New features**: Add new machine learning algorithms, visualizations, or functionality
- **Documentation**: Improve README, docstrings, or add tutorials
- **Tests**: Add or improve test coverage
- **Examples**: Add Jupyter notebooks demonstrating usage
- **Performance improvements**: Optimize existing code

### Contribution Workflow

1. **Check existing issues**: Look for existing issues or create a new one to discuss your idea
2. **Create a branch**: Create a descriptive branch name (e.g., `feature/add-neural-network`, `fix/data-loader-bug`)
3. **Make changes**: Implement your changes following our coding standards
4. **Write tests**: Add tests for new functionality
5. **Update documentation**: Update docstrings and README if necessary
6. **Run tests**: Ensure all tests pass
7. **Submit PR**: Create a pull request with a clear description

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and under 50 lines when possible
- Add docstrings to all public functions, classes, and modules

### Code Formatting

We use **Black** for code formatting:

```bash
black src/iris_classifier
```

### Linting

We use **flake8** for linting:

```bash
flake8 src/iris_classifier --max-line-length=100
```

### Type Hints

We encourage the use of type hints for function signatures:

```python
def train_model(X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """Train a machine learning model."""
    pass
```

### Documentation

- Add docstrings to all public classes, methods, and functions
- Follow Google-style docstring format:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Short description of the function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
    """
    pass
```

## Testing

### Running Tests

Run all tests:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=iris_classifier --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_data_loader.py
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:

```python
def test_data_loader_initialization():
    """Test that data loader initializes correctly."""
    # Arrange
    loader = IrisDataLoader()

    # Act & Assert
    assert loader.data is not None
    assert len(loader.data) == 150
```

## Pull Request Process

1. **Update documentation**: Ensure README and docstrings are up to date
2. **Add tests**: Include tests for new functionality
3. **Run all tests**: Ensure all tests pass locally
4. **Update CHANGELOG**: Add your changes to CHANGELOG.md (if it exists)
5. **Create PR**: Submit a pull request with a clear title and description
6. **Link issues**: Reference any related issues (e.g., "Fixes #123")
7. **Request review**: Wait for maintainer review
8. **Address feedback**: Make requested changes
9. **Merge**: Once approved, your PR will be merged

### PR Title Format

Use conventional commit format:

- `feat: Add new model algorithm`
- `fix: Correct data loading bug`
- `docs: Update README with new examples`
- `test: Add tests for evaluator module`
- `refactor: Simplify model factory`
- `style: Format code with black`
- `chore: Update dependencies`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

## Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Python version, package versions
6. **Screenshots**: If applicable
7. **Additional context**: Any other relevant information

## Suggesting Enhancements

When suggesting enhancements:

1. **Use case**: Describe the use case for the enhancement
2. **Proposed solution**: Explain your proposed solution
3. **Alternatives**: Describe alternative solutions you've considered
4. **Additional context**: Add any other context or screenshots

## Questions?

If you have questions, feel free to:

- Open an issue with the "question" label
- Contact the maintainers
- Check existing documentation and issues

## Recognition

Contributors will be recognized in:

- GitHub contributors page
- Project documentation
- Release notes

Thank you for contributing to Iris Flower Classification!
