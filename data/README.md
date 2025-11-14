# Data Directory

This directory is used for storing:

- Custom datasets (if any)
- Cached data
- Exported results
- Data preprocessing artifacts

## Iris Dataset

The Iris dataset is automatically loaded from scikit-learn and does not need to be stored here.

### Dataset Information

- **Source**: UCI Machine Learning Repository (via scikit-learn)
- **Samples**: 150 (50 per class)
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Missing Values**: None

### Usage

The dataset is loaded automatically using the `IrisDataLoader` class:

```python
from iris_classifier import IrisDataLoader

loader = IrisDataLoader()
X, y = loader.get_full_dataset()
```

## Custom Data

If you want to use custom data, place your CSV files here and modify the data loader accordingly.
