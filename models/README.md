# Models Directory

This directory stores trained machine learning models that have been saved to disk.

## Saving Models

You can save trained models using the `ModelTrainer` class:

```python
from iris_classifier.models import ModelFactory, ModelTrainer

# Create and train model
model = ModelFactory.create_model('random_forest')
trainer = ModelTrainer(model, 'random_forest')
trainer.train(X_train, y_train)

# Save model
model_path = trainer.save()
print(f"Model saved to: {model_path}")
```

Or using the `ModelFactory` directly:

```python
ModelFactory.save_model(model, 'random_forest', 'my_model.pkl')
```

## Loading Models

Load saved models using the `ModelFactory`:

```python
from iris_classifier.models import ModelFactory

# Load model
model = ModelFactory.load_model('random_forest_model.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## Model Files

Saved models are stored as pickle files (`.pkl`) with the naming convention:
- `{model_name}_model.pkl` (default)
- Custom names can be specified when saving

## Note

Model files are ignored by git (see `.gitignore`) to avoid committing large binary files.
