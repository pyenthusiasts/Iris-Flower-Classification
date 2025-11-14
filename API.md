# API Documentation

Complete reference for the Iris Flower Classification REST API.

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

Currently, the API is open. For production deployments, configure API key authentication:

```bash
# Set in .env
API_KEY_ENABLED=true
API_KEY=your-secure-api-key
```

Include in requests:
```bash
curl -H "X-API-Key: your-secure-api-key" http://localhost:8000/predict
```

## Endpoints

### GET /

Root endpoint with API information.

**Response:**
```json
{
  "name": "Iris Flower Classification API",
  "version": "2.0.0",
  "documentation": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "models_loaded": 3,
  "uptime_seconds": 3600.5
}
```

### GET /metrics

Prometheus metrics endpoint.

Returns metrics in Prometheus format for monitoring.

### GET /models

List all available models.

**Response:**
```json
[
  {
    "name": "random_forest",
    "description": "Random Forest Classifier",
    "loaded": true,
    "parameters": {
      "description": "Random Forest Classifier",
      "accuracy": 0.9778,
      "loaded_at": "2024-01-15T10:00:00Z",
      "train_samples": 105,
      "test_samples": 45
    }
  },
  ...
]
```

### GET /models/{model_name}

Get information about a specific model.

**Parameters:**
- `model_name` (path): Name of the model

**Response:**
```json
{
  "name": "random_forest",
  "description": "Random Forest Classifier",
  "loaded": true,
  "parameters": {
    "accuracy": 0.9778,
    "loaded_at": "2024-01-15T10:00:00Z"
  }
}
```

**Error Response (404):**
```json
{
  "detail": "Model random_forest_v2 not found"
}
```

### POST /models/{model_name}/load

Load a model into memory.

**Parameters:**
- `model_name` (path): Name of the model to load

**Response:**
```json
{
  "message": "Model random_forest loaded successfully",
  "metadata": {
    "description": "Random Forest Classifier",
    "accuracy": 0.9778,
    "loaded_at": "2024-01-15T10:30:00Z",
    "train_samples": 105,
    "test_samples": 45
  }
}
```

### POST /predict

Make a prediction for a single iris sample.

**Request Body:**
```json
{
  "sample": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "model_name": "random_forest",
  "include_probabilities": true
}
```

**Field Validations:**
- `sepal_length`: 0.0 to 10.0 cm (must be positive)
- `sepal_width`: 0.0 to 10.0 cm (must be positive)
- `petal_length`: 0.0 to 10.0 cm (must be positive)
- `petal_width`: 0.0 to 10.0 cm (must be positive)
- `model_name`: One of the available models (default: "random_forest")
- `include_probabilities`: boolean (default: true)

**Response:**
```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_name": "random_forest",
  "timestamp": "2024-01-15T10:30:00Z",
  "duration_ms": 2.5
}
```

**cURL Example:**
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

**Python Example:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "sample": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },
    "model_name": "random_forest",
    "include_probabilities": True
}

response = requests.post(url, json=data)
print(response.json())
```

### POST /predict/batch

Make predictions for multiple samples.

**Request Body:**
```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.7,
      "sepal_width": 3.0,
      "petal_length": 5.2,
      "petal_width": 2.3
    }
  ],
  "model_name": "random_forest",
  "include_probabilities": true
}
```

**Constraints:**
- Minimum samples: 1
- Maximum samples: 1000

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "setosa",
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      },
      "model_name": "random_forest",
      "timestamp": "2024-01-15T10:30:00Z",
      "duration_ms": 1.2
    },
    {
      "prediction": "virginica",
      "probabilities": {
        "setosa": 0.0,
        "versicolor": 0.1,
        "virginica": 0.9
      },
      "model_name": "random_forest",
      "timestamp": "2024-01-15T10:30:00Z",
      "duration_ms": 1.1
    }
  ],
  "total_samples": 2,
  "total_duration_ms": 5.3
}
```

## Error Responses

### 400 Bad Request

Invalid input parameters or validation errors.

```json
{
  "error": "Invalid Input",
  "detail": "All measurements must be positive values",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 404 Not Found

Resource not found (e.g., model not found).

```json
{
  "error": "Model Not Found",
  "detail": "Model invalid_model not loaded",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 500 Internal Server Error

Server error during processing.

```json
{
  "detail": "Prediction failed: <error message>"
}
```

## Rate Limiting

API implements rate limiting to prevent abuse:

- **Default**: 100 requests per minute per IP
- **Configurable** via `RATE_LIMIT_CALLS` and `RATE_LIMIT_PERIOD`

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1234567890
```

When rate limited (429 status):
```json
{
  "error": "Too Many Requests",
  "detail": "Rate limit exceeded. Please retry after 60 seconds"
}
```

## Response Headers

All responses include:
- `X-Process-Time`: Request processing time in seconds
- `Content-Type`: `application/json`

## Interactive Documentation

Visit these URLs when the API is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Performance

### Expected Latency

- Single prediction: 1-5ms
- Batch prediction (100 samples): 50-100ms
- Model loading: 100-500ms

### Throughput

With default configuration (4 workers):
- ~200-500 requests/second for single predictions
- ~50-100 requests/second for batch predictions (10 samples each)

## Best Practices

### 1. Use Batch Predictions

For multiple samples, use `/predict/batch` instead of multiple `/predict` calls:

**❌ Inefficient:**
```python
for sample in samples:
    response = requests.post(f"{url}/predict", json={"sample": sample})
```

**✅ Efficient:**
```python
response = requests.post(
    f"{url}/predict/batch",
    json={"samples": samples}
)
```

### 2. Cache Model Loading

Load models once and reuse:

```python
# Load model once
requests.post(f"{url}/models/random_forest/load")

# Make multiple predictions
for sample in samples:
    response = requests.post(f"{url}/predict", json={
        "sample": sample,
        "model_name": "random_forest"
    })
```

### 3. Handle Errors Gracefully

```python
try:
    response = requests.post(url, json=data, timeout=5)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### 4. Use Connection Pooling

For high-throughput applications:

```python
import requests

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20
)
session.mount('http://', adapter)

# Make requests with session
response = session.post(url, json=data)
```

## SDKs and Examples

### Python SDK

```python
import requests
from typing import Dict, List

class IrisClassifierClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def predict(self, sample: Dict, model_name: str = "random_forest") -> Dict:
        """Make a single prediction."""
        response = self.session.post(
            f"{self.base_url}/predict",
            json={
                "sample": sample,
                "model_name": model_name,
                "include_probabilities": True
            }
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, samples: List[Dict], model_name: str = "random_forest") -> Dict:
        """Make batch predictions."""
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={
                "samples": samples,
                "model_name": model_name,
                "include_probabilities": True
            }
        )
        response.raise_for_status()
        return response.json()

    def get_models(self) -> List[Dict]:
        """Get available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

# Usage
client = IrisClassifierClient()

sample = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

result = client.predict(sample)
print(f"Prediction: {result['prediction']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const baseURL = 'http://localhost:8000';

async function predict(sample, modelName = 'random_forest') {
  try {
    const response = await axios.post(`${baseURL}/predict`, {
      sample: sample,
      model_name: modelName,
      include_probabilities: true
    });
    return response.data;
  } catch (error) {
    console.error('Prediction error:', error.response.data);
    throw error;
  }
}

// Usage
const sample = {
  sepal_length: 5.1,
  sepal_width: 3.5,
  petal_length: 1.4,
  petal_width: 0.2
};

predict(sample).then(result => {
  console.log('Prediction:', result.prediction);
  console.log('Probabilities:', result.probabilities);
});
```

## Monitoring and Metrics

### Prometheus Queries

```promql
# Request rate
rate(iris_predictions_total[5m])

# Average prediction latency
rate(iris_prediction_duration_seconds_sum[5m]) / rate(iris_prediction_duration_seconds_count[5m])

# Error rate
rate(iris_errors_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(iris_prediction_duration_seconds_bucket[5m]))
```

### Health Check Monitoring

```bash
# Simple uptime check
curl -f http://localhost:8000/health || echo "API is down"

# Detailed health check
curl -s http://localhost:8000/health | jq '.status'
```

## Troubleshooting

### Common Issues

**Issue**: "Model not loaded" error
**Solution**: Load the model first using `/models/{model_name}/load`

**Issue**: Slow predictions
**Solution**: Use batch predictions, check model complexity, ensure adequate resources

**Issue**: Connection timeout
**Solution**: Increase timeout, check network, verify API is running

**Issue**: High error rate
**Solution**: Check logs, verify input validation, review resource usage

## Support

- **Interactive Docs**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/pyenthusiasts/Iris-Flower-Classification/issues
- **Email**: support@example.com
