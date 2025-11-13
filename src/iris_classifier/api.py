"""
FastAPI REST API for Iris Classification.

This module provides a production-ready REST API for serving
iris classification models with health checks, monitoring,
and comprehensive documentation.
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .data_loader import IrisDataLoader
from .models import ModelFactory, ModelTrainer
from .evaluator import ModelEvaluator
from .exceptions import (
    InvalidInputError,
    ModelNotFoundError,
    PredictionError,
    ModelNotTrainedError
)
from .config import AVAILABLE_MODELS, TARGET_NAMES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('iris_predictions_total', 'Total number of predictions')
prediction_duration = Histogram('iris_prediction_duration_seconds', 'Prediction duration')
error_counter = Counter('iris_errors_total', 'Total number of errors', ['error_type'])

# API metadata
API_VERSION = "2.0.0"
API_TITLE = "Iris Flower Classification API"
API_DESCRIPTION = """
A production-ready REST API for classifying iris flowers using machine learning.

## Features

* **Multiple ML Models**: Support for 8 different classification algorithms
* **Real-time Predictions**: Fast, reliable predictions with probability scores
* **Batch Processing**: Support for batch predictions
* **Model Management**: Load and switch between different models
* **Health Monitoring**: Health checks and metrics endpoints
* **API Documentation**: Interactive API documentation with examples

## Usage

1. Check API health: `GET /health`
2. Get available models: `GET /models`
3. Make a prediction: `POST /predict`
4. Batch predictions: `POST /predict/batch`
5. Get model info: `GET /models/{model_name}`
"""

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded models
loaded_models: Dict[str, any] = {}
model_metadata: Dict[str, Dict] = {}


# Pydantic models for request/response validation
class IrisSample(BaseModel):
    """Input schema for a single iris sample."""

    sepal_length: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Sepal length in centimeters",
        example=5.1
    )
    sepal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Sepal width in centimeters",
        example=3.5
    )
    petal_length: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Petal length in centimeters",
        example=1.4
    )
    petal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Petal width in centimeters",
        example=0.2
    )

    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('All measurements must be positive')
        return v


class PredictionRequest(BaseModel):
    """Request schema for prediction."""

    sample: IrisSample
    model_name: str = Field(
        default="random_forest",
        description="Name of the model to use for prediction",
        example="random_forest"
    )
    include_probabilities: bool = Field(
        default=True,
        description="Whether to include prediction probabilities"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    samples: List[IrisSample] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of iris samples to classify"
    )
    model_name: str = Field(
        default="random_forest",
        description="Name of the model to use for prediction"
    )
    include_probabilities: bool = Field(default=True)


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    prediction: str = Field(..., description="Predicted iris species")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Prediction probabilities for each class"
    )
    model_name: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    duration_ms: float = Field(..., description="Prediction duration in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: List[PredictionResponse]
    total_samples: int
    total_duration_ms: float


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    version: str
    timestamp: str
    models_loaded: int
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str
    description: str
    loaded: bool
    parameters: Optional[Dict] = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Iris Classification API...")
    logger.info(f"API Version: {API_VERSION}")

    # Load default model
    try:
        await load_model("random_forest")
        logger.info("Default model (random_forest) loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default model: {str(e)}")


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Helper functions
async def load_model(model_name: str):
    """Load a model into memory."""
    if model_name not in AVAILABLE_MODELS:
        raise InvalidInputError(f"Unknown model: {model_name}")

    if model_name in loaded_models:
        logger.info(f"Model {model_name} already loaded")
        return

    try:
        logger.info(f"Loading model: {model_name}")

        # Load data and train model
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test = loader.get_train_test_split()

        # Create and train model
        model = ModelFactory.create_model(model_name)
        model.fit(X_train, y_train)

        # Evaluate model
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test, model_name)

        # Store model and metadata
        loaded_models[model_name] = model
        model_metadata[model_name] = {
            "description": ModelFactory.get_model_description(model_name),
            "accuracy": results["accuracy"],
            "loaded_at": datetime.utcnow().isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        logger.info(f"Model {model_name} loaded successfully (accuracy: {results['accuracy']:.4f})")

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise


def get_model(model_name: str):
    """Get a loaded model."""
    if model_name not in loaded_models:
        raise ModelNotFoundError(f"Model {model_name} not loaded")
    return loaded_models[model_name]


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API including:
    - Service status
    - Version information
    - Number of loaded models
    - Uptime
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(loaded_models),
        uptime_seconds=time.process_time()
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for monitoring.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """
    List all available models.

    Returns information about all supported models including
    their loading status and descriptions.
    """
    models_info = []
    for model_name in AVAILABLE_MODELS:
        info = ModelInfo(
            name=model_name,
            description=ModelFactory.get_model_description(model_name),
            loaded=model_name in loaded_models,
            parameters=model_metadata.get(model_name)
        )
        models_info.append(info)

    return models_info


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Parameters:
    - **model_name**: Name of the model

    Returns detailed information about the model including
    performance metrics if the model is loaded.
    """
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found"
        )

    return ModelInfo(
        name=model_name,
        description=ModelFactory.get_model_description(model_name),
        loaded=model_name in loaded_models,
        parameters=model_metadata.get(model_name)
    )


@app.post("/models/{model_name}/load", tags=["Models"])
async def load_model_endpoint(model_name: str):
    """
    Load a model into memory.

    Parameters:
    - **model_name**: Name of the model to load

    This endpoint loads and trains a model, making it available
    for predictions.
    """
    try:
        await load_model(model_name)
        return {
            "message": f"Model {model_name} loaded successfully",
            "metadata": model_metadata.get(model_name)
        }
    except InvalidInputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        error_counter.labels(error_type="model_load_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for a single iris sample.

    Parameters:
    - **sample**: Iris flower measurements (sepal and petal dimensions)
    - **model_name**: Name of the model to use (default: random_forest)
    - **include_probabilities**: Whether to include prediction probabilities

    Returns the predicted iris species with optional probability scores.

    Example:
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
    """
    start_time = time.time()

    try:
        # Load model if not already loaded
        if request.model_name not in loaded_models:
            await load_model(request.model_name)

        model = get_model(request.model_name)

        # Prepare data
        loader = IrisDataLoader()
        sample_df = loader.predict_sample(
            request.sample.sepal_length,
            request.sample.sepal_width,
            request.sample.petal_length,
            request.sample.petal_width
        )

        # Make prediction
        prediction_idx = model.predict(sample_df)[0]
        prediction_name = TARGET_NAMES[prediction_idx]

        # Get probabilities if requested
        probabilities = None
        if request.include_probabilities and hasattr(model, 'predict_proba'):
            probs = model.predict_proba(sample_df)[0]
            probabilities = {
                name: float(prob)
                for name, prob in zip(TARGET_NAMES, probs)
            }

        duration_ms = (time.time() - start_time) * 1000

        # Update metrics
        prediction_counter.inc()
        prediction_duration.observe(time.time() - start_time)

        return PredictionResponse(
            prediction=prediction_name,
            probabilities=probabilities,
            model_name=request.model_name,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration_ms
        )

    except ModelNotFoundError as e:
        error_counter.labels(error_type="model_not_found").inc()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        error_counter.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple iris samples.

    Parameters:
    - **samples**: List of iris flower measurements
    - **model_name**: Name of the model to use
    - **include_probabilities**: Whether to include prediction probabilities

    Returns predictions for all samples with timing information.

    Maximum batch size: 1000 samples
    """
    start_time = time.time()

    try:
        # Load model if not already loaded
        if request.model_name not in loaded_models:
            await load_model(request.model_name)

        model = get_model(request.model_name)
        loader = IrisDataLoader()

        predictions = []

        for sample in request.samples:
            sample_start = time.time()

            # Prepare data
            sample_df = loader.predict_sample(
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            )

            # Make prediction
            prediction_idx = model.predict(sample_df)[0]
            prediction_name = TARGET_NAMES[prediction_idx]

            # Get probabilities if requested
            probabilities = None
            if request.include_probabilities and hasattr(model, 'predict_proba'):
                probs = model.predict_proba(sample_df)[0]
                probabilities = {
                    name: float(prob)
                    for name, prob in zip(TARGET_NAMES, probs)
                }

            sample_duration = (time.time() - sample_start) * 1000

            predictions.append(PredictionResponse(
                prediction=prediction_name,
                probabilities=probabilities,
                model_name=request.model_name,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=sample_duration
            ))

            # Update metrics
            prediction_counter.inc()

        total_duration = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(request.samples),
            total_duration_ms=total_duration
        )

    except Exception as e:
        error_counter.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Exception handlers
@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    """Handle invalid input errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid Input",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    """Handle model not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Model Not Found",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "iris_classifier.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_api(reload=True)
