"""
Load testing for Iris Classification API using Locust.

Run with: locust -f tests/load_test.py --host=http://localhost:8000
"""

import random
from locust import HttpUser, task, between


class IrisAPIUser(HttpUser):
    """Simulated user for load testing the Iris Classification API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a simulated user starts."""
        self.sample_data = self._generate_sample()

    def _generate_sample(self):
        """Generate random iris sample data."""
        return {
            "sample": {
                "sepal_length": random.uniform(4.0, 8.0),
                "sepal_width": random.uniform(2.0, 4.5),
                "petal_length": random.uniform(1.0, 7.0),
                "petal_width": random.uniform(0.1, 2.5)
            },
            "model_name": random.choice([
                "decision_tree",
                "random_forest",
                "svm",
                "knn",
                "logistic_regression"
            ]),
            "include_probabilities": True
        }

    @task(10)
    def predict(self):
        """Test single prediction endpoint."""
        self.client.post(
            "/predict",
            json=self._generate_sample(),
            name="/predict"
        )

    @task(5)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        batch_size = random.randint(5, 20)
        samples = [self._generate_sample()["sample"] for _ in range(batch_size)]

        self.client.post(
            "/predict/batch",
            json={
                "samples": samples,
                "model_name": "random_forest",
                "include_probabilities": True
            },
            name="/predict/batch"
        )

    @task(3)
    def list_models(self):
        """Test list models endpoint."""
        self.client.get("/models", name="/models")

    @task(2)
    def get_model_info(self):
        """Test get model info endpoint."""
        model_name = random.choice([
            "decision_tree",
            "random_forest",
            "svm"
        ])
        self.client.get(f"/models/{model_name}", name="/models/{model_name}")

    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health", name="/health")
