# Deployment Guide

This guide covers various deployment options for the Iris Flower Classification API.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Docker Compose (Recommended)](#docker-compose-recommended)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Production Checklist](#production-checklist)

## Local Development

### Quick Start

```bash
# Install dependencies
make install-dev

# Run the API
make api
```

The API will be available at http://localhost:8000

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# Run the API
uvicorn iris_classifier.api:app --reload
```

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
make docker-build

# Run the container
make docker-run
```

Or manually:

```bash
docker build -t iris-classifier:latest .
docker run -d -p 8000:8000 --name iris-api iris-classifier:latest
```

### Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your configuration
```

Run with environment file:

```bash
docker run -d -p 8000:8000 --env-file .env iris-classifier:latest
```

## Docker Compose (Recommended)

Docker Compose provides a complete stack with monitoring.

### Start Services

```bash
make docker-compose-up
```

This starts:
- **API** (http://localhost:8000)
- **Prometheus** (http://localhost:9090)
- **Grafana** (http://localhost:3000) - user: admin, pass: admin

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Restart a service
docker-compose restart api

# Scale the API
docker-compose up -d --scale api=3
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube, GKE, EKS, AKS, etc.)
- kubectl configured
- Container registry (optional)

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace iris-classifier

# Apply all configurations
kubectl apply -f k8s/ -n iris-classifier

# Check deployment status
kubectl get pods -n iris-classifier
kubectl get svc -n iris-classifier
```

### Individual Components

```bash
# Deploy configuration
kubectl apply -f k8s/configmap.yaml -n iris-classifier
kubectl apply -f k8s/secret.yaml -n iris-classifier

# Deploy storage
kubectl apply -f k8s/pvc.yaml -n iris-classifier

# Deploy application
kubectl apply -f k8s/deployment.yaml -n iris-classifier
kubectl apply -f k8s/service.yaml -n iris-classifier

# Deploy autoscaling
kubectl apply -f k8s/hpa.yaml -n iris-classifier

# Deploy ingress (optional)
kubectl apply -f k8s/ingress.yaml -n iris-classifier
```

### Update Deployment

```bash
# Update image
kubectl set image deployment/iris-classifier-api \
  api=iris-classifier:v2.1.0 -n iris-classifier

# Rollout status
kubectl rollout status deployment/iris-classifier-api -n iris-classifier

# Rollback if needed
kubectl rollout undo deployment/iris-classifier-api -n iris-classifier
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment iris-classifier-api --replicas=5 -n iris-classifier

# Check HPA status
kubectl get hpa -n iris-classifier
```

### Monitoring

```bash
# View logs
kubectl logs -f deployment/iris-classifier-api -n iris-classifier

# View metrics
kubectl top pods -n iris-classifier
kubectl top nodes

# Port forward for local access
kubectl port-forward svc/iris-classifier-api 8000:80 -n iris-classifier
```

## Cloud Deployment

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag iris-classifier:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/iris-classifier:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/iris-classifier:latest

# Deploy using ECS CLI or Console
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/iris-classifier

# Deploy to Cloud Run
gcloud run deploy iris-classifier \
  --image gcr.io/PROJECT_ID/iris-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image iris-classifier:latest .

# Deploy to ACI
az container create \
  --resource-group <resource-group> \
  --name iris-classifier \
  --image <registry-name>.azurecr.io/iris-classifier:latest \
  --cpu 1 --memory 1 \
  --registry-login-server <registry-name>.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label iris-classifier \
  --ports 8000
```

## Monitoring Setup

### Prometheus

Prometheus scrapes metrics from `/metrics` endpoint.

Configuration in `monitoring/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'iris-api'
    static_configs:
      - targets: ['api:8000']
```

Access Prometheus at http://localhost:9090

### Grafana

1. Access Grafana at http://localhost:3000
2. Login (admin/admin)
3. Add Prometheus datasource
4. Import dashboard from `monitoring/grafana/dashboards/`

### Key Metrics

- `iris_predictions_total`: Total predictions
- `iris_prediction_duration_seconds`: Prediction latency
- `iris_errors_total`: Error count

## Production Checklist

### Security

- [ ] Change default SECRET_KEY
- [ ] Change default API_KEY
- [ ] Update secrets in k8s/secret.yaml
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS appropriately
- [ ] Enable rate limiting
- [ ] Review security groups/firewall rules
- [ ] Scan for vulnerabilities (`make security`)

### Configuration

- [ ] Set appropriate resource limits
- [ ] Configure log rotation
- [ ] Set up backup strategy for models
- [ ] Configure persistent storage
- [ ] Set environment-specific variables
- [ ] Review autoscaling thresholds

### Monitoring

- [ ] Set up alerting rules
- [ ] Configure log aggregation
- [ ] Set up uptime monitoring
- [ ] Configure dashboards
- [ ] Test health checks
- [ ] Set up error tracking

### Testing

- [ ] Run load tests (`make load-test`)
- [ ] Run integration tests
- [ ] Test rollback procedures
- [ ] Verify health checks
- [ ] Test autoscaling

### Documentation

- [ ] Update API documentation
- [ ] Document deployment process
- [ ] Document rollback procedures
- [ ] Create runbooks for common issues
- [ ] Document monitoring setup

## Troubleshooting

### API Not Responding

```bash
# Check container logs
docker logs iris-api

# Check Kubernetes pods
kubectl logs deployment/iris-classifier-api -n iris-classifier

# Check health endpoint
curl http://localhost:8000/health
```

### High Memory Usage

- Adjust `MODEL_CACHE_SIZE` in configuration
- Limit number of loaded models
- Increase container memory limits
- Review model complexity

### Slow Predictions

- Check model complexity
- Enable caching
- Use faster models for production
- Scale horizontally
- Review resource allocation

### Connection Issues

- Check firewall rules
- Verify service/ingress configuration
- Check DNS resolution
- Verify SSL/TLS certificates
- Review CORS configuration

## Support

For issues and questions:
- GitHub Issues: https://github.com/pyenthusiasts/Iris-Flower-Classification/issues
- Documentation: See README.md
