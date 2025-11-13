#!/bin/bash
# Deployment script for Iris Classification API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-"development"}
NAMESPACE="iris-classifier"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Iris Classifier Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${BLUE}Checking prerequisites...${NC}"

if ! command_exists kubectl; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}Error: docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"

# Build Docker image
echo -e "\n${BLUE}Building Docker image...${NC}"
docker build -t iris-classifier:latest .
echo -e "${GREEN}✓ Docker image built${NC}"

# Deploy to Kubernetes
if [ "$ENVIRONMENT" = "kubernetes" ]; then
    echo -e "\n${BLUE}Deploying to Kubernetes...${NC}"

    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Apply Kubernetes configurations
    echo -e "${BLUE}Applying configurations...${NC}"
    kubectl apply -f k8s/configmap.yaml -n $NAMESPACE
    kubectl apply -f k8s/secret.yaml -n $NAMESPACE
    kubectl apply -f k8s/pvc.yaml -n $NAMESPACE
    kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/service.yaml -n $NAMESPACE
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE

    # Wait for deployment to be ready
    echo -e "${BLUE}Waiting for deployment to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/iris-classifier-api -n $NAMESPACE

    echo -e "${GREEN}✓ Deployment complete${NC}"

    # Get service information
    echo -e "\n${BLUE}Service Information:${NC}"
    kubectl get svc iris-classifier-api -n $NAMESPACE

elif [ "$ENVIRONMENT" = "docker-compose" ]; then
    echo -e "\n${BLUE}Starting with Docker Compose...${NC}"
    docker-compose up -d

    echo -e "${GREEN}✓ Services started${NC}"
    echo -e "\n${BLUE}Access points:${NC}"
    echo -e "  API: http://localhost:8000"
    echo -e "  Docs: http://localhost:8000/docs"
    echo -e "  Prometheus: http://localhost:9090"
    echo -e "  Grafana: http://localhost:3000"

else
    echo -e "\n${BLUE}Starting API locally...${NC}"
    python -m uvicorn iris_classifier.api:app --host 0.0.0.0 --port 8000
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
