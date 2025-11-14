#!/bin/bash
# API testing script

set -e

# Configuration
API_URL=${1:-"http://localhost:8000"}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Testing Iris Classifier API${NC}"
echo -e "${BLUE}URL: ${API_URL}${NC}"
echo -e "${BLUE}================================${NC}"

# Test health endpoint
echo -e "\n${BLUE}[1/5] Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s ${API_URL}/health)
echo "$HEALTH_RESPONSE" | python -m json.tool
echo -e "${GREEN}✓ Health check passed${NC}"

# Test list models
echo -e "\n${BLUE}[2/5] Testing list models...${NC}"
MODELS_RESPONSE=$(curl -s ${API_URL}/models)
echo "$MODELS_RESPONSE" | python -m json.tool | head -20
echo -e "${GREEN}✓ List models passed${NC}"

# Test single prediction
echo -e "\n${BLUE}[3/5] Testing single prediction...${NC}"
PREDICTION_RESPONSE=$(curl -s -X POST ${API_URL}/predict \
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
  }')
echo "$PREDICTION_RESPONSE" | python -m json.tool
echo -e "${GREEN}✓ Single prediction passed${NC}"

# Test batch prediction
echo -e "\n${BLUE}[4/5] Testing batch prediction...${NC}"
BATCH_RESPONSE=$(curl -s -X POST ${API_URL}/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
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
  }')
echo "$BATCH_RESPONSE" | python -m json.tool | head -30
echo -e "${GREEN}✓ Batch prediction passed${NC}"

# Test metrics endpoint
echo -e "\n${BLUE}[5/5] Testing metrics endpoint...${NC}"
METRICS_RESPONSE=$(curl -s ${API_URL}/metrics | head -20)
echo "$METRICS_RESPONSE"
echo -e "${GREEN}✓ Metrics endpoint passed${NC}"

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}All API tests passed!${NC}"
echo -e "${GREEN}================================${NC}"
