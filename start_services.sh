#!/bin/bash

echo "Starting Object Detection API..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to API directory
cd detection_api

# Build and start the service
echo "Building Docker image..."
docker-compose up --build -d

# Wait for service to be ready
echo "Waiting for API to start..."
sleep 10

# Check if API is healthy
echo "Checking API health..."
curl -s http://localhost:8000/health | jq .

echo ""
echo "API is running at http://localhost:8000"
echo "Interactive docs at http://localhost:8000/docs"
echo ""
echo "To stop the service, run:"
echo "cd detection_api && docker-compose down"
