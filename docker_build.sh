#!/bin/bash
# Docker build script for LegalGPT FastAPI

echo "=== Building LegalGPT Docker Container ==="
echo "Using tinyllama as the default model"

# Build the Docker image
docker-compose build

echo "=== Build Complete ==="
echo "To start the container, run: docker-compose up -d"
echo "To view logs, run: docker-compose logs -f"
echo "To test the API, run: ./docker_test.sh"
