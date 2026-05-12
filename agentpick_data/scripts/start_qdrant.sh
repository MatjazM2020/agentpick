#!/bin/bash
# Start Qdrant Docker container locally

echo "Starting Qdrant on localhost:6333..."
echo ""
echo "If Docker is not installed, install it from: https://www.docker.com/products/docker-desktop"
echo ""

docker run -p 6333:6333 \
    --name qdrant \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant

echo ""
echo "Qdrant is now running on http://localhost:6333"
echo "API documentation: http://localhost:6333/api/swagger/index.html"
echo ""
echo "To stop Qdrant, press Ctrl+C or in another terminal run: docker stop qdrant"
