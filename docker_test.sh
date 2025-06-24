#!/bin/bash
# Docker test script for LegalGPT FastAPI

echo "=== Testing LegalGPT API in Docker ==="

# Check if the container is running
CONTAINER_ID=$(docker-compose ps -q legal-gpt)
if [ -z "$CONTAINER_ID" ]; then
    echo "❌ LegalGPT container is not running"
    echo "Please start it with: docker-compose up -d"
    exit 1
fi

echo "✅ LegalGPT container is running: $CONTAINER_ID"

# Check API status
echo "Testing API status..."
RESPONSE=$(curl -s http://localhost:8000/api)
if [[ $RESPONSE == *"ok"* ]]; then
    echo "✅ API is responding: $RESPONSE"
else
    echo "❌ API status test failed: $RESPONSE"
    echo "Check the logs with: docker-compose logs -f"
    exit 1
fi

# Test sample questions endpoint
echo "Testing sample questions..."
QUESTIONS=$(curl -s http://localhost:8000/sample-questions)
if [[ $QUESTIONS == *"questions"* ]]; then
    echo "✅ Sample questions available"
else
    echo "❌ Sample questions test failed"
fi

# Test chat endpoint (this is the one that was failing)
echo "Testing chat endpoint (this may take a minute)..."
echo "Sending test query to /chat endpoint..."

RESPONSE=$(curl -s -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"query":"What are the requirements for starting a business in Kenya?","model_name":"'${DEFAULT_MODEL:-tinyllama}'","site_filter":null}')

if [[ $RESPONSE == *"response"* ]]; then
    echo "✅ Chat endpoint is working!"
    echo "First part of response: $(echo $RESPONSE | grep -o '"response":"[^"]*' | head -c 100)..."
else
    echo "❌ Chat endpoint test failed"
    echo "Response: $RESPONSE"
    echo "Check the logs with: docker-compose logs -f"
    
    echo "Checking container memory usage..."
    docker stats --no-stream $CONTAINER_ID
    
    echo "Checking container logs for errors..."
    docker-compose logs --tail 50 | grep -i error
fi

echo "=== Test Complete ==="
