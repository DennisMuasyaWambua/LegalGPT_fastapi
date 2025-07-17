#!/bin/bash
set -e

# Set Ollama resource parameters optimized for Railway
export OLLAMA_NUM_THREADS=1
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_GPU=0
export OLLAMA_HOST=0.0.0.0

echo "Setting OLLAMA_NUM_THREADS=1"
echo "Setting OLLAMA_KEEP_ALIVE=5m"
echo "Setting OLLAMA_MAX_LOADED_MODELS=1"
echo "Setting OLLAMA_NUM_GPU=0"

# Wait for Ollama to start with Railway-optimized health check
echo "Waiting for Ollama to start..."
MAX_RETRIES=120  # 2 minutes timeout for Railway
RETRY_COUNT=0
RETRY_DELAY=1

while ! curl -s --connect-timeout 3 http://localhost:11434/api/tags > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Ollama failed to start after $MAX_RETRIES retries, continuing anyway..."
        break
    fi
    sleep $RETRY_DELAY
    # Print status every 20 seconds
    if [ $((RETRY_COUNT % 20)) -eq 0 ]; then
        echo "Still waiting for Ollama after $RETRY_COUNT seconds..."
    fi
done

if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "Ollama is running!"
    # Verify Ollama health
    OLLAMA_STATUS=$(curl -s --connect-timeout 3 http://localhost:11434/api/tags || echo '{"error":"Failed to connect"}')
    echo "Ollama status: $OLLAMA_STATUS"
    
    # Pull the tinyllama model (smaller and faster for Railway)
    echo "Checking for tinyllama model..."
    if ! curl -s --connect-timeout 3 http://localhost:11434/api/tags | grep -q '"name":"tinyllama"'; then
        echo "Pulling tinyllama model (optimized for Railway)..."
        ollama pull tinyllama
        echo "Model tinyllama ready!"
    else
        echo "Model tinyllama already exists!"
    fi
fi

# Configure Ollama model parameters for Railway
MODEL_NAME=${DEFAULT_MODEL:-tinyllama}
echo "Using model: $MODEL_NAME"

# Create optimized model configuration for Railway
echo "Creating Railway-optimized model configuration..."
ollama create ${MODEL_NAME}-railway << EOF
FROM $MODEL_NAME
PARAMETER num_ctx 2048
PARAMETER num_gpu 0
PARAMETER num_thread 1
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
EOF

echo "Railway-optimized $MODEL_NAME model created"

# Start the API with single worker for Railway
exec python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30