#!/bin/bash
set -e

# Set Ollama resource parameters if provided
if [ ! -z "$OLLAMA_NUM_THREADS" ]; then
    export OLLAMA_NUM_THREADS=$OLLAMA_NUM_THREADS
    echo "Setting OLLAMA_NUM_THREADS=$OLLAMA_NUM_THREADS"
fi

# Set keep alive to prevent constant model loading/unloading
export OLLAMA_KEEP_ALIVE=120m
echo "Setting OLLAMA_KEEP_ALIVE=120m"

if [ ! -z "$OLLAMA_NUM_GPU" ]; then
    export OLLAMA_NUM_GPU=$OLLAMA_NUM_GPU
    echo "Setting OLLAMA_NUM_GPU=$OLLAMA_NUM_GPU"
fi

# Wait for Ollama to start with improved health check
echo "Waiting for Ollama to start..."
MAX_RETRIES=300  # 5 minutes timeout
RETRY_COUNT=0
RETRY_DELAY=1

while ! curl -s --connect-timeout 5 http://localhost:11434/api/tags > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Ollama failed to start after $MAX_RETRIES retries, continuing anyway..."
        break
    fi
    sleep $RETRY_DELAY
    # Print status every 30 seconds
    if [ $((RETRY_COUNT % 30)) -eq 0 ]; then
        echo "Still waiting for Ollama after $RETRY_COUNT seconds..."
    fi
done

if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "Ollama is running!"
    # Verify Ollama health
    OLLAMA_STATUS=$(curl -s --connect-timeout 5 http://localhost:11434/api/tags || echo '{"error":"Failed to connect"}')
    echo "Ollama status: $OLLAMA_STATUS"
fi

# Pull the Llama3 model if it doesn't exist
echo "Checking for llama3 model..."
if ! curl -s --connect-timeout 5 http://localhost:11434/api/tags | grep -q '"name":"llama3"'; then
    echo "Pulling llama3 model (this may take a while)..."
    ollama pull llama3
    echo "Model llama3 ready!"
else
    echo "Model llama3 already exists!"
fi

# Configure Ollama model parameters
# Get model name from environment or use default
MODEL_NAME=${DEFAULT_MODEL:-tinyllama}
echo "Using model: $MODEL_NAME"

# Configure optimized model parameters
echo "Configuring $MODEL_NAME model parameters..."
# Set context size and threads
ollama create ${MODEL_NAME}-optimiz
# Monitor memory usage
monitor_memory() {
  while true; do
    free -h
    sleep 60
  done
}
monitor_memory &
MONITOR_PID=$!

# Trap to kill memory monitor on script exit
trap "kill $MONITOR_PID" EXIT

ed -f - << EOF
FROM $MODEL_NAME
PARAMETER num_ctx 4096
PARAMETER num_gpu $OLLAMA_NUM_GPU
PARAMETER num_thread $OLLAMA_NUM_THREADS
EOF

echo "Using optimized $MODEL_NAME model with expanded context window"

# Start the API with increased worker threads and timeout
exec python -m uvicorn api:app --timeout-keep-alive 75 --workers 1 --host 0.0.0.0 --port 8000 --timeout-keep-alive 75 --workers 2