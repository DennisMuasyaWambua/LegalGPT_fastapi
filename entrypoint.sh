#!/bin/bash
set -e

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is running!"

# Pull the Llama3 model if it doesn't exist
echo "Checking for llama3 model..."
if ! curl -s http://localhost:11434/api/tags | grep -q '"name":"llama3"'; then
    echo "Pulling llama3 model (this may take a while)..."
    ollama pull llama3
    echo "Model llama3 ready!"
else
    echo "Model llama3 already exists!"
fi

# Start the API
exec python -m uvicorn api:app --host 0.0.0.0 --port 8000