#!/bin/bash
cd C:/Users/timot/Desktop/Muasya/LegalGPT_fastapi

# Install required dependencies
echo "Installing required dependencies..."
pip3 install beautifulsoup4 sentence-transformers chromadb python-docx PyPDF2 pandas openpyxl requests uvicorn fastapi

# Set environment variables
export OLLAMA_HOST=http://localhost:11434
export VECTOR_DB_PATH=./vector_db

# Create vector_db directory if it doesn't exist
mkdir -p ./vector_db

# Start the API with debug output
echo "Starting API with debug output..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --log-level debug