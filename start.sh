#!/bin/bash
# Activate virtual environment if it exists
if [ -d "/opt/venv" ]; then
    source /opt/venv/bin/activate
fi

# Start the application
exec uvicorn api:app --host 0.0.0.0 --port $PORT