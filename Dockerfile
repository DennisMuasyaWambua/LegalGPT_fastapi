FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for Ollama installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    curl \
    ca-certificates \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip --no-cache-dir install --retries 5 --timeout 600 -r requirements.txt

# Copy application code
COPY . .

# Set up Supervisord to manage both services
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/vector_db /app/static

# Set environment variables with optimized settings
ENV OLLAMA_HOST=http://localhost:11434
ENV DEFAULT_MODEL=tinyllama
ENV VECTOR_DB_PATH=/app/vector_db
ENV HTTP_TIMEOUT=30
ENV OLLAMA_TIMEOUT=300
ENV OLLAMA_NUM_THREADS=2
ENV OLLAMA_NUM_GPU=0
ENV OLLAMA_KEEP_ALIVE=120m

# Expose ports for both Ollama and the API
EXPOSE 8000 11434

# Start services using supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
