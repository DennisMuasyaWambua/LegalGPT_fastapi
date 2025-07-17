FROM python:3.11-slim

WORKDIR /app

# Install system dependencies with retry logic
RUN for i in 1 2 3; do \
        echo "Attempt $i: Updating package lists..." && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        curl \
        ca-certificates \
        supervisor \
        wget \
        && rm -rf /var/lib/apt/lists/* \
        && break || \
        (echo "Attempt $i failed, retrying in 10 seconds..." && sleep 10); \
    done

# Install Ollama with Railway optimization
RUN echo "Installing Ollama..." && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    ollama --version

# Copy requirements first for better caching
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with comprehensive retry logic
RUN pip install --upgrade pip && \
    echo "Installing Python dependencies..." && \
    for i in 1 2 3 4 5; do \
        echo "Attempt $i: Installing Python dependencies..." && \
        pip --no-cache-dir install \
        --retries 10 \
        --timeout 600 \
        --index-url https://pypi.org/simple/ \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        -r requirements.txt && \
        echo "Python dependencies installed successfully" && break || \
        (echo "Attempt $i failed, retrying in 15 seconds..." && sleep 15); \
    done

# Copy application code
COPY . .

# Set up Supervisord to manage both services
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/vector_db /app/static

# Set environment variables with Railway optimization
ENV OLLAMA_HOST=http://localhost:11434
ENV DEFAULT_MODEL=tinyllama
ENV VECTOR_DB_PATH=/app/vector_db
ENV HTTP_TIMEOUT=30
ENV OLLAMA_TIMEOUT=300
ENV OLLAMA_NUM_THREADS=1
ENV OLLAMA_NUM_GPU=0
ENV OLLAMA_KEEP_ALIVE=5m
ENV OLLAMA_MAX_LOADED_MODELS=1

# Expose ports for both Ollama and the API
EXPOSE 8000 11434

# Start services using supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]