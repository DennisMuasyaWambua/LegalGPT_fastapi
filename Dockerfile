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

# Download and install Ollama with multiple fallback methods
RUN echo "Attempting to install Ollama..." && \
    # Method 1: Try the install script with retries
    for i in 1 2 3; do \
        echo "Method 1 - Attempt $i: Using install script..." && \
        curl -fsSL --connect-timeout 30 --max-time 600 \
        --retry 3 --retry-delay 5 \
        https://ollama.com/install.sh | sh && \
        echo "Ollama installed successfully via script" && exit 0 || \
        (echo "Script attempt $i failed, retrying in 30 seconds..." && sleep 30); \
    done; \
    # Method 2: Manual installation as fallback
    echo "Install script failed, trying manual installation..." && \
    OLLAMA_VERSION="0.1.32" && \
    for i in 1 2 3; do \
        echo "Method 2 - Attempt $i: Manual installation..." && \
        curl -fsSL --connect-timeout 30 --max-time 600 \
        --retry 3 --retry-delay 5 \
        "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64" \
        -o /usr/local/bin/ollama && \
        chmod +x /usr/local/bin/ollama && \
        echo "Ollama installed successfully via manual method" && exit 0 || \
        (echo "Manual attempt $i failed, retrying in 30 seconds..." && sleep 30); \
    done; \
    # Method 3: Use wget as final fallback
    echo "curl failed, trying wget..." && \
    wget --timeout=600 --tries=3 --waitretry=5 \
    "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64" \
    -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama && \
    echo "Ollama installed successfully via wget"

# Verify Ollama installation
RUN ollama --version || echo "Ollama installation verification failed, but continuing..."

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
        --resume-retries \
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