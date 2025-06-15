FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Added git and cmake, which are essential for llama_cpp_python to build correctly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-railway.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# Added --retries, --timeout, and --no-cache-dir for more robust downloads.
RUN pip install --upgrade pip && \
    pip --no-cache-dir install --retries 5 --timeout 600 -r requirements-railway.txt

# Copy application code
COPY . .

# Expose port for the application
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
