# Kenya Law Assistant API

A FastAPI-based service that provides RAG-powered (Retrieval Augmented Generation) chatbot capabilities for Kenyan legal content. The system crawls, indexes, and allows natural language queries against Kenya's legal framework.

## Features

- Retrieval-augmented chatbot for Kenyan legal information
- Crawls and indexes content from kenyalaw.org and new.kenyalaw.org
- Vector-based similarity search for relevant context retrieval
- Integration with open-source LLMs via Ollama

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (optional, for local LLM inference)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LegalGPT_fastapi
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API:
   ```bash
   python api.py
   ```

## API Endpoints

### 1. GET `/`

Serves the static web interface or returns API status if no interface is available.

**Response:**
- Returns the index.html file if available
- Otherwise returns a JSON status message

### 2. GET `/api`

Simple endpoint to check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "message": "Kenya Law Assistant API is running"
}
```

### 3. GET `/sample-questions`

Provides a list of sample questions that can be used with the chatbot.

**Response:**
```json
{
  "questions": [
    "What are the key provisions of the Kenyan Constitution?",
    "What is the process for filing a case in the Kenyan High Court?",
    ...
  ]
}
```

### 4. POST `/chat`

Main endpoint for querying the Kenya Law Assistant chatbot.

**Request Body:**
```json
{
  "query": "What are the requirements for starting a business in Kenya?",
  "site_filter": "kenyalaw.org",  // Optional: "kenyalaw.org" or "new.kenyalaw.org"
  "model_name": "llama3"  // Optional: Default is "llama3"
}
```

**Response:**
```json
{
  "response": "The requirements for starting a business in Kenya include...",
  "sources": [
    {
      "url": "https://kenyalaw.org/path/to/document",
      "title": "Business Registration Act"
    },
    ...
  ],
  "query": "What are the requirements for starting a business in Kenya?"
}
```

### 5. GET `/status`

Check the current status of the API.

**Response:**
```json
{
  "status": "ready",  // "initializing", "crawling", or "ready"
  "message": "Kenya Law Assistant is ready for queries"
}
```

### 6. POST `/crawl`

Start a crawl of the Kenya Law websites.

**Request Body:**
```json
{
  "max_pages": 500,  // Optional: Default is 100
  "max_depth": 5,    // Optional: Default is 3
  "resume": true     // Optional: Default is true
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Started crawling with max_pages=500, max_depth=5"
}
```

## Usage Examples

### Querying the API

```python
import requests

# Chat endpoint
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "query": "What rights are protected under the Bill of Rights in Kenya?",
        "model_name": "llama3"
    }
)
result = response.json()
print(result["response"])

# For source attribution
for source in result["sources"]:
    print(f"Source: {source['title']} - {source['url']}")
```

### Checking API Status

```python
import requests

response = requests.get("http://localhost:8000/status")
print(response.json())
```

### Starting a Web Crawl

```python
import requests

response = requests.post(
    "http://localhost:8000/crawl",
    json={
        "max_pages": 200,
        "max_depth": 4,
        "resume": True
    }
)
print(response.json())
```

## Configuration Options

The API can be configured with the following environment variables:

- `VECTOR_DB_PATH`: Path to store the vector database (default: "./vector_db")
- `CONCURRENT_REQUESTS`: Number of concurrent requests for crawling (default: 4)
- `REQUEST_DELAY`: Delay between requests in seconds (default: 1.0)
- `OLLAMA_HOST`: Host for Ollama API (default: "http://localhost:11434"). Set this to the correct address if your Ollama instance is running elsewhere.

## Running with Docker

This project includes a complete Docker setup with Ollama running in the same container for easy deployment.

### Using Docker Compose (Recommended)

The easiest way to deploy is with Docker Compose:

```bash
# Build and start the container
docker-compose up -d

# To view logs
docker-compose logs -f
```

This will:
1. Build the image with both the FastAPI application and Ollama
2. Start both services managed by supervisord
3. Download and pull the Llama3 model if needed
4. Persist both the vector database and Ollama models with volumes

### Manual Docker Commands

If you prefer to use Docker directly:

```bash
# Build the image
docker build -t legal-gpt-fastapi .

# Run the container with volumes for persistence
docker run -d -p 8000:8000 -p 11434:11434 \
  -v $(pwd)/vector_db:/app/vector_db \
  -v ollama_models:/root/.ollama \
  legal-gpt-fastapi
```

### Using an External Ollama Instance

If you prefer to use Ollama running elsewhere, you can set the OLLAMA_HOST environment variable:

```bash
docker run -p 8000:8000 -e VECTOR_DB_PATH=/app/vector_db -e OLLAMA_HOST=http://your-ollama-host:11434 legal-gpt-fastapi
```

### First-Time Setup

When you start the container for the first time:

1. It will download and install Ollama
2. It will download the Llama3 model (this may take several minutes)
3. The API will start after Ollama is ready

You can check the status with:

```bash
curl http://localhost:8000/status
```

## License

[Specify your license here]

## Acknowledgements

- This project utilizes the Kenya Law website (https://kenyalaw.org) as a data source.
- Built with FastAPI, ChromaDB, and Sentence Transformers.