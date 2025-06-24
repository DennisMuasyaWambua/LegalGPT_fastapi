#!/usr/bin/env python3
import requests
import logging
import time
import asyncio
import os
import json
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ollama_connection():
    # Get Ollama host from environment variable or use default
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    logger.info(f"Checking Ollama availability at {ollama_host}")
    
    try:
        # Check if Ollama is available with multiple retries
        max_retries = 3
        retry_delay = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f"Attempt {retry_count+1}/{max_retries} to check Ollama...")
                response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = [model.get("name") for model in response.json().get("models", [])]
                    logger.info(f"Ollama connected successfully. Available models: {models}")
                    
                    # Try to run a simple generation
                    logger.info("Testing Ollama generation...")
                    model = models[0] if models else "llama3"
                    
                    prompt = "What is the capital of France?"
                    
                    # Prepare request payload
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.95,
                            "top_k": 40
                        }
                    }
                    
                    logger.info(f"Sending test generation request to Ollama with model: {model}")
                    
                    # Get timeout from environment variable or use default
                    ollama_timeout = int(os.environ.get("OLLAMA_TIMEOUT", 300))
                    logger.info(f"Using Ollama timeout of {ollama_timeout} seconds")
                    
                    # Configure session with timeouts and retry strategy
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        max_retries=requests.adapters.Retry(
                            total=3,
                            backoff_factor=1,
                            status_forcelist=[408, 429, 500, 502, 503, 504],
                            allowed_methods=["POST"]
                        )
                    )
                    session.mount('http://', adapter)
                    session.mount('https://', adapter)
                    
                    # Make request to Ollama API with dynamically set timeout
                    start_time = time.time()
                    try:
                        response = session.post(
                            f"{ollama_host}/api/generate",
                            json=payload,
                            timeout=(10, ollama_timeout)  # (connect_timeout, read_timeout)
                        )
                        
                        end_time = time.time()
                        logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"Ollama generation successful!")
                            logger.info(f"Response: {result['response'][:100]}...")
                            return True
                        else:
                            logger.error(f"Ollama generation failed with status code: {response.status_code}")
                            logger.error(f"Response: {response.text}")
                            return False
                    except requests.exceptions.Timeout:
                        logger.error(f"Ollama generation timed out after {ollama_timeout} seconds")
                        return False
                    except Exception as e:
                        logger.error(f"Error during Ollama generation: {str(e)}")
                        return False
                    
                else:
                    logger.warning(f"Ollama responded with status code {response.status_code}. Retrying...")
                    retry_count += 1
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logger.warning(f"Ollama connection error: {str(e)}. Retrying...")
                retry_count += 1
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        if retry_count >= max_retries:
            logger.error(f"Could not connect to Ollama after {max_retries} attempts.")
            return False
            
    except Exception as e:
        logger.error(f"Error during Ollama connection check: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_ollama_connection())
    if result:
        print("Ollama connectivity test passed!")
        sys.exit(0)
    else:
        print("Ollama connectivity test failed!")
        sys.exit(1)