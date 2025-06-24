#!/usr/bin/env python3
import requests
import time
import sys
import os

def check_ollama():
    # Get Ollama host from environment variable or use default
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"Checking Ollama at: {ollama_host}")
    
    try:
        # Attempt to connect to Ollama API
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model.get("name") for model in response.json().get("models", [])]
            print(f"Ollama is available. Models: {models}")
            return True
        else:
            print(f"Ollama responded with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    # Try multiple times with exponential backoff
    max_retries = 3
    retry_delay = 1
    
    for i in range(max_retries):
        print(f"Attempt {i+1}/{max_retries}...")
        if check_ollama():
            print("Ollama is working!")
            sys.exit(0)
        
        if i < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
    
    print("Failed to connect to Ollama after multiple attempts")
    sys.exit(1)