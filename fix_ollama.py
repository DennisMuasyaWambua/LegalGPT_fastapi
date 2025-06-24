#!/usr/bin/env python3
import requests
import logging
import sys
import os
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_ollama():
    # Get Ollama host from environment variable or use default
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    logger.info(f"Checking Ollama at: {ollama_host}")
    
    try:
        # Check if Ollama is available
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            logger.info(f"Ollama is available. Models: {model_names}")
            
            # Check specific model details
            for model_name in model_names:
                try:
                    model_info_response = requests.post(
                        f"{ollama_host}/api/show", 
                        json={"name": model_name},
                        timeout=5
                    )
                    
                    if model_info_response.status_code == 200:
                        model_info = model_info_response.json()
                        logger.info(f"Model {model_name} details:")
                        logger.info(f"  - Parameters: {json.dumps(model_info.get('parameters', {}), indent=2)}")
                        logger.info(f"  - Template: {model_info.get('template', 'None')}")
                    else:
                        logger.warning(f"Failed to get model info for {model_name}: {model_info_response.status_code}")
                except Exception as e:
                    logger.error(f"Error getting model info for {model_name}: {str(e)}")
            
            # Try pulling llama3 model again if there are issues
            logger.info("Trying to pull llama3 model to ensure it's available...")
            try:
                # Send pull request
                pull_response = requests.post(
                    f"{ollama_host}/api/pull",
                    json={"name": "llama3"},
                    timeout=60  # Allow more time for pull
                )
                
                if pull_response.status_code == 200:
                    logger.info("Successfully pulled llama3 model")
                else:
                    logger.warning(f"Failed to pull llama3 model: {pull_response.status_code}")
                    logger.warning(f"Response: {pull_response.text}")
            except Exception as e:
                logger.error(f"Error pulling llama3 model: {str(e)}")
                
            return True
        else:
            logger.error(f"Ollama responded with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    result = diagnose_ollama()
    if result:
        print("Ollama diagnosis completed!")
        sys.exit(0)
    else:
        print("Ollama diagnosis failed!")
        sys.exit(1)