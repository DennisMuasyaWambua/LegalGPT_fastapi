#!/usr/bin/env python3
import requests
import sys
import os

def find_working_model():
    """Find a model that works with the available memory"""
    ollama_host = "http://localhost:11434"
    print(f"Checking Ollama models at {ollama_host}")
    
    try:
        # Get available models
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model.get("name").split(':')[0] for model in response.json().get("models", [])]
            print(f"Available models: {models}")
            
            # Try each model with a simple prompt to see which one works
            working_models = []
            
            # Simple prompt for testing
            prompt = "What is 2+2?"
            
            for model in models:
                print(f"Testing model: {model}")
                try:
                    # Try to generate with this model
                    response = requests.post(
                        f"{ollama_host}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 10  # Keep it short for testing
                            }
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        print(f"Model {model} works!")
                        working_models.append(model)
                    else:
                        print(f"Model {model} failed with status {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"Error testing model {model}: {str(e)}")
            
            return working_models
        else:
            print(f"Error getting models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return []

def apply_fix():
    """Apply a simple fix to use the working model"""
    working_models = find_working_model()
    
    if not working_models:
        print("No working models found.")
        return False
        
    print(f"Found working models: {working_models}")
    model_to_use = working_models[0]
    
    # Create a simple fix to update the model_name default in api.py
    api_file = "/home/dennis/Desktop/projects/LegalGPT_fastapi/api.py"
    
    try:
        with open(api_file, "r") as f:
            content = f.read()
            
        # Find the model_name parameter in ChatRequest class
        if "model_name: str = Field(" in content:
            # Replace the default model
            content = content.replace(
                'model_name: str = Field("llama3-optimized"', 
                f'model_name: str = Field("{model_to_use}"'
            )
            
            with open(api_file, "w") as f:
                f.write(content)
                
            print(f"Updated api.py to use {model_to_use} as the default model")
        else:
            print("Could not find the model_name field in ChatRequest class")
        
        return True
    except Exception as e:
        print(f"Error updating api.py: {str(e)}")
        return False

if __name__ == "__main__":
    print("Finding and applying fix for memory issues...")
    if apply_fix():
        print("\nAPI has been updated to use a model that works with available memory.")
        print("Run the following command to start the API:")
        print("python3 /home/dennis/Desktop/projects/LegalGPT_fastapi/start_api.sh")
        sys.exit(0)
    else:
        print("\nFailed to update the API.")
        sys.exit(1)