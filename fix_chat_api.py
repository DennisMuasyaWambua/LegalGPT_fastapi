#!/usr/bin/env python3
import requests
import json
import time
import sys

def fix_chat_api():
    """Try to generate a response using a simplified prompt format"""
    # Get Ollama host from environment variable or use default
    ollama_host = "http://localhost:11434"
    print(f"Connecting to Ollama at {ollama_host}")
    
    # Test model availability
    try:
        models_response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if models_response.status_code == 200:
            models = [model.get("name") for model in models_response.json().get("models", [])]
            print(f"Available models: {models}")
        else:
            print(f"Error getting models: {models_response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return False
    
    # Simplified prompt format for LLaMA3
    prompt = """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant answering a question about Kenya Law.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What are the requirements for starting a business in Kenya?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    # Try generation with simplified prompt
    try:
        # Prepare request payload
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40
            }
        }
        
        print(f"Sending test generation request with simplified prompt...")
        start_time = time.time()
        
        # Make request to Ollama API
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        end_time = time.time()
        print(f"Request completed in {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response received successfully!")
            print(f"First 100 chars of response: {result['response'][:100]}...")
            return True
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Ollama chat with simplified prompt...")
    success = fix_chat_api()
    if success:
        print("Test successful! The model is working properly.")
        
        # Now create a script to fix the API's prompt template
        with open("/home/dennis/Desktop/projects/LegalGPT_fastapi/api_prompt_fix.py", "w") as f:
            f.write('''#!/usr/bin/env python3
import os

# Path to the law.py file
law_py_path = "/home/dennis/Desktop/projects/LegalGPT_fastapi/law.py"

# Load the content
with open(law_py_path, "r") as f:
    content = f.read()

# Find the prompt template for Llama3 and update it
if '<|im_start|>system' in content:
    # Replace the old prompt format with the new one that works with llama3
    content = content.replace('<|im_start|>system', '<|start_header_id|>system<|end_header_id|>')
    content = content.replace('<|im_end|>', '<|eot_id|>')
    content = content.replace('<|im_start|>user', '<|start_header_id|>user<|end_header_id|>')
    content = content.replace('<|im_start|>assistant', '<|start_header_id|>assistant<|end_header_id|>')
    
    # Write the updated content back
    with open(law_py_path, "w") as f:
        f.write(content)
    
    print("Successfully updated the prompt template in law.py")
else:
    print("Could not find the prompt template section in law.py")
''')
        
        # Make the script executable
        import os
        os.chmod("/home/dennis/Desktop/projects/LegalGPT_fastapi/api_prompt_fix.py", 0o755)
        
        print("\nCreated a script to fix the API's prompt template.")
        print("Run the following commands to fix the API and try again:")
        print("1. python3 /home/dennis/Desktop/projects/LegalGPT_fastapi/api_prompt_fix.py")
        print("2. python3 /home/dennis/Desktop/projects/LegalGPT_fastapi/start_api.sh")
        sys.exit(0)
    else:
        print("Test failed! There are still issues with the Ollama integration.")
        sys.exit(1)