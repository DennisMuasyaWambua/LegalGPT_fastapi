#!/usr/bin/env python3
import os
import sys
import json
import requests
import time

# Set working directory
os.chdir("/home/dennis/Desktop/projects/LegalGPT_fastapi")

print("=== Comprehensive API Fix ===")
print("This script will fix the issues with the Legal GPT FastAPI application")

# Step 1: Test Ollama connection and find working models
print("\nStep 1: Testing Ollama connection and finding working models...")
ollama_host = "http://localhost:11434"

try:
    # Check if Ollama is running
    response = requests.get(f"{ollama_host}/api/tags", timeout=5)
    if response.status_code != 200:
        print(f"Error: Ollama is not running or not accessible at {ollama_host}")
        sys.exit(1)
        
    # Get available models
    available_models = [model.get("name").split(":")[0] for model in response.json().get("models", [])]
    print(f"Available models: {available_models}")
    
    # Test each model with a simple prompt
    working_models = []
    for model in available_models:
        print(f"Testing model: {model}")
        try:
            # Skip embedding models
            if "embed" in model.lower():
                print(f"Skipping embedding model: {model}")
                continue
                
            response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": "What is 2+2?",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Model {model} works!")
                working_models.append(model)
            else:
                print(f"❌ Model {model} failed: {response.text}")
        except Exception as e:
            print(f"❌ Error testing {model}: {str(e)}")
    
    if not working_models:
        print("No working models found. Let's pull a lightweight model...")
        
        # Try to pull tinyllama
        try:
            print("Pulling tinyllama model (this may take a few minutes)...")
            response = requests.post(
                f"{ollama_host}/api/pull",
                json={"name": "tinyllama"},
                timeout=300
            )
            
            if response.status_code == 200:
                print("✅ Successfully pulled tinyllama model")
                working_models = ["tinyllama"]
            else:
                print(f"❌ Failed to pull tinyllama: {response.text}")
        except Exception as e:
            print(f"❌ Error pulling tinyllama: {str(e)}")
    
    if not working_models:
        print("Failed to find or create any working models.")
        print("Please allocate more memory to your environment or pull a smaller model manually.")
        sys.exit(1)
        
    # Select the best working model
    # Prefer smaller models like tinyllama, then llama3.2, then others
    preferred_order = ["tinyllama", "llama3.2", "llama3", "llama2", "codellama"]
    selected_model = None
    
    for model in preferred_order:
        if model in working_models:
            selected_model = model
            break
    
    if not selected_model:
        selected_model = working_models[0]
        
    print(f"\nSelected model for API: {selected_model}")
    
except Exception as e:
    print(f"Error connecting to Ollama: {str(e)}")
    sys.exit(1)

# Step 2: Fix API.py to use the working model
print("\nStep 2: Updating API.py to use the working model...")

try:
    with open("api.py", "r") as f:
        api_content = f.read()
        
    # Update the default model in ChatRequest
    if "model_name: str = Field(" in api_content:
        api_content = api_content.replace(
            'model_name: str = Field("llama3-optimized"',
            f'model_name: str = Field("{selected_model}"'
        )
        
        with open("api.py", "w") as f:
            f.write(api_content)
            
        print(f"✅ Updated api.py to use {selected_model} as the default model")
    else:
        print("❌ Could not find model_name field in ChatRequest class")
except Exception as e:
    print(f"❌ Error updating api.py: {str(e)}")

# Step 3: Update law.py to handle memory errors gracefully
print("\nStep 3: Updating law.py to handle memory errors gracefully...")

try:
    with open("law.py", "r") as f:
        law_content = f.read()
        
    # Update the prompt format for the selected model
    if '<|im_start|>system' in law_content:
        # For llama3.2 we need to use the appropriate prompt format
        if selected_model == "llama3.2":
            law_content = law_content.replace('<|im_start|>system', '<|start_header_id|>system<|end_header_id|>')
            law_content = law_content.replace('<|im_end|>', '<|eot_id|>')
            law_content = law_content.replace('<|im_start|>user', '<|start_header_id|>user<|end_header_id|>')
            law_content = law_content.replace('<|im_start|>assistant', '<|start_header_id|>assistant<|end_header_id|>')
            print("✅ Updated prompt format for llama3.2")
            
    # Add code to handle memory errors
    if "model requires more system memory" not in law_content:
        memory_error_handler = """
                if response.status_code != 200:
                    # Check for memory error
                    if response.status_code == 500 and "memory" in response.text.lower():
                        logger.error(f"Memory error with model {model_name}. Returning context-only response.")
                        return f"I found relevant information about your query, but don't have enough memory to generate a full response. Here's the relevant context:\\n\\n{context_text[:1500]}..."
                    logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                    return f"Error accessing Ollama (HTTP {response.status_code}). Using context-only response:\\n\\n{context_text[:1500]}..."
        """
        
        # Find where to insert the error handler
        error_check_pos = law_content.find("if response.status_code == 200:")
        if error_check_pos > 0:
            # Find the next line after this check
            next_line_pos = law_content.find("\n", error_check_pos)
            if next_line_pos > 0:
                # Insert our error handler before the if statement
                law_content = law_content[:error_check_pos] + memory_error_handler + law_content[error_check_pos:]
                print("✅ Added memory error handling to law.py")
            else:
                print("❌ Could not find position to insert memory error handler")
        else:
            print("❌ Could not find position to insert memory error handler")
            
    # Write updated content
    with open("law.py", "w") as f:
        f.write(law_content)
        
    print("✅ Updated law.py successfully")
except Exception as e:
    print(f"❌ Error updating law.py: {str(e)}")

# Step 4: Update Docker configuration
print("\nStep 4: Updating Docker configuration...")

try:
    # Update Dockerfile
    with open("Dockerfile", "r") as f:
        dockerfile_content = f.read()
        
    # Update default model in Dockerfile
    if "ENV OLLAMA_HOST" in dockerfile_content and f"ENV DEFAULT_MODEL={selected_model}" not in dockerfile_content:
        # Find the environment variables section
        env_pos = dockerfile_content.find("ENV OLLAMA_HOST")
        if env_pos > 0:
            # Add our new environment variable
            env_section_end = dockerfile_content.find("\n\n", env_pos)
            if env_section_end < 0:
                env_section_end = len(dockerfile_content)
                
            # Insert the new environment variable
            new_env = f"ENV DEFAULT_MODEL={selected_model}\n"
            dockerfile_content = dockerfile_content[:env_section_end] + new_env + dockerfile_content[env_section_end:]
            
            with open("Dockerfile", "w") as f:
                f.write(dockerfile_content)
                
            print(f"✅ Added DEFAULT_MODEL={selected_model} to Dockerfile")
        else:
            print("❌ Could not find ENV section in Dockerfile")
    else:
        print("✅ No changes needed for Dockerfile")
        
    # Update docker-compose.yml
    if os.path.exists("docker-compose.yml"):
        with open("docker-compose.yml", "r") as f:
            compose_content = f.read()
            
        # Update environment variables
        if "environment:" in compose_content and f"DEFAULT_MODEL={selected_model}" not in compose_content:
            env_pos = compose_content.find("environment:")
            if env_pos > 0:
                # Find the end of the environment section
                env_section_end = compose_content.find("\n\n", env_pos)
                if env_section_end < 0:
                    # Find the next section
                    env_section_end = compose_content.find("\n    ", env_pos + 20)
                    
                if env_section_end > 0:
                    # Insert the new environment variable
                    new_env = f"      - DEFAULT_MODEL={selected_model}\n"
                    compose_content = compose_content[:env_section_end] + new_env + compose_content[env_section_end:]
                    
                    with open("docker-compose.yml", "w") as f:
                        f.write(compose_content)
                        
                    print(f"✅ Added DEFAULT_MODEL={selected_model} to docker-compose.yml")
                else:
                    print("❌ Could not find environment section end in docker-compose.yml")
            else:
                print("❌ Could not find environment section in docker-compose.yml")
        else:
            print("✅ No changes needed for docker-compose.yml")
    else:
        print("⚠️ docker-compose.yml not found")
        
except Exception as e:
    print(f"❌ Error updating Docker configuration: {str(e)}")

# Step 5: Create a start script with dependencies
print("\nStep 5: Creating improved start script...")

start_script = f"""#!/bin/bash
cd "$(dirname "$0")"

# Install required dependencies
echo "Installing required dependencies..."
pip3 install -r requirements.txt || pip3 install beautifulsoup4 sentence-transformers chromadb python-docx PyPDF2 pandas openpyxl requests uvicorn fastapi

# Set environment variables
export OLLAMA_HOST=http://localhost:11434
export VECTOR_DB_PATH=./vector_db
export DEFAULT_MODEL={selected_model}

# Create vector_db directory if it doesn't exist
mkdir -p ./vector_db

# Make sure model is available
echo "Ensuring {selected_model} model is available..."
curl -s "http://localhost:11434/api/tags" | grep -q "{selected_model}" || {{
    echo "Pulling {selected_model} model..."
    curl -X POST "http://localhost:11434/api/pull" -d '{{"name": "{selected_model}"}}'
}}

# Start the API
echo "Starting API..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --log-level info
"""

with open("start_api.sh", "w") as f:
    f.write(start_script)
    
os.chmod("start_api.sh", 0o755)
print("✅ Created improved start_api.sh script")

# Step 6: Update entrypoint.sh to use the working model
print("\nStep 6: Updating entrypoint.sh...")

try:
    with open("entrypoint.sh", "r") as f:
        entrypoint_content = f.read()
        
    # Replace llama3 with our selected model
    if "llama3" in entrypoint_content:
        entrypoint_content = entrypoint_content.replace("llama3", f"{selected_model}")
        entrypoint_content = entrypoint_content.replace("llama3-optimized", f"{selected_model}-optimized")
        
        with open("entrypoint.sh", "w") as f:
            f.write(entrypoint_content)
            
        print(f"✅ Updated entrypoint.sh to use {selected_model}")
    else:
        print("⚠️ No changes needed for entrypoint.sh")
except Exception as e:
    print(f"❌ Error updating entrypoint.sh: {str(e)}")

# Step 7: Create a test script
print("\nStep 7: Creating test script...")

test_script = """#!/usr/bin/env python3
import requests
import json
import time
import sys

def test_api():
    \"\"\"Test the API endpoints\"\"\"
    base_url = "http://localhost:8000"
    
    # 1. Test API status
    print("Testing API status...")
    try:
        response = requests.get(f"{base_url}/api", timeout=5)
        if response.status_code == 200:
            print(f"✅ API status: {response.json()}")
        else:
            print(f"❌ API status failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ API connection error: {str(e)}")
        print("Make sure the API is running (./start_api.sh)")
        return False
    
    # 2. Test sample questions
    print("\\nTesting sample questions...")
    try:
        response = requests.get(f"{base_url}/sample-questions", timeout=5)
        if response.status_code == 200:
            questions = response.json().get("questions", [])
            print(f"✅ Got {len(questions)} sample questions")
        else:
            print(f"❌ Sample questions failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Sample questions error: {str(e)}")
    
    # 3. Test chat endpoint
    print("\\nTesting chat endpoint (this may take a minute)...")
    try:
        payload = {{
            "query": "What are the requirements for starting a business in Kenya?",
            "model_name": "%s",
            "site_filter": None
        }}
        
        print(f"Sending query to /chat endpoint with model: {payload['model_name']}")
        start_time = time.time()
        
        response = requests.post(f"{base_url}/chat", json=payload, timeout=60)
        
        end_time = time.time()
        print(f"Request took {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat response received successfully!")
            print(f"Sources: {len(result.get('sources', []))}")
            print(f"First part of response: {result.get('response', '')[:100]}...")
            print("\\n🎉 All tests passed! The API is working correctly.")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat endpoint error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Legal GPT API...")
    if not test_api():
        print("\\n❌ API test failed. Please check the logs for details.")
        sys.exit(1)
    else:
        print("\\n✅ API is working correctly!")
        sys.exit(0)
""" % selected_model

with open("test_api.py", "w") as f:
    f.write(test_script)
    
os.chmod("test_api.py", 0o755)
print("✅ Created test_api.py script")

# Final step: Summary and instructions
print("\n=== Fix Complete ===")
print(f"Your Legal GPT API has been updated to use the {selected_model} model, which works with your available memory.")
print("\nTo start the API, run:")
print("./start_api.sh")
print("\nTo test the API, run (in a different terminal):")
print("./test_api.py")
print("\nIf you're using Docker, rebuild and restart your containers:")
print("docker-compose down && docker-compose up --build -d")
print("\nIf you want to add more memory to your system in the future, you can")
print("update api.py to use a larger model like llama3 for better responses.")