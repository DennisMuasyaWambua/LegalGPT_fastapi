#!/usr/bin/env python3
import os
import sys
import json
import requests
import time

# Set working directory
os.chdir("/home/dennis/Desktop/projects/LegalGPT_fastapi")

print("=== Docker Container Optimization ===")
print("This script will optimize your LegalGPT FastAPI application for Docker deployment")

# Step 1: Identify working models
print("\nStep 1: Identifying optimal models for Docker deployment...")
ollama_host = "http://localhost:11434"

try:
    # Check if Ollama is running locally - if not, we'll choose models based on memory size
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
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
                        print(f"✅ Model {model} works locally: {model}")
                        working_models.append(model)
                    else:
                        print(f"❌ Model {model} failed locally: {response.text}")
                except Exception as e:
                    print(f"❌ Error testing {model}: {str(e)}")
        
            if working_models:
                print(f"Found working models locally: {working_models}")
                # Choose smallest working model (prefer llama3.2, tinyllama, etc.)
                preferred_order = ["tinyllama", "llama3.2", "phi3", "gemma", "llama3", "llama2", "codellama"]
                selected_model = None
                
                for model in preferred_order:
                    if model in working_models:
                        selected_model = model
                        break
                
                if not selected_model:
                    selected_model = working_models[0]
                    
                print(f"Selected model for Docker deployment: {selected_model}")
            else:
                print("No working models found locally, will use memory-efficient defaults")
                selected_model = "tinyllama"  # Conservative default
        else:
            print(f"Ollama API returned status code: {response.status_code}")
            selected_model = "tinyllama"  # Conservative default
    except Exception as e:
        print(f"Could not connect to local Ollama: {str(e)}")
        print("Will use memory-efficient defaults for Docker")
        selected_model = "tinyllama"  # Conservative default

    # If we couldn't test locally, set a reasonable default for Docker
    if not 'selected_model' in locals():
        selected_model = "tinyllama"
        print(f"Selected model for Docker deployment: {selected_model} (conservative default)")
    
except Exception as e:
    print(f"Error during model selection: {str(e)}")
    selected_model = "tinyllama"  # Fallback to most conservative option
    print(f"Falling back to {selected_model} as default model")

# Step 2: Update Dockerfile to use smaller models and optimize memory
print("\nStep 2: Updating Dockerfile for optimized memory usage...")

try:
    with open("Dockerfile", "r") as f:
        dockerfile_content = f.read()
    
    updated_dockerfile = dockerfile_content
    
    # 1. Update the model to use
    if "ENV DEFAULT_MODEL" not in dockerfile_content:
        # Add DEFAULT_MODEL environment variable
        if "ENV OLLAMA_HOST" in dockerfile_content:
            # Find position after OLLAMA_HOST
            ollama_host_pos = dockerfile_content.find("ENV OLLAMA_HOST")
            next_line_pos = dockerfile_content.find("\n", ollama_host_pos)
            if next_line_pos > 0:
                # Insert our new env var
                model_env = f"\nENV DEFAULT_MODEL={selected_model}"
                updated_dockerfile = updated_dockerfile[:next_line_pos] + model_env + updated_dockerfile[next_line_pos:]
                print(f"✅ Added DEFAULT_MODEL={selected_model} to Dockerfile")
    
    # 2. Add memory optimization for Ollama
    if "ENV OLLAMA_NUM_THREADS" in updated_dockerfile:
        # Update threads to be more conservative
        updated_dockerfile = updated_dockerfile.replace("ENV OLLAMA_NUM_THREADS=4", "ENV OLLAMA_NUM_THREADS=2")
        print("✅ Reduced OLLAMA_NUM_THREADS to 2 for better memory efficiency")
    
    # 3. Add shared memory configuration if missing
    if "ENV OLLAMA_KEEP_ALIVE" not in updated_dockerfile:
        # Add keep alive to prevent model from being loaded/unloaded frequently
        if "ENV OLLAMA_NUM_GPU" in updated_dockerfile:
            # Insert after other OLLAMA settings
            gpu_pos = updated_dockerfile.find("ENV OLLAMA_NUM_GPU")
            next_line_pos = updated_dockerfile.find("\n", gpu_pos)
            if next_line_pos > 0:
                keep_alive_env = "\nENV OLLAMA_KEEP_ALIVE=120m"  # Keep model loaded for 2 hours
                updated_dockerfile = updated_dockerfile[:next_line_pos] + keep_alive_env + updated_dockerfile[next_line_pos:]
                print("✅ Added OLLAMA_KEEP_ALIVE=120m to Dockerfile")
    
    # 4. Update CMD to use the entrypoint script with the proper model
    if "ollama serve" in updated_dockerfile and "RUN chmod +x /app/entrypoint.sh" in updated_dockerfile:
        print("✅ Dockerfile already uses entrypoint.sh for startup")
    
    # Write updated Dockerfile
    if updated_dockerfile != dockerfile_content:
        with open("Dockerfile", "w") as f:
            f.write(updated_dockerfile)
        print("✅ Dockerfile updated successfully")
    else:
        print("⚠️ No changes needed to Dockerfile")
        
except Exception as e:
    print(f"❌ Error updating Dockerfile: {str(e)}")

# Step 3: Update docker-compose.yml file
print("\nStep 3: Updating docker-compose.yml...")

try:
    if os.path.exists("docker-compose.yml"):
        with open("docker-compose.yml", "r") as f:
            docker_compose = f.read()
        
        updated_compose = docker_compose
        
        # 1. Add DEFAULT_MODEL environment variable
        if "DEFAULT_MODEL" not in docker_compose and "environment:" in docker_compose:
            env_pos = docker_compose.find("environment:")
            if env_pos > 0:
                # Find last environment variable
                env_section_end = -1
                last_env_var = None
                
                # Look for patterns like "      - KEY=VALUE"
                env_lines = [line for line in docker_compose[env_pos:].split("\n") if line.strip().startswith("- ")]
                if env_lines:
                    last_env_var = env_lines[-1]
                    last_env_pos = docker_compose.rfind(last_env_var, env_pos)
                    if last_env_pos > 0:
                        env_section_end = last_env_pos + len(last_env_var)
                
                if env_section_end > 0:
                    # Add our new environment variable after the last one
                    new_env = f"\n      - DEFAULT_MODEL={selected_model}"
                    updated_compose = updated_compose[:env_section_end] + new_env + updated_compose[env_section_end:]
                    print(f"✅ Added DEFAULT_MODEL={selected_model} to docker-compose.yml")
        
        # 2. Add/update memory limits
        if "mem_limit:" not in docker_compose:
            # Find the service section
            service_pos = docker_compose.find("services:")
            if service_pos > 0:
                # Find the legal-gpt service
                legal_gpt_pos = docker_compose.find("legal-gpt:", service_pos)
                if legal_gpt_pos > 0:
                    # Find end of service definition or next service
                    next_section = docker_compose.find("\n\n", legal_gpt_pos)
                    if next_section < 0:
                        next_section = len(docker_compose)
                    
                    # Add memory limits
                    mem_limits = "\n    deploy:\n      resources:\n        limits:\n          memory: 4G\n        reservations:\n          memory: 2G"
                    updated_compose = updated_compose[:next_section] + mem_limits + updated_compose[next_section:]
                    print("✅ Added memory limits to docker-compose.yml")
        
        # 3. Add shared memory configuration if using SHM
        if "shm_size:" not in docker_compose:
            # Find the legal-gpt service
            legal_gpt_pos = docker_compose.find("legal-gpt:")
            if legal_gpt_pos > 0:
                # Find ports or volumes section to insert before
                ports_pos = docker_compose.find("ports:", legal_gpt_pos)
                volumes_pos = docker_compose.find("volumes:", legal_gpt_pos)
                insert_pos = min(p for p in [ports_pos, volumes_pos] if p > 0)
                
                if insert_pos > 0:
                    # Add shm_size
                    shm_config = "    shm_size: 1gb\n"
                    updated_compose = updated_compose[:insert_pos] + shm_config + updated_compose[insert_pos:]
                    print("✅ Added shared memory configuration to docker-compose.yml")
        
        # 4. Reduce OLLAMA_NUM_THREADS if present
        if "OLLAMA_NUM_THREADS" in docker_compose:
            updated_compose = updated_compose.replace("OLLAMA_NUM_THREADS=4", "OLLAMA_NUM_THREADS=2")
            print("✅ Reduced OLLAMA_NUM_THREADS to 2 in docker-compose.yml")
        
        # 5. Add restart policy if not present
        if "restart:" not in docker_compose:
            # Find the legal-gpt service
            legal_gpt_pos = docker_compose.find("legal-gpt:")
            if legal_gpt_pos > 0:
                # Find next line after service name
                next_line_pos = docker_compose.find("\n", legal_gpt_pos)
                if next_line_pos > 0:
                    # Add restart policy
                    restart_policy = "    restart: unless-stopped\n"
                    updated_compose = updated_compose[:next_line_pos+1] + restart_policy + updated_compose[next_line_pos+1:]
                    print("✅ Added restart policy to docker-compose.yml")
        
        # Write updated docker-compose.yml
        if updated_compose != docker_compose:
            with open("docker-compose.yml", "w") as f:
                f.write(updated_compose)
            print("✅ docker-compose.yml updated successfully")
        else:
            print("⚠️ No changes needed to docker-compose.yml")
    else:
        print("⚠️ docker-compose.yml not found, skipping updates")
        
except Exception as e:
    print(f"❌ Error updating docker-compose.yml: {str(e)}")

# Step 4: Update entrypoint.sh script
print("\nStep 4: Updating entrypoint.sh...")

try:
    with open("entrypoint.sh", "r") as f:
        entrypoint_content = f.read()
    
    updated_entrypoint = entrypoint_content
    
    # 1. Use DEFAULT_MODEL environment variable
    if "ollama create llama3-optimized" in entrypoint_content and "${DEFAULT_MODEL:-" not in entrypoint_content:
        # Replace hardcoded model with environment variable
        model_creation = """# Get model name from environment or use default
MODEL_NAME=${DEFAULT_MODEL:-tinyllama}
echo "Using model: $MODEL_NAME"

# Configure optimized model parameters
echo "Configuring $MODEL_NAME model parameters..."
# Set context size and threads
ollama create ${MODEL_NAME}-optimized -f - << EOF
FROM $MODEL_NAME
PARAMETER num_ctx 4096
PARAMETER num_gpu $OLLAMA_NUM_GPU
PARAMETER num_thread $OLLAMA_NUM_THREADS
EOF

echo "Using optimized $MODEL_NAME model with expanded context window"
"""
        # Find the model creation section
        create_pos = entrypoint_content.find("echo \"Configuring Llama3 model parameters...\"")
        if create_pos > 0:
            # Find the end of this section
            section_end = entrypoint_content.find("echo \"Using optimized Llama3 model with expanded context window\"")
            if section_end > 0:
                # Find the end of this line
                line_end = entrypoint_content.find("\n", section_end)
                if line_end > 0:
                    # Replace the section
                    updated_entrypoint = entrypoint_content[:create_pos] + model_creation + entrypoint_content[line_end+1:]
                    print("✅ Updated entrypoint.sh to use dynamic model selection")
    
    # 2. Add memory optimization for Ollama startup
    if "OLLAMA_KEEP_ALIVE" not in entrypoint_content and "# Set Ollama resource parameters if provided" in entrypoint_content:
        # Find where to insert keep alive setting
        params_pos = entrypoint_content.find("# Set Ollama resource parameters if provided")
        if params_pos > 0:
            # Find end of this section
            fi_pos = entrypoint_content.find("fi\n", params_pos)
            if fi_pos > 0:
                # Insert keep alive setting after the last fi
                keep_alive_setting = "\n# Set keep alive to prevent constant model loading/unloading\nexport OLLAMA_KEEP_ALIVE=120m\necho \"Setting OLLAMA_KEEP_ALIVE=120m\"\n"
                updated_entrypoint = updated_entrypoint[:fi_pos+3] + keep_alive_setting + updated_entrypoint[fi_pos+3:]
                print("✅ Added OLLAMA_KEEP_ALIVE setting to entrypoint.sh")
    
    # 3. Add memory management helper
    if "# Monitor memory usage" not in entrypoint_content:
        # Add memory monitoring at the end but before starting API
        api_start_pos = entrypoint_content.find("# Start the API")
        if api_start_pos > 0:
            memory_monitor = """
# Monitor memory usage
monitor_memory() {
  while true; do
    free -h
    sleep 60
  done
}
monitor_memory &
MONITOR_PID=$!

# Trap to kill memory monitor on script exit
trap "kill $MONITOR_PID" EXIT

"""
            updated_entrypoint = updated_entrypoint[:api_start_pos] + memory_monitor + updated_entrypoint[api_start_pos:]
            print("✅ Added memory monitoring to entrypoint.sh")
    
    # 4. Use optimized model name in API startup
    if "exec python -m uvicorn api:app" in updated_entrypoint:
        # Update the API command to use the optimized model name
        updated_entrypoint = updated_entrypoint.replace(
            "exec python -m uvicorn api:app",
            "exec python -m uvicorn api:app --timeout-keep-alive 75 --workers 1"
        )
        print("✅ Updated API startup command with optimized settings")
    
    # Write updated entrypoint.sh
    if updated_entrypoint != entrypoint_content:
        with open("entrypoint.sh", "w") as f:
            f.write(updated_entrypoint)
        print("✅ entrypoint.sh updated successfully")
    else:
        print("⚠️ No changes needed to entrypoint.sh")
        
except Exception as e:
    print(f"❌ Error updating entrypoint.sh: {str(e)}")

# Step 5: Update API code to handle memory errors gracefully
print("\nStep 5: Updating API code for Docker deployment...")

try:
    # Update api.py to use the selected model
    with open("api.py", "r") as f:
        api_content = f.read()
    
    updated_api = api_content
    
    # 1. Use DEFAULT_MODEL environment variable
    if "model_name: str = Field(" in api_content:
        # Update to use environment variable with fallback
        env_var_code = """
    # Get default model from environment or fallback to a conservative option
    import os
    default_model = os.environ.get("DEFAULT_MODEL", "tinyllama")
"""
        # Find position to insert code (before ChatRequest class)
        chat_request_pos = api_content.find("class ChatRequest(BaseModel):")
        if chat_request_pos > 0:
            # Find appropriate position before the class
            import_end_pos = api_content.rfind("\n\n", 0, chat_request_pos)
            if import_end_pos > 0:
                # Insert environment variable code
                updated_api = api_content[:import_end_pos+2] + env_var_code + api_content[import_end_pos+2:]
                print("✅ Added DEFAULT_MODEL environment variable support to api.py")
        
        # Update the model_name field to use the environment variable
        updated_api = updated_api.replace(
            'model_name: str = Field("llama3-optimized"',
            'model_name: str = Field(default_model'
        )
        print("✅ Updated api.py to use dynamic model selection")
    
    # 2. Add memory error handling
    if "timeout_keep_alive" not in updated_api and "def run_api(" in updated_api:
        # Find the run_api function
        run_api_pos = updated_api.find("def run_api(")
        if run_api_pos > 0:
            # Find the function body
            body_pos = updated_api.find(":", run_api_pos)
            if body_pos > 0:
                # Find the uvicorn.run line
                uvicorn_pos = updated_api.find("uvicorn.run", body_pos)
                if uvicorn_pos > 0:
                    # Replace with optimized settings
                    old_uvicorn = 'uvicorn.run("api:app", host=host, port=port, reload=reload)'
                    new_uvicorn = 'uvicorn.run("api:app", host=host, port=port, reload=reload, timeout_keep_alive=75, workers=1)'
                    updated_api = updated_api.replace(old_uvicorn, new_uvicorn)
                    print("✅ Updated uvicorn settings for better memory management")
    
    # Write updated api.py
    if updated_api != api_content:
        with open("api.py", "w") as f:
            f.write(updated_api)
        print("✅ api.py updated successfully")
    else:
        print("⚠️ No changes needed to api.py")
    
    # Update law.py to handle memory errors
    with open("law.py", "r") as f:
        law_content = f.read()
    
    updated_law = law_content
    
    # 1. Update template format for selected model if needed
    if selected_model == "llama3.2" and '<|im_start|>system' in law_content:
        # Replace with llama3.2 format
        updated_law = updated_law.replace('<|im_start|>system', '<|start_header_id|>system<|end_header_id|>')
        updated_law = updated_law.replace('<|im_end|>', '<|eot_id|>')
        updated_law = updated_law.replace('<|im_start|>user', '<|start_header_id|>user<|end_header_id|>')
        updated_law = updated_law.replace('<|im_start|>assistant', '<|start_header_id|>assistant<|end_header_id|>')
        print("✅ Updated prompt format for llama3.2 in law.py")
    
    # 2. Add memory error handling
    if "model requires more system memory" not in updated_law:
        # Find the appropriate section
        error_check_pos = updated_law.find("if response.status_code == 200:")
        if error_check_pos > 0:
            # Add error handling before this check
            memory_error_handler = """
                if response.status_code != 200:
                    # Check for memory error
                    if response.status_code == 500 and "memory" in response.text.lower():
                        logger.error(f"Memory error with model {model_name}. Returning context-only response.")
                        return f"I found relevant information about your query, but don't have enough memory to generate a full response. Here's the relevant context:\\n\\n{context_text[:1500]}..."
                    logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                    return f"Error accessing Ollama (HTTP {response.status_code}). Using context-only response:\\n\\n{context_text[:1500]}..."
            """
            updated_law = updated_law[:error_check_pos] + memory_error_handler + updated_law[error_check_pos:]
            print("✅ Added memory error handling to law.py")
    
    # 3. Add dynamic model selection based on environment variable
    if "Get model name from environment" not in updated_law:
        # Find the get_response_with_context method
        method_pos = updated_law.find("async def get_response_with_context")
        if method_pos > 0:
            # Find where model_name is used
            model_name_pos = updated_law.find("model_name: str =", method_pos)
            if model_name_pos > 0:
                # Find line end
                line_end = updated_law.find("\n", model_name_pos)
                if line_end > 0:
                    # Update with environment variable
                    updated_law = updated_law[:model_name_pos] + 'model_name: str = None' + updated_law[line_end:]
                    
                    # Add code to get model from environment
                    body_pos = updated_law.find(":", method_pos)
                    body_start = updated_law.find("\n", body_pos)
                    if body_start > 0:
                        env_code = """
        # Get model name from environment or use parameter
        if model_name is None:
            model_name = os.environ.get("DEFAULT_MODEL", "tinyllama")
            logger.info(f"Using model from environment: {model_name}")
        """
                        updated_law = updated_law[:body_start+1] + env_code + updated_law[body_start+1:]
                        print("✅ Added dynamic model selection from environment to law.py")
    
    # 4. Add memory-efficient options
    if "num_predict" not in updated_law:
        # Find the Ollama API payload creation
        payload_pos = updated_law.find("payload = {")
        if payload_pos > 0:
            # Find the options section
            options_pos = updated_law.find("\"options\": {", payload_pos)
            if options_pos > 0:
                # Find the end of options
                options_end = updated_law.find("}", options_pos)
                if options_end > 0:
                    # Add num_predict option
                    updated_law = updated_law[:options_end] + ",\n                        \"num_predict\": 1024" + updated_law[options_end:]
                    print("✅ Added memory-efficient options to law.py")
    
    # Write updated law.py
    if updated_law != law_content:
        with open("law.py", "w") as f:
            f.write(updated_law)
        print("✅ law.py updated successfully")
    else:
        print("⚠️ No changes needed to law.py")
        
except Exception as e:
    print(f"❌ Error updating API code: {str(e)}")

# Step 6: Update supervisord.conf for better process management
print("\nStep 6: Updating supervisord configuration...")

try:
    with open("supervisord.conf", "r") as f:
        supervisord_content = f.read()
    
    updated_supervisord = supervisord_content
    
    # 1. Add memory limits to Ollama
    if "OLLAMA_NUM_THREADS" not in updated_supervisord and "[program:ollama]" in updated_supervisord:
        # Find ollama program section
        ollama_pos = updated_supervisord.find("[program:ollama]")
        if ollama_pos > 0:
            # Find where to add environment variables
            command_pos = updated_supervisord.find("command=", ollama_pos)
            if command_pos > 0:
                # Replace command with environment variables
                env_vars = """environment=OLLAMA_NUM_THREADS=2,OLLAMA_KEEP_ALIVE=120m
command=/bin/bash -c "ollama serve"""
                updated_supervisord = updated_supervisord.replace("command=/bin/bash -c \"ollama serve", env_vars)
                print("✅ Added memory optimization to supervisord.conf")
    
    # 2. Add memory monitoring
    if "memory_monitor" not in updated_supervisord:
        # Add new program section at the end
        memory_monitor = """
[program:memory_monitor]
command=/bin/bash -c "while true; do free -h >> /app/memory.log; sleep 60; done"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
priority=30
autostart=true
autorestart=true
"""
        updated_supervisord = updated_supervisord + memory_monitor
        print("✅ Added memory monitoring to supervisord.conf")
    
    # 3. Update API settings
    if "api" in updated_supervisord and "--timeout-keep-alive" not in updated_supervisord:
        # Find api program section
        api_pos = updated_supervisord.find("[program:api]")
        if api_pos > 0:
            # Find the command line
            api_command_pos = updated_supervisord.find("command=", api_pos)
            if api_command_pos > 0:
                # Find the end of the line
                line_end = updated_supervisord.find("\n", api_command_pos)
                if line_end > 0:
                    # Replace with optimized command
                    old_command = updated_supervisord[api_command_pos:line_end]
                    if "uvicorn" not in old_command:
                        # Update entrypoint.sh usage
                        updated_supervisord = updated_supervisord.replace(
                            "command=/bin/bash -c \"cd /app && chmod +x entrypoint.sh && ./entrypoint.sh\"",
                            "command=/bin/bash -c \"cd /app && chmod +x entrypoint.sh && ./entrypoint.sh\""
                        )
                    else:
                        # Direct uvicorn command
                        new_command = "command=/bin/bash -c \"cd /app && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75 --workers 1\""
                        updated_supervisord = updated_supervisord.replace(old_command, new_command)
                        print("✅ Updated API settings in supervisord.conf")
    
    # Write updated supervisord.conf
    if updated_supervisord != supervisord_content:
        with open("supervisord.conf", "w") as f:
            f.write(updated_supervisord)
        print("✅ supervisord.conf updated successfully")
    else:
        print("⚠️ No changes needed to supervisord.conf")
        
except Exception as e:
    print(f"❌ Error updating supervisord.conf: {str(e)}")

# Step 7: Create build and test scripts
print("\nStep 7: Creating Docker build and test scripts...")

docker_build_script = f"""#!/bin/bash
# Docker build script for LegalGPT FastAPI

echo "=== Building LegalGPT Docker Container ==="
echo "Using {selected_model} as the default model"

# Build the Docker image
docker-compose build

echo "=== Build Complete ==="
echo "To start the container, run: docker-compose up -d"
echo "To view logs, run: docker-compose logs -f"
echo "To test the API, run: ./docker_test.sh"
"""

with open("docker_build.sh", "w") as f:
    f.write(docker_build_script)
    
os.chmod("docker_build.sh", 0o755)
print("✅ Created docker_build.sh script")

docker_test_script = """#!/bin/bash
# Docker test script for LegalGPT FastAPI

echo "=== Testing LegalGPT API in Docker ==="

# Check if the container is running
CONTAINER_ID=$(docker-compose ps -q legal-gpt)
if [ -z "$CONTAINER_ID" ]; then
    echo "❌ LegalGPT container is not running"
    echo "Please start it with: docker-compose up -d"
    exit 1
fi

echo "✅ LegalGPT container is running: $CONTAINER_ID"

# Check API status
echo "Testing API status..."
RESPONSE=$(curl -s http://localhost:8000/api)
if [[ $RESPONSE == *"ok"* ]]; then
    echo "✅ API is responding: $RESPONSE"
else
    echo "❌ API status test failed: $RESPONSE"
    echo "Check the logs with: docker-compose logs -f"
    exit 1
fi

# Test sample questions endpoint
echo "Testing sample questions..."
QUESTIONS=$(curl -s http://localhost:8000/sample-questions)
if [[ $QUESTIONS == *"questions"* ]]; then
    echo "✅ Sample questions available"
else
    echo "❌ Sample questions test failed"
fi

# Test chat endpoint (this is the one that was failing)
echo "Testing chat endpoint (this may take a minute)..."
echo "Sending test query to /chat endpoint..."

RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the requirements for starting a business in Kenya?","model_name":"'${DEFAULT_MODEL:-tinyllama}'","site_filter":null}')

if [[ $RESPONSE == *"response"* ]]; then
    echo "✅ Chat endpoint is working!"
    echo "First part of response: $(echo $RESPONSE | grep -o '\"response\":\"[^\"]*' | head -c 100)..."
else
    echo "❌ Chat endpoint test failed"
    echo "Response: $RESPONSE"
    echo "Check the logs with: docker-compose logs -f"
    
    echo "Checking container memory usage..."
    docker stats --no-stream $CONTAINER_ID
    
    echo "Checking container logs for errors..."
    docker-compose logs --tail 50 | grep -i error
fi

echo "=== Test Complete ==="
"""

with open("docker_test.sh", "w") as f:
    f.write(docker_test_script)
    
os.chmod("docker_test.sh", 0o755)
print("✅ Created docker_test.sh script")

# Final step: Summary and instructions
print("\n=== Docker Optimization Complete ===")
print(f"Your LegalGPT FastAPI application has been optimized for Docker deployment using the {selected_model} model.")
print("\nChanges made:")
print("1. Updated Dockerfile with memory optimizations")
print("2. Updated docker-compose.yml with resource limits")
print("3. Enhanced entrypoint.sh with dynamic model selection")
print("4. Improved API code to handle memory errors gracefully")
print("5. Updated supervisord.conf for better process management")
print("6. Created Docker build and test scripts")
print("\nTo deploy your application:")
print("1. Build the Docker image:")
print("   ./docker_build.sh")
print("\n2. Start the container:")
print("   docker-compose up -d")
print("\n3. Test the API:")
print("   ./docker_test.sh")
print("\n4. Monitor container resources:")
print("   docker stats")
print("\n5. View logs:")
print("   docker-compose logs -f")
print("\nIf you need more memory-efficient options in the future, consider:")
print("- Using a quantized model like tinyllama (requires less memory)")
print("- Increasing container memory limit in docker-compose.yml")
print("- Using a cloud instance with more RAM")