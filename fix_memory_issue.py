#!/usr/bin/env python3
import requests
import json
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
            
            # Check if any models worked
            if working_models:
                print(f"Working models: {working_models}")
                return working_models
            else:
                print("No models worked with the available memory")
                return None
        else:
            print(f"Error getting models: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return None

def create_ollama_fallback_patch():
    """Create a patch to update the API to handle memory issues with fallback to smaller models"""
    working_models = find_working_model()
    
    if not working_models:
        print("No working models found. Trying to create a minimal embedding-only model.")
        
        # Try to pull a tiny model
        try:
            response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": "tinyllama"},
                timeout=60
            )
            
            if response.status_code == 200:
                print("Successfully pulled tinyllama model")
                working_models = ["tinyllama"]
            else:
                print(f"Failed to pull tinyllama: {response.status_code}")
        except Exception as e:
            print(f"Error pulling tinyllama: {str(e)}")
    
    # Create a fix script that will update the API to handle memory issues
    with open("/home/dennis/Desktop/projects/LegalGPT_fastapi/api_memory_fix.py", "w") as f:
        f.write(f'''#!/usr/bin/env python3
import os

# Path to the law.py file
law_py_path = "/home/dennis/Desktop/projects/LegalGPT_fastapi/law.py"

# Load the content
with open(law_py_path, "r") as f:
    content = f.read()

# Define the working models found during diagnosis
working_models = {working_models or ["llama2"]}

# Create a fallback model selection function to replace in the code
fallback_function = """
    async def get_response_with_context(self, query: str, top_k: int = None, site_filter: str = None,
                                       model_name: str = "llama3") -> str:
        """
        Get response with context from both Kenya Law sites
        
        Args:
            query: User query
            top_k: Number of context passages to retrieve (defaults to self.max_context_chunks)
            site_filter: Optional site to filter by ("kenyalaw.org" or "new.kenyalaw.org")
            model_name: Name of the model to use with Ollama (if available)
            
        Returns:
            LLM response with context
        """
        logger.info(f"Getting response for query: '{query[:50]}...'")
        
        try:
            # Use default top_k if not specified
            if top_k is None:
                top_k = self.max_context_chunks
                
            # Query for relevant context across both sites
            context_results = self.query(query, top_k=top_k, site_filter=site_filter)
            
            # Build context string, respecting the context limit
            context_text = ""
            sources = []
            
            for result in context_results:
                # Get source information
                url = result["metadata"].get("url", "unknown")
                title = result["metadata"].get("title", "")
                site = result.get("site", "unknown")
                
                # Track sources for attribution
                if url not in [s[0] for s in sources]:
                    sources.append((url, title))
                
                # Format source with title when available
                source_info = f"Source: {title} ({url})" if title else f"Source: {url}"
                
                # Add text with a separator
                new_context = f"\\n\\n{source_info}:\\n{result['text']}"
                
                # Check if adding this would exceed the context limit
                if len(context_text) + len(new_context) > self.context_limit:
                    # If we're at the limit, stop adding more
                    if context_text:
                        break
                        
                    # If the first context is already too large, truncate it
                    new_context = new_context[:self.context_limit]
                    
                context_text += new_context
            
            # If no context was found, provide a helpful message
            if not context_text:
                return "I couldn't find any relevant information to answer your question about Kenya Law. Please try rephrasing your question or ask about a different legal topic."
            
            # Create prompt - use the right format for available models
            # Working models determined by the memory check: {working_models or ["llama2"]}
            # Try to use model_name if provided, otherwise fall back to a working model
            
            # Define the fallback model if the requested one isn't available
            fallback_model = "{working_models[0] if working_models else 'llama2'}"
            
            # Try to use Ollama if available
            try:
                import requests
                import os
                import json
                
                # Get Ollama host from environment variable or use default
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                logger.info(f"Sending request to Ollama at: {ollama_host}")
                
                # Check if Ollama is available with multiple retries
                max_retries = 3
                retry_delay = 2
                retry_count = 0
                status_response = None
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"Checking Ollama status (attempt {retry_count+1}/{max_retries})...")
                        status_response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                        if status_response.status_code == 200:
                            logger.info("Ollama status check successful")
                            break
                        else:
                            logger.warning(f"Ollama status check failed with code: {status_response.status_code}")
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.info(f"Retrying in {retry_delay} seconds...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                    except Exception as e:
                        logger.warning(f"Ollama status check error: {str(e)}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                
                if retry_count >= max_retries or not status_response or status_response.status_code != 200:
                    logger.error("Ollama not available after multiple retries")
                    return f"I found some information that might help answer your question, but the AI model service is currently unavailable.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                
                # Check if the requested model is available and select a fallback if needed
                available_models = []
                requested_model_available = False
                
                try:
                    models_json = status_response.json()
                    available_models = [model.get("name", "").split(':')[0] for model in models_json.get("models", [])]
                    logger.info(f"Available models: {available_models}")
                    
                    # Check if requested model is available
                    if model_name in available_models:
                        requested_model_available = True
                    else:
                        # Remove ":latest" suffix if present
                        model_name_base = model_name.split(':')[0] if ':' in model_name else model_name
                        requested_model_available = model_name_base in available_models
                        if requested_model_available:
                            model_name = model_name_base
                    
                    # Fall back to a working model if needed
                    if not requested_model_available:
                        # Find the first working model that's available
                        working_models_available = [m for m in {working_models} if m in available_models]
                        
                        if working_models_available:
                            model_name = working_models_available[0]
                            logger.info(f"Requested model '{model_name}' not available. Using fallback model: {model_name}")
                        elif available_models:
                            model_name = available_models[0]
                            logger.info(f"Using first available model: {model_name}")
                        else:
                            logger.error("No models available in Ollama")
                            return f"I found some information that might help answer your question, but no language models are available.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                except Exception as e:
                    logger.error(f"Error checking available models: {str(e)}")
                    # Try to use the fallback model
                    model_name = fallback_model
                    logger.info(f"Using fallback model due to error: {model_name}")
                
                # Different prompt formats based on model
                if model_name.startswith("llama3"):
                    # LLaMA 3 format
                    prompt = f\"\"\"<|start_header_id|>system<|end_header_id|>
You are a Kenya Law Assistant providing accurate information based solely on the Kenya Law website content provided. 
Your role is to assist with queries related to Kenyan laws, statutes, case law, and legal frameworks.

Carefully analyze the following context information to provide accurate answers:

{context_text}

Important guidelines:
1. ONLY use the information provided in the context. Do not rely on prior knowledge.
2. If the context doesn't contain information to answer the question fully, clearly state what information is missing.
3. For legal queries, cite specific sections, cases, or statutes from the context when applicable.
4. Use formal, professional language appropriate for legal discussions.
5. Avoid speculating about legal interpretations beyond what's explicitly stated in the context.
6. When uncertain, acknowledge the limitations of the available information.
7. Keep your response focused and concise.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
\"\"\"
                elif model_name.startswith("llama2") or model_name.startswith("codellama"):
                    # LLaMA 2 format
                    prompt = f\"\"\"[INST] <<SYS>>
You are a Kenya Law Assistant providing accurate information based solely on the Kenya Law website content provided. 
Your role is to assist with queries related to Kenyan laws, statutes, case law, and legal frameworks.

Carefully analyze the following context information to provide accurate answers:

{context_text}

Important guidelines:
1. ONLY use the information provided in the context. Do not rely on prior knowledge.
2. If the context doesn't contain information to answer the question fully, clearly state what information is missing.
3. For legal queries, cite specific sections, cases, or statutes from the context when applicable.
4. Use formal, professional language appropriate for legal discussions.
5. Avoid speculating about legal interpretations beyond what's explicitly stated in the context.
6. When uncertain, acknowledge the limitations of the available information.
7. Keep your response focused and concise.
<</SYS>>

{query} [/INST]
\"\"\"
                else:
                    # Generic format for other models
                    prompt = f\"\"\"System: You are a Kenya Law Assistant providing accurate information based solely on the Kenya Law website content provided.

Context information:
{context_text}

User question: {query}

Answer:
\"\"\"
                
                # Prepare request payload with adaptive settings based on available memory
                payload = {{
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {{
                        "temperature": 0.1,  # Lower temperature for more factual responses
                        "top_p": 0.95,
                        "top_k": 40,
                        "num_predict": 1024  # Limit output tokens to avoid memory issues
                    }}
                }}
                
                logger.info(f"Sending request to Ollama with model: {model_name}")
                
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
                
                # Make request to Ollama API with dynamically set timeout and retry strategy
                try:
                    response = session.post(
                        f"{ollama_host}/api/generate",
                        json=payload,
                        timeout=(10, ollama_timeout)  # (connect_timeout, read_timeout)
                    )
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.info("Successfully received response from Ollama")
                            return result["response"]
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Error decoding Ollama response: {str(json_err)}")
                            return f"I found some information that might help answer your question, but there was an error processing the AI response.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                    else:
                        # If we get a memory error, try with a smaller model
                        if response.status_code == 500 and "memory" in response.text.lower():
                            logger.error(f"Memory error with model {model_name}, trying smaller model")
                            
                            # Find the smallest available model or fall back to just showing context
                            if len(available_models) > 1:
                                # Try a different model
                                smaller_model = None
                                for m in available_models:
                                    if m != model_name:
                                        smaller_model = m
                                        break
                                
                                if smaller_model:
                                    logger.info(f"Retrying with smaller model: {smaller_model}")
                                    # Update payload with smaller model
                                    payload["model"] = smaller_model
                                    # Use a simpler prompt
                                    payload["prompt"] = f"Context: {context_text[:2000]}...\\n\\nQuestion: {query}\\n\\nAnswer:"
                                    
                                    # Try again with smaller model
                                    retry_response = session.post(
                                        f"{ollama_host}/api/generate",
                                        json=payload,
                                        timeout=(10, ollama_timeout)
                                    )
                                    
                                    if retry_response.status_code == 200:
                                        try:
                                            retry_result = retry_response.json()
                                            logger.info(f"Successfully received response from smaller model {smaller_model}")
                                            return retry_result["response"]
                                        except Exception:
                                            pass
                            
                            # If retrying fails or no other models, return context-only response
                            logger.error("All models failed due to memory constraints")
                            return f"I found some information that might help answer your question, but don't have enough memory to generate a proper response.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                        
                        logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                        return f"I found some information that might help answer your question, but the AI model encountered an error.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                
                except requests.RequestException as req_err:
                    logger.error(f"Request error connecting to Ollama: {str(req_err)}")
                    return f"I found some information that might help answer your question, but couldn't connect to the AI model.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                    
            except requests.RequestException as req_err:
                logger.error(f"Request error connecting to Ollama: {str(req_err)}")
                return f"I found some information that might help answer your question, but couldn't connect to the AI model.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
            except Exception as e:
                logger.error(f"Unexpected error with Ollama: {str(e)}")
                return f"I found some information that might help answer your question, but there was an error using the AI model: {str(e)}.\\n\\nHere's what I found:\\n{context_text[:1000]}..."
                
        except Exception as e:
            logger.error(f"Error in get_response_with_context: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
"""

# Replace the original function with our improved version
if "get_response_with_context" in content:
    # Find the function definition
    start_idx = content.find("    async def get_response_with_context")
    
    if start_idx != -1:
        # Find the end of the function by searching for the next class or function definition
        next_def_idx = content.find("    async def", start_idx + 1)
        next_class_idx = content.find("class ", start_idx + 1)
        
        end_idx = min(next_def_idx if next_def_idx != -1 else float('inf'), 
                      next_class_idx if next_class_idx != -1 else float('inf'))
        
        if end_idx != float('inf'):
            # Replace the function
            content = content[:start_idx] + fallback_function + content[end_idx:]
            
            # Write the updated content back
            with open(law_py_path, "w") as f:
                f.write(content)
            
            print("Successfully updated the get_response_with_context function in law.py")
        else:
            print("Could not find the end of the get_response_with_context function")
    else:
        print("Could not find the get_response_with_context function in law.py")
else:
    print("Could not find the get_response_with_context function in law.py")
''')
    
    # Make the script executable
    os.chmod("/home/dennis/Desktop/projects/LegalGPT_fastapi/api_memory_fix.py", 0o755)
    
    print("\nCreated a script to fix the API's memory handling.")
    print("Run the following commands to fix the memory issues and try again:")
    print("1. python3 /home/dennis/Desktop/projects/LegalGPT_fastapi/api_memory_fix.py")
    print("2. python3 /home/dennis/Desktop/projects/LegalGPT_fastapi/start_api.sh")
    
    # Return success/failure based on whether we found working models
    return working_models is not None

if __name__ == "__main__":
    print("Testing for models that work with available memory...")
    if find_working_model():
        create_ollama_fallback_patch()
        sys.exit(0)
    else:
        print("Failed to find any working models.")
        # Create fix script anyway
        create_ollama_fallback_patch()
        sys.exit(1)