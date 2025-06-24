#!/usr/bin/env python3
import requests
import json
import time
import sys

def test_chat_api():
    # Chat endpoint
    url = "http://localhost:8000/chat"
    
    # Sample query
    payload = {
        "query": "What are the requirements for starting a business in Kenya?",
        "model_name": "llama3", 
        "site_filter": None
    }
    
    try:
        print(f"Sending request to {url} with payload: {json.dumps(payload)}")
        start_time = time.time()
        
        # Send POST request to API
        response = requests.post(url, json=payload, timeout=600)
        
        end_time = time.time()
        print(f"Request took {end_time - start_time:.2f} seconds")
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response content:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing chat API endpoint...")
    success = test_chat_api()
    if success:
        print("Chat API test succeeded!")
        sys.exit(0)
    else:
        print("Chat API test failed!")
        sys.exit(1)