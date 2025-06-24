#!/usr/bin/env python3
import requests
import json
import time
import sys

def test_api_status():
    """Test if the API is running"""
    url = "http://localhost:8000/api"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"API status: {response.json()}")
            return True
        else:
            print(f"API status error: {response.status_code}")
            return False
    except Exception as e:
        print(f"API connection error: {str(e)}")
        return False

def test_sample_questions():
    """Test sample questions endpoint"""
    url = "http://localhost:8000/sample-questions"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            questions = response.json().get("questions", [])
            print(f"Sample questions available: {len(questions)}")
            return True
        else:
            print(f"Sample questions error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Sample questions error: {str(e)}")
        return False

def test_chat():
    """Test chat endpoint"""
    url = "http://localhost:8000/chat"
    
    # Use a simple query
    payload = {
        "query": "What are the key provisions of the Kenyan Constitution?",
        "model_name": "llama3.2",  # Use the model we know works
        "site_filter": None
    }
    
    try:
        print(f"Testing chat endpoint with query: {payload['query']}")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=60)
        
        end_time = time.time()
        print(f"Request took {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Chat response received successfully!")
            print(f"Sources found: {len(result.get('sources', []))}")
            print(f"First 100 chars of response: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"Chat endpoint error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API endpoints...")
    
    # Test if API is running
    print("\n1. Testing API status...")
    if not test_api_status():
        print("API is not running. Please start the API first.")
        sys.exit(1)
    
    # Test sample questions
    print("\n2. Testing sample questions...")
    test_sample_questions()
    
    # Test chat endpoint
    print("\n3. Testing chat endpoint...")
    if test_chat():
        print("\nAll tests passed! The API is working correctly.")
        sys.exit(0)
    else:
        print("\nChat endpoint test failed. Please check the API logs for more details.")
        sys.exit(1)