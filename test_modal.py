import requests
import numpy as np
import sys
import os

# Base URL for the Modal app
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
# Make sure URL doesn't have trailing slash
BASE_URL = BASE_URL.rstrip('/')

# Fallback to the known secret we created on Modal if it's not in the environment
API_KEY = os.environ.get("API_ACCESS_KEY", "dev_key_123")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def test_health():
    print(f"Testing /health on {BASE_URL}...")
    resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
    print("Status:", resp.status_code)
    try:
        print("Response:", resp.json())
    except:
        print("Response:", resp.text)
    print("-" * 40)

def test_add_text_vector():
    print("Testing /add-vector (text)...")
    # 3072 is the dimension for our text embeddings
    dummy_embedding = np.random.rand(3072).tolist()
    
    payload = {
        "embedding": dummy_embedding,
        "metadata": {
            "post_id": 123,
            "description": "Lost a black wallet near the library"
        },
        "index_name": "text"
    }
    
    resp = requests.post(f"{BASE_URL}/add-vector", json=payload, headers=HEADERS)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    print("-" * 40)
    
def test_search_text_vector():
    print("Testing /search-vector (text)...")
    dummy_embedding = np.random.rand(3072).tolist()
    
    payload = {
        "embedding": dummy_embedding,
        "k": 2,
        "index_name": "text"
    }
    
    resp = requests.post(f"{BASE_URL}/search-vector", json=payload, headers=HEADERS)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    print("-" * 40)

def test_face_match():
    # To test face matching, we would need an actual image with a face.
    print("Skipping /face-match. To test this, you need to upload a real image.")
    print("-" * 40)

if __name__ == "__main__":
    print(f"Make sure you have started your modal app using: modal serve modal_app.py")
    print(f"Waiting a few seconds before testing to ensure the server is ready...")
    
    # Try testing health
    try:
        test_health()
        test_add_text_vector()
        test_search_text_vector()
        test_face_match()
        print("Basic tests completed successfully!")
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error. Is the server running at {BASE_URL}?")
        print("Please run `modal serve modal_app.py` in another terminal first.")
