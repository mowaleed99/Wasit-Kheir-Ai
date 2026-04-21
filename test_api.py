import io
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

def create_dummy_image():
    # Create a simple 100x100 black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white rectangle
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
    
    # Encode as JPEG
    is_success, buffer = cv2.imencode(".jpg", img)
    if is_success:
        return io.BytesIO(buffer).getvalue()
    return None

client = TestClient(app)

def run_tests():
    print("Testing Endpoints...")

    # 1. Health check
    response = client.get("/api/v1/health")
    print(f"GET /health: {response.status_code} - {response.text}")

    # 2. Text embedding
    response = client.post("/api/v1/text-embedding", data={"text": "Lost red wallet with green ID card"})
    print(f"POST /text-embedding: {response.status_code} - {response.text}")

    # 3. Add text
    response = client.post("/api/v1/add-text", data={"text": "Lost red wallet with green ID card", "post_id": "post_123"})
    print(f"POST /add-text: {response.status_code} - {response.text}")

    # 4. Search text
    response = client.post("/api/v1/search-text", data={"text": "red wallet", "k": 5})
    print(f"POST /search-text: {response.status_code} - {response.text}")

    # Create dummy image
    img_bytes = create_dummy_image()
    if img_bytes:
        files = {"image": ("dummy.jpg", img_bytes, "image/jpeg")}

        # 5. Image embedding
        response = client.post("/api/v1/image-embedding", files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /image-embedding: {response.status_code} - {response.text}")

        # 6. Add image
        response = client.post("/api/v1/add-image", data={"post_id": "post_124"}, files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /add-image: {response.status_code} - {response.text}")

        # 7. Search image
        response = client.post("/api/v1/search-image", data={"k": 5}, files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /search-image: {response.status_code} - {response.text}")

        # Face endpoints
        # 8. Add face
        response = client.post("/api/v1/add-face", data={"person_id": "person_abc"}, files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /add-face: {response.status_code} - {response.text}")

        # 9. Face match
        response = client.post("/api/v1/face-match", data={"k": 5}, files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /face-match: {response.status_code} - {response.text}")

        # 10. Multimodal search
        response = client.post("/api/v1/multimodal-search", data={"text": "wallet", "k": 5}, files={"image": ("dummy.jpg", img_bytes, "image/jpeg")})
        print(f"POST /multimodal-search: {response.status_code} - {response.text}")
    else:
        print("Failed to create dummy image. Skipping image endpoints.")

if __name__ == "__main__":
    run_tests()
