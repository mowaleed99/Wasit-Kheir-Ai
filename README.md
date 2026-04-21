# Lost & Found AI Backend

This is the backend service for the Lost & Found Platform, providing multimodal search (text and image) and face recognition endpoints.

## Setup Instructions

1. **Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables:**
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Make sure to replace `your_key_here` with your actual Google Gemini API key.

4. **Run the server:**
```bash
uvicorn app.main:app --reload
```

## Testing the Endpoints with `curl`

### 1. Multimodal Text + Image Search

Add text and an image of a lost item to the database:
```bash
# Add a lost brown leather wallet
curl -X POST http://localhost:8000/api/v1/add-text \
  -F "text=Lost a brown leather wallet near the library" \
  -F "post_id=post_1"

curl -X POST http://localhost:8000/api/v1/add-image \
  -F "post_id=post_1" \
  -F "image=@/path/to/wallet_image.jpg"
```

Perform a Multimodal Search:
```bash
curl -X POST http://localhost:8000/api/v1/multimodal-search \
  -F "text=Brown wallet" \
  -F "image=@/path/to/similar_wallet_image.jpg" \
  -F "k=5"
```

### 2. Face Matching (Missing Persons)

Add a face to the database:
```bash
curl -X POST http://localhost:8000/api/v1/add-face \
  -F "person_id=person_123" \
  -F "image=@/path/to/clear_face.jpg"
```

Search for a matching face:
```bash
curl -X POST http://localhost:8000/api/v1/face-match \
  -F "k=3" \
  -F "image=@/path/to/found_person_face.jpg"
```

## Integrating with Backend/Flutter
- All endpoints are scoped under `/api/v1/`.
- Send text payloads directly in Form Data.
- Upload files using `multipart/form-data` with the key `image`.
- On successful requests, it returns a standard JSON envelope: `{"status": "success", "data": {...}}`.
