"""
API routes for embeddings, vector storage and specific entity fetching.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import numpy as np

from app.core.embeddings import GeminiEmbedder
from app.core.vector_store import VectorStore
from app.core.face_matcher import FaceRecognizer
from app.core.multimodal import MultimodalMatcher
from app.config import TEXT_EMBED_DIM, IMAGE_EMBED_DIM, FACE_EMBED_DIM

router = APIRouter()

# تهيئة المكونات العالمية
embedder = GeminiEmbedder()
text_store = VectorStore(TEXT_EMBED_DIM)
image_store = VectorStore(IMAGE_EMBED_DIM)
face_store = VectorStore(FACE_EMBED_DIM)

try:
    face_recognizer = FaceRecognizer()
except Exception as e:
    print(f"Failed to initialize FaceRecognizer: {e}")
    face_recognizer = None

multimodal_matcher = MultimodalMatcher(text_store, image_store)

@router.post("/text-embedding")
async def get_text_embedding(text: str = Form(...)):
    try:
        embedding = embedder.get_text_embedding(text)
        return {"status": "success", "data": {"embedding": embedding}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image-embedding")
async def get_image_embedding(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        embedding = embedder.get_image_embedding(image_bytes)
        return {"status": "success", "data": {"embedding": embedding}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-text")
async def add_text(text: str = Form(...), post_id: str = Form(...)):
    try:
        embedding = embedder.get_text_embedding(text)
        text_store.add(np.array(embedding, dtype=np.float32), {"post_id": post_id, "text": text})
        return {"status": "success", "data": {"message": "Text added successfully", "post_id": post_id}}
    except Exception as e:
        print(f"ERROR in add-text: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/add-image")
async def add_image(post_id: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        embedding = embedder.get_image_embedding(image_bytes)
        image_store.add(np.array(embedding, dtype=np.float32), {"post_id": post_id})
        return {"status": "success", "data": {"message": "Image added successfully", "post_id": post_id}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-face")
async def add_face(person_id: str = Form(...), image: UploadFile = File(...)):
    if not face_recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer not initialized.")
    try:
        image_bytes = await image.read()
        embedding = face_recognizer.get_face_embedding(image_bytes)
        if embedding is None:
            return {"status": "error", "error": "No face detected in the image"}
        face_store.add(embedding, {"person_id": person_id})
        return {"status": "success", "data": {"message": "Face added successfully", "person_id": person_id}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-text")
async def search_text(text: str = Form(...), k: int = Form(5)):
    try:
        embedding = embedder.get_text_embedding(text)
        results = text_store.search(np.array(embedding, dtype=np.float32), k)
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-image")
async def search_image(k: int = Form(5), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        embedding = embedder.get_image_embedding(image_bytes)
        results = image_store.search(np.array(embedding, dtype=np.float32), k)
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/face-match")
async def face_match(k: int = Form(5), image: UploadFile = File(...)):
    if not face_recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer not initialized.")
    try:
        image_bytes = await image.read()
        embedding = face_recognizer.get_face_embedding(image_bytes)
        if embedding is None:
            return {"status": "success", "data": {"results": [], "message": "No face detected"}}
        results = face_store.search(embedding, k)
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multimodal-search")
async def multimodal_search(
    text: Optional[str] = Form(None), 
    image: Optional[UploadFile] = File(None),
    k: int = Form(5)
):
    try:
        if not text and not image:
            raise HTTPException(status_code=400, detail="Must provide at least text or image")
        text_emb = None
        if text:
            text_emb = np.array(embedder.get_text_embedding(text), dtype=np.float32)
        image_emb = None
        if image:
            image_bytes = await image.read()
            image_emb = np.array(embedder.get_image_embedding(image_bytes), dtype=np.float32)
        results = multimodal_matcher.search(text_emb, image_emb, k)
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok", "data": {"message": "Server is running"}}