"""
API routes for embeddings, vector storage and specific entity fetching.
Uses Modal for FAISS operations and InsightFace queries.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import httpx
import traceback
from app.core.embeddings import GeminiEmbedder
from app.config import MODAL_URL, MODAL_API_KEY

router = APIRouter()

# تهيئة المكونات العالمية (Gemini Embedder works locally)
embedder = GeminiEmbedder()

async def add_vector_to_modal(embedding: list, metadata: dict, index_name: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MODAL_URL}/add-vector",
            json={
                "embedding": embedding,
                "metadata": metadata,
                "index_name": index_name
            },
            headers={"X-API-Key": MODAL_API_KEY},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

async def search_vector_on_modal(embedding: list, k: int, index_name: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MODAL_URL}/search-vector",
            json={
                "embedding": embedding,
                "k": k,
                "index_name": index_name
            },
            headers={"X-API-Key": MODAL_API_KEY},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json().get("data", {}).get("results", [])


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
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        await add_vector_to_modal(embedding, {"post_id": post_id, "text": text}, "text")
        return {"status": "success", "data": {"message": "Text added successfully", "post_id": post_id}}
    except Exception as e:
        print(f"ERROR in add-text: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-image")
async def add_image(post_id: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        embedding = embedder.get_image_embedding(image_bytes)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        await add_vector_to_modal(embedding, {"post_id": post_id}, "image")
        return {"status": "success", "data": {"message": "Image added successfully", "post_id": post_id}}
    except Exception as e:
        print(f"ERROR in add-image: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-face")
async def add_face(person_id: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{MODAL_URL}/add-face",
                data={"person_id": person_id},
                files={"image": (image.filename, image_bytes, image.content_type)},
                headers={"X-API-Key": MODAL_API_KEY},
                timeout=60.0,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except Exception as e:
        print(f"ERROR in add-face: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-text")
async def search_text(text: str = Form(...), k: int = Form(5)):
    try:
        embedding = embedder.get_text_embedding(text)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        results = await search_vector_on_modal(embedding, k, "text")
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        print(f"ERROR in search-text: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-image")
async def search_image(k: int = Form(5), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        embedding = embedder.get_image_embedding(image_bytes)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        results = await search_vector_on_modal(embedding, k, "image")
        return {"status": "success", "data": {"results": results}}
    except Exception as e:
        print(f"ERROR in search-image: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/face-match")
async def face_match(k: int = Form(5), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{MODAL_URL}/face-match",
                data={"k": k},
                files={"image": (image.filename, image_bytes, image.content_type)},
                headers={"X-API-Key": MODAL_API_KEY},
                timeout=60.0,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except Exception as e:
        print(f"ERROR in face-match: {e}")
        traceback.print_exc()
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
        
        text_results = []
        image_results = []

        if text:
            text_emb = embedder.get_text_embedding(text)
            if hasattr(text_emb, "tolist"): text_emb = text_emb.tolist()
            text_results = await search_vector_on_modal(text_emb, k * 2, "text")

        if image:
            image_bytes = await image.read()
            image_emb = embedder.get_image_embedding(image_bytes)
            if hasattr(image_emb, "tolist"): image_emb = image_emb.tolist()
            image_results = await search_vector_on_modal(image_emb, k * 2, "image")
        
        # دمج النتائج باستخدام قاموس لتجميع الدرجات حسب post_id
        text_weight = 0.5
        image_weight = 0.5
        combined = {}
        
        for res in text_results:
            pid = res['metadata'].get('post_id')
            if pid:
                combined[pid] = {
                    'score': res['score'] * text_weight,
                    'metadata': res['metadata']
                }
        
        for res in image_results:
            pid = res['metadata'].get('post_id')
            if pid:
                if pid in combined:
                    combined[pid]['score'] += res['score'] * image_weight
                else:
                    combined[pid] = {
                        'score': res['score'] * image_weight,
                        'metadata': res['metadata']
                    }
        
        # تحويل إلى قائمة وترتيب تنازلي حسب الدرجة
        results_list = list(combined.values())
        results_list.sort(key=lambda x: x['score'], reverse=True)
        return {"status": "success", "data": {"results": results_list[:k]}}

    except Exception as e:
        print(f"ERROR in multimodal-search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    # Check local status
    status = {"status": "ok", "data": {"message": "Server is running (with Modal backend)", "modal": "unknown"}}
    
    # Let's ping Modal also to ensure connectivity
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MODAL_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                status["data"]["modal"] = "connected"
                status["data"]["modal_stats"] = resp.json().get("data")
            else:
                status["data"]["modal"] = "error"
    except Exception:
        status["data"]["modal"] = "unreachable"
        
    return status