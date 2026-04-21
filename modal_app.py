"""
Modal deployment for Lost & Found AI microservice.
Handles: FAISS vector storage/search + Face recognition (InsightFace).
Does NOT handle: Gemini embeddings (done by main backend).

Deploy:   modal deploy modal_app.py
Serve:    modal serve modal_app.py   (dev mode with hot reload)
"""
import modal
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = modal.App("lost-found-ai")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Container Image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def download_insightface_models():
    """Pre-download InsightFace buffalo_l at BUILD time (~300MB).
    This runs ONCE during `modal deploy`, not on every cold start.
    """
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ InsightFace buffalo_l cached in image.")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # System deps for OpenCV headless + insightface ONNX
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    # Python deps — only what Modal actually needs (no google-generativeai!)
    .pip_install(
        "fastapi",
        "uvicorn",
        "faiss-cpu",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "insightface",
        "python-multipart",
        "onnxruntime",
    )
    # Bake InsightFace model into image
    .run_function(download_insightface_models)
    # ↓ This copies _faiss_store.py into the container
    .add_local_file("_faiss_store.py", remote_path="/root/_faiss_store.py")
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persistent Volume (FAISS indexes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the ONE shared volume instance used everywhere.
# NEVER create a second Volume.from_name() — always use this reference.
faiss_volume = modal.Volume.from_name("lost-found-faiss", create_if_missing=True)
VOLUME_PATH = "/data/faiss_indexes"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI Service (single class, serialized requests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("lost-found-secrets")],
    volumes={VOLUME_PATH: faiss_volume},
    allow_concurrent_inputs=1,   # ← CRITICAL: serializes all requests (FAISS safety)
    keep_warm=1,                 # Keep 1 container warm (no cold starts)
    timeout=300,
    cpu=2.0,
    memory=2048,
)
class LostFoundAI:
    """
    Main AI service. All requests are serialized (one at a time per container)
    to prevent FAISS index corruption.
    
    Modal can scale by spawning MORE containers, but each container
    processes requests sequentially.
    """

    @modal.enter()
    def startup(self):
        """Runs ONCE when the container starts. Loads models + FAISS indexes."""
        import logging
        import sys
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("lost-found-ai")

        # Make sure /root is in sys.path so we can import _faiss_store
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")

        # --- Load InsightFace (already in image, no download) ---
        from insightface.app import FaceAnalysis
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.logger.info("✅ InsightFace loaded")

        # --- Load FAISS indexes from Volume ---
        self._load_stores()
        self.logger.info("✅ Startup complete")

    def _load_stores(self):
        """Load all FAISS indexes from the persistent volume."""
        from _faiss_store import FAISSStore

        self.text_store = FAISSStore(
            dimension=3072, index_name="text_index", persist_dir=VOLUME_PATH
        )
        self.image_store = FAISSStore(
            dimension=3072, index_name="image_index", persist_dir=VOLUME_PATH
        )
        self.face_store = FAISSStore(
            dimension=512, index_name="face_index", persist_dir=VOLUME_PATH
        )

    def _commit_volume(self):
        """Commit volume changes so they persist across container restarts.
        Uses the MODULE-LEVEL volume instance (correct pattern).
        """
        faiss_volume.commit()
        self.logger.info("📦 Volume committed")

    def _reload_volume(self):
        """Reload volume to see writes from other containers."""
        faiss_volume.reload()

    # ──────────────────────────────────────
    # WRITE Endpoints
    # ──────────────────────────────────────

    @modal.asgi_app()
    def web(self):
        """Mount the FastAPI app."""
        from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        from typing import List, Dict, Any, Optional
        import numpy as np
        import traceback

        web_app = FastAPI(title="Lost & Found AI — Modal Service")

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # --- Optional API Key auth ---
        API_ACCESS_KEY = os.environ.get("API_ACCESS_KEY", "")

        @web_app.middleware("http")
        async def verify_api_key(request: Request, call_next):
            if request.url.path in ("/health", "/docs", "/openapi.json"):
                return await call_next(request)
            if API_ACCESS_KEY:
                key = request.headers.get("X-API-Key", "")
                if key != API_ACCESS_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"status": "error", "error": "Invalid API key"}
                    )
            return await call_next(request)

        # --- Pydantic models ---
        class AddVectorRequest(BaseModel):
            embedding: List[float]
            metadata: Dict[str, Any]
            index_name: str  # "text", "image", or "face"

        class BatchAddRequest(BaseModel):
            items: List[Dict[str, Any]]  # [{embedding: [...], metadata: {...}}]
            index_name: str

        class SearchVectorRequest(BaseModel):
            embedding: List[float]
            k: int = 5
            index_name: str  # "text", "image", or "face"

        def _get_store(name: str):
            stores = {
                "text": self.text_store,
                "image": self.image_store,
                "face": self.face_store,
            }
            store = stores.get(name)
            if not store:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid index_name '{name}'. Must be: text, image, face"
                )
            return store

        # ──── WRITE: Add single vector ────
        @web_app.post("/add-vector")
        async def add_vector(req: AddVectorRequest):
            try:
                store = _get_store(req.index_name)
                embedding = np.array(req.embedding, dtype=np.float32)

                if embedding.shape[0] != store.dimension:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Embedding dim {embedding.shape[0]} != expected {store.dimension}"
                    )

                idx = store.add(embedding, req.metadata)
                self._commit_volume()

                return {
                    "status": "success",
                    "data": {
                        "message": "Vector added",
                        "index_position": idx,
                        "total_vectors": store.index.ntotal,
                    }
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"add-vector failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        # ──── WRITE: Batch add vectors ────
        @web_app.post("/batch-add-vectors")
        async def batch_add_vectors(req: BatchAddRequest):
            try:
                store = _get_store(req.index_name)
                embeddings = []
                metadatas = []
                for item in req.items:
                    emb = np.array(item["embedding"], dtype=np.float32)
                    if emb.shape[0] != store.dimension:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Embedding dim {emb.shape[0]} != expected {store.dimension}"
                        )
                    embeddings.append(emb)
                    metadatas.append(item["metadata"])

                embeddings_arr = np.stack(embeddings)
                store.add_batch(embeddings_arr, metadatas)
                self._commit_volume()

                return {
                    "status": "success",
                    "data": {
                        "message": f"Added {len(embeddings)} vectors",
                        "total_vectors": store.index.ntotal,
                    }
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"batch-add failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        # ──── WRITE: Add face (InsightFace extract + store) ────
        @web_app.post("/add-face")
        async def add_face(person_id: str = Form(...), image: UploadFile = File(...)):
            try:
                image_bytes = await image.read()
                embedding = self._extract_face(image_bytes)

                if embedding is None:
                    return {
                        "status": "error",
                        "error": "No face detected in the image",
                        "data": None,
                    }

                idx = self.face_store.add(embedding, {"person_id": person_id})
                self._commit_volume()

                return {
                    "status": "success",
                    "data": {
                        "message": "Face added",
                        "person_id": person_id,
                        "index_position": idx,
                    }
                }
            except Exception as e:
                self.logger.error(f"add-face failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        # ──── READ: Search vector ────
        @web_app.post("/search-vector")
        async def search_vector(req: SearchVectorRequest):
            try:
                store = _get_store(req.index_name)
                embedding = np.array(req.embedding, dtype=np.float32)

                if embedding.shape[0] != store.dimension:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Embedding dim {embedding.shape[0]} != expected {store.dimension}"
                    )

                if store.index.ntotal == 0:
                    return {
                        "status": "success",
                        "data": {"results": [], "message": "Index is empty"}
                    }

                results = store.search(embedding, req.k)
                return {"status": "success", "data": {"results": results}}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"search-vector failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        # ──── READ: Face match (InsightFace extract + search) ────
        @web_app.post("/face-match")
        async def face_match(k: int = Form(5), image: UploadFile = File(...)):
            try:
                image_bytes = await image.read()
                embedding = self._extract_face(image_bytes)

                if embedding is None:
                    return {
                        "status": "success",
                        "data": {"results": [], "message": "No face detected"}
                    }

                if self.face_store.index.ntotal == 0:
                    return {
                        "status": "success",
                        "data": {"results": [], "message": "Face index is empty"}
                    }

                results = self.face_store.search(embedding, k)
                return {"status": "success", "data": {"results": results}}
            except Exception as e:
                self.logger.error(f"face-match failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        # ──── Health ────
        @web_app.get("/health")
        async def health():
            return {
                "status": "ok",
                "data": {
                    "text_vectors": self.text_store.index.ntotal,
                    "image_vectors": self.image_store.index.ntotal,
                    "face_vectors": self.face_store.index.ntotal,
                }
            }

        return web_app

    # ──────────────────────────────────────
    # Face Recognition Helper
    # ──────────────────────────────────────
    def _extract_face(self, image_bytes: bytes):
        """Extract 512-dim face embedding from image bytes.
        Returns normalized numpy array or None if no face found.
        """
        import io
        import cv2
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = self.face_app.get(img)
        if len(faces) == 0:
            self.logger.warning("No face detected in image")
            return None

        embedding = faces[0].embedding  # shape (512,)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
