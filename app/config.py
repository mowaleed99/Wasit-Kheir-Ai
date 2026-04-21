
"""
Configuration management for the application.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # تحميل المتغيرات من ملف .env

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "models/embedding-001")
MULTIMODAL_EMBED_MODEL = os.getenv("MULTIMODAL_EMBED_MODEL", "models/multimodalembedding@001")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6))

# أبعاد المتجهات (حسب نموذج Gemini)
TEXT_EMBED_DIM = 3072     # نموذج gemini-embedding-001 يخرج 3072 بُعدًا
IMAGE_EMBED_DIM = 3072    # gemini-embedding-2-preview يخرج 3072 بُعدًا
FACE_EMBED_DIM = 512      # ArcFace يخرج 512 بُعدًا

# إعدادات FAISS
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_indexes")
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)