"""
Core capabilities for interaction with Google Gemini models.
"""
import io
from typing import List
import numpy as np
import google.generativeai as genai
import PIL.Image

from app.config import GEMINI_API_KEY, TEXT_EMBED_MODEL, MULTIMODAL_EMBED_MODEL

class GeminiEmbedder:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.text_model = TEXT_EMBED_MODEL
        self.multimodal_model = MULTIMODAL_EMBED_MODEL

    def _normalize(self, embedding: List[float]) -> List[float]:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """نص فقط - يستخدم نموذج gemini-embedding-001"""
        result = genai.embed_content(
            model=self.text_model,
            content=text,
            task_type="retrieval_document"
        )
        embedding = result['embedding'] if isinstance(result, dict) else result
        return self._normalize(embedding)

    def get_image_embedding(self, image_bytes: bytes) -> List[float]:
        """صورة فقط - يستخدم نموذج gemini-embedding-2-preview"""
        image = PIL.Image.open(io.BytesIO(image_bytes))
        # النموذج المتعدد الوسائط يستقبل صورة ويعيد embedding
        result = genai.embed_content(
            model=self.multimodal_model,
            content=image,
            task_type="retrieval_document"
        )
        embedding = result['embedding'] if isinstance(result, dict) else result
        return self._normalize(embedding)