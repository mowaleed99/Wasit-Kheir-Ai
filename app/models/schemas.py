"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TextEmbeddingRequest(BaseModel):
    text: str

class ImageEmbeddingRequest(BaseModel):
    # سيتم التعامل مع الملفات عبر Form، لذا هذا مجرد توثيق
    pass

class AddTextRequest(BaseModel):
    text: str
    post_id: str

class AddImageRequest(BaseModel):
    post_id: str

class SearchRequest(BaseModel):
    k: int = 5

class MultimodalSearchRequest(BaseModel):
    text: Optional[str] = None
    k: int = 5

class SearchResponse(BaseModel):
    status: str
    data: Dict[str, Any]