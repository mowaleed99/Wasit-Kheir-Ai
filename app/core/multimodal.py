"""
Multimodal search combining text and image embeddings.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from app.core.vector_store import VectorStore

class MultimodalMatcher:
    """Combine text and image search results with weighted scoring."""
    
    def __init__(self, text_store: VectorStore, image_store: VectorStore, text_weight: float = 0.5):
        self.text_store = text_store
        self.image_store = image_store
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
    
    def search(self, text_embedding: Optional[np.ndarray], image_embedding: Optional[np.ndarray], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform multimodal search.
        If both are provided, combine scores with weights.
        If only one is provided, return results from that modality.
        """
        if text_embedding is None and image_embedding is None:
            return []
        
        # حالة النص فقط
        if text_embedding is not None and image_embedding is None:
            return self.text_store.search(text_embedding, k)
        
        # حالة الصورة فقط
        if image_embedding is not None and text_embedding is None:
            return self.image_store.search(image_embedding, k)
        
        # حالة الاثنين معًا
        text_results = self.text_store.search(text_embedding, k * 2)
        image_results = self.image_store.search(image_embedding, k * 2)
        
        # دمج النتائج باستخدام قاموس لتجميع الدرجات حسب post_id
        combined = {}
        
        for res in text_results:
            pid = res['metadata'].get('post_id')
            if pid:
                combined[pid] = {
                    'score': res['score'] * self.text_weight,
                    'metadata': res['metadata']
                }
        
        for res in image_results:
            pid = res['metadata'].get('post_id')
            if pid:
                if pid in combined:
                    combined[pid]['score'] += res['score'] * self.image_weight
                else:
                    combined[pid] = {
                        'score': res['score'] * self.image_weight,
                        'metadata': res['metadata']
                    }
        
        # تحويل إلى قائمة وترتيب تنازلي حسب الدرجة
        results_list = list(combined.values())
        results_list.sort(key=lambda x: x['score'], reverse=True)
        return results_list[:k]