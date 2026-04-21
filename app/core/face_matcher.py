"""
Face recognition using InsightFace (ArcFace).
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from typing import Optional, List
import io
from PIL import Image

class FaceRecognizer:
    """Face embedding extraction using InsightFace."""
    
    def __init__(self):
        # تهيئة InsightFace مع النموذج المناسب
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face embedding from image bytes.
        Returns 512-dim embedding or None if no face detected.
        """
        # تحويل البايتات إلى صورة numpy
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        
        # نأخذ أول وجه يتم اكتشافه (أكبر وجه غالبًا)
        face = faces[0]
        embedding = face.embedding  # numpy array of shape (512,)
        # تطبيع المتجه
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)