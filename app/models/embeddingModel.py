from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingModel:
  def __init__(self, model_name: str):
    self.model = SentenceTransformer(model_name)

  def encode(self, text: List[str], batch_size: int = 256) -> np.ndarray:
    embs = self.model.encode(
        text,
        batch_size = batch_size,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return embs.astype(np.float32)
