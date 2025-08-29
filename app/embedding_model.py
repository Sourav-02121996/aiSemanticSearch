from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingModel:
  def __init__(self, model_name: str):
    self.model = SentenceTransformer(model_name)

  def encode(self, text: List[str]) -> np.ndarray:
    embs = self.model.encode(text)
    return embs.astype(np.float32)
