import numpy as np
from app.models.embeddingModel import EmbeddingModel

def testEmbeddingModelEncode():
  text = ["hello world","semantic search"]
  model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
  embs = model.encode(text, batch_size = 2)
  assert embs.shape[0] == len(text)
  assert embs.shape[1] == 384
  assert embs.dtype == np.float32


