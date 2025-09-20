import numpy as np
from app.models.embeddingModel import EmbeddingModel
from app.models.faissIndexes import normalize

def testEmbeddingModelEncode():
  x = np.array([[3, 4],[1, 0,],],dtype=np.float32)
  y = normalize(x.copy())
  norms = np.sqrt((y * y).sum(axis = 1))
  assert np.allclose(norms, np.ones_like(norms))



