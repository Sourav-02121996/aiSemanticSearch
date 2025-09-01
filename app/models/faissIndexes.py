import numpy as np
import faiss
from typing import Tuple
import os
def normalize(vecs: np.ndarray) -> np.ndarray:
  vecs = vecs.astype(np.float32, copy=False)
  faiss.normalize_L2(vecs)
  return vecs

def buildFlatIP(embs: np.ndarray) -> np.ndarray:
  d = embs.shape[1]
  idx = faiss.IndexFlat(d)
  idx.add(normalize(embs.copy()))
  return idx

def search(index: faiss.Index, query_vecs: np.ndarray, k: int ) -> Tuple[np.ndarray, np.ndarray]:
  q = normalize(query_vecs.copy())
  D, I = index.search(q,k)
  return D, I

def saveIndex(index:faiss.Index, path: str)->None:
  faiss.write_index(index, path)

def indexSizeMb(path: str)-> float:
  return os.path.getsize(path) / (1024 * 1024)


