import numpy as np
import faiss
from typing import Tuple, Optional
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

def pickMforPq(d: int) -> int:
  for m in [16, 24, 32, 48, 64]:
    if m <= d and d % m == 0:
      return m
  for m in range(8, d + 1):
    if d % m == 0:
      return m
  return min(32, d)

def build_ivf_flat(embs: np.ndarray, nlist: int) -> faiss.Index:
  d = embs.shape[1]
  quantizer = faiss.IndexFlatIP(d)
  idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
  train_n = min(50_000, embs.shape[0])
  idx.train(normalize(embs[:train_n].copy()))
  idx.add(normalize(embs.copy()))
  return idx


def build_ivf_pq(embs: np.ndarray, nlist: int, m: Optional[int] = None,
    nbits: int = 8, use_opq: bool = False) -> faiss.Index:
  d = embs.shape[1]
  if m is None:
    m = pickMforPq(d)
  quantizer = faiss.IndexFlatIP(d)
  base = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits,
                          faiss.METRIC_INNER_PRODUCT)
  train_n = min(50_000, embs.shape[0])
  train_vecs = normalize(embs[:train_n].copy())
  base.train(train_vecs)
  if use_opq:
    opq = faiss.OPQMatrix(d, m)
    opq.train(train_vecs)
    idx = faiss.IndexPreTransform(opq, base)
    idx.add(normalize(embs.copy()))
    return idx
  else:
    base.add(normalize(embs.copy()))
    return base


