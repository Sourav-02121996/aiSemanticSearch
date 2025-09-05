import os, json
from typing import List
import numpy as np


from app.config import (
    MODEL_NAME, TOP_K, N_CORPUS, N_QUERIES, ARTIFACT_DIR, RESULTS_CSV,
    IVF_FLAT_NLIST, IVF_FLAT_NPROBE, IVF_PQ_NLIST, IVF_PQ_NPROBE,
)

from app.data.dataloader import load_ag_news_texts
from app.models.embeddingModel import EmbeddingModel
from app.models import faissIndexes as FX
from app.utils.ioUtils import RunResult, save_results_csv
from app.utils.metrics import time_ms_per_query,recall_at_k


def runBenchmark(viewPrint):
  corpus_texts, query_texts = load_ag_news_texts(N_CORPUS, N_QUERIES)
  print(f"Corpus: {len(corpus_texts)} | Queries: {len(query_texts)}")

  model = EmbeddingModel(MODEL_NAME)
  corpus_embs = model.encode(corpus_texts)
  query_embs = model.encode(query_texts)

  np.save(os.path.join(ARTIFACT_DIR, "corpus.npy"), corpus_embs)
  np.save(os.path.join(ARTIFACT_DIR, "queries.npy"), query_embs)

  exact_idx = FX.buildFlatIP(corpus_embs)
  ms_exact, (D_exact, I_exact) = time_ms_per_query(FX.search, exact_idx,
                                                   query_embs, TOP_K)
  exact_path = os.path.join(ARTIFACT_DIR, "flatip.faiss")
  FX.saveIndex( exact_idx, exact_path)
  size_exact = FX.indexSizeMb(exact_path)

  results: List[RunResult] = [
    RunResult("FlatIP (exact)", {}, ms_exact, 1.0, size_exact)
  ]

  for nlist in IVF_FLAT_NLIST:
    idx = FX.build_ivf_flat(corpus_embs, nlist)
    for nprobe in IVF_FLAT_NPROBE:
      idx.nprobe = nprobe
      ms, (D, I) = time_ms_per_query(FX.search, idx, query_embs, TOP_K)
      path = os.path.join(ARTIFACT_DIR,
                          f"ivf_flat_nlist{nlist}_nprobe{nprobe}.faiss")
      FX.saveIndex(idx, path)
      sz = FX.indexSizeMb(path)
      rec = recall_at_k(I, I_exact, TOP_K)
      results.append(
        RunResult("IVF-Flat", {"nlist": nlist, "nprobe": nprobe}, ms, rec, sz))

  d = corpus_embs.shape[1]
  m = FX.pickMforPq(d)
  for nlist in IVF_PQ_NLIST:
    for use_opq in [False, True]:
      idx = FX.build_ivf_pq(corpus_embs, nlist=nlist, m=m, nbits=8,
                            use_opq=use_opq)
      base = idx if hasattr(idx, "nprobe") else FX.faiss.downcast_index( idx.index)
      for nprobe in IVF_PQ_NPROBE:
        if hasattr(base, "nprobe"):
          base.nprobe = nprobe
        ms, (D, I) = time_ms_per_query(FX.search, idx, query_embs, TOP_K)
        path = os.path.join(ARTIFACT_DIR,
                            f"ivfpq_nlist{nlist}_m{m}_opq{int(use_opq)}_nprobe{nprobe}.faiss")
        FX.saveIndex(idx, path)
        sz = FX.indexSizeMb(path)
        rec = recall_at_k(I, I_exact, TOP_K)
        results.append(RunResult("IVF-PQ" + ("+OPQ" if use_opq else ""),
                                 {"nlist": nlist, "m": m, "nprobe": nprobe}, ms,
                                 rec, sz))

  df = save_results_csv(results, RESULTS_CSV)
  viewPrint(results)
  print(f"\nSaved metrics to: {RESULTS_CSV}")

