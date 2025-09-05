# app/search_query.py
# Run with:  python -m app.search_query --index artifacts/ivf_flat_nlist1024_nprobe16.faiss --k 5
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import faiss
import numpy as np

from app.config import MODEL_NAME, N_CORPUS
from app.data.dataloader import load_ag_news_texts
from app.models.embeddingModel import EmbeddingModel
from app.models.faissIndexes import normalize

def main():
    parser = argparse.ArgumentParser(description="Interactive semantic search against a saved FAISS index.")
    parser.add_argument("--index", required=True, help="Path to .faiss file under artifacts/")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to show")
    args = parser.parse_args()

    # 1) Load corpus texts in the SAME order used for benchmarking
    corpus_texts, _ = load_ag_news_texts(N_CORPUS, 0)
    print(f"Loaded corpus texts: {len(corpus_texts)}")

    # 2) Load FAISS index + embedding model
    print(f"Loading index: {args.index}")
    index = faiss.read_index(args.index)
    model = EmbeddingModel(MODEL_NAME)

    # 3) REPL loop: get a query, embed, normalize, search, print results
    print("\nType a query (or just press Enter to quit).")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break
        q_emb = model.encode([q])                     # shape: (1, d)
        q_emb = normalize(q_emb)
        D, I = index.search(q_emb, args.k)            # I: (1, k) doc ids; D: (1, k) scores (cosine sim)
        I = I[0]; D = D[0]

        print("\nTop results:")
        for rank, (doc_id, score) in enumerate(zip(I, D), start=1):
            text = corpus_texts[int(doc_id)]
            snippet = (text[:220] + "…") if len(text) > 220 else text
            print(f"{rank:>2}. [id={int(doc_id)}  score={float(score):.4f}]  {snippet}")

if __name__ == "__main__":
    main()
