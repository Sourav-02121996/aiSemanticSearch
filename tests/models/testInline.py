import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    faiss.normalize_L2(x)
    return x

def main():
    # 1) Load a small AG News sample
    ds_train = load_dataset("ag_news", split="train")
    corpus_texts = ds_train["text"][:5000] # 5k rows

    # 2) Encode corpus
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus_embs = model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    # 3) Build FlatIP exact index
    d = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(normalize(corpus_embs.copy()))

    # 4) Query
    user_query = "new apple models"
    q = model.encode([user_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    D, I = index.search(q, k=5)

    # 5) Print results
    print(f"\nQuery: {user_query!r}")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        snippet = corpus_texts[idx].replace("\n", " ")
        if len(snippet) > 140: snippet = snippet[:140] + "…"
        print(f"{rank}. score={score:.3f} | {snippet}")

if __name__ == "__main__":
    main()
