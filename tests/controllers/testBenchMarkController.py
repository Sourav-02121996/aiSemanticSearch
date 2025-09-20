import os
import numpy as np
import pandas as pd

from app.controllers import benchMarkController as ctrl


class _FakeIndex:
    def __init__(self):
        self.nprobe = 1
    def search(self, q_emb, k):
        qn = q_emb.shape[0]
        I = np.tile(np.arange(k, dtype=np.int64), (qn, 1))
        D = np.tile(np.linspace(1.0, 0.5, k, dtype=np.float32), (qn, 1))
        return D, I


def _apply_fast_fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(ctrl, "ARTIFACT_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(ctrl, "RESULTS_CSV", os.path.join(str(tmp_path), "results.csv"), raising=False)
    monkeypatch.setattr(ctrl, "N_CORPUS", 8, raising=False)
    monkeypatch.setattr(ctrl, "N_QUERIES", 3, raising=False)
    monkeypatch.setattr(ctrl, "TOP_K", 2, raising=False)
    monkeypatch.setattr(ctrl, "IVF_FLAT_NLIST", [1], raising=False)
    monkeypatch.setattr(ctrl, "IVF_FLAT_NPROBE", [1], raising=False)
    monkeypatch.setattr(ctrl, "IVF_PQ_NLIST", [], raising=False)
    monkeypatch.setattr(ctrl, "IVF_PQ_NPROBE", [], raising=False)

    #Fake dataset loader used by controller
    def _fake_loader(n_corpus, n_queries):
        corpus = [f"doc {i}" for i in range(n_corpus)]
        queries = [f"query {i}" for i in range(n_queries)]
        return corpus, queries
    monkeypatch.setattr(ctrl, "load_ag_news_texts", _fake_loader, raising=True)

    #Fake embedding model used by controller
    class _DummyEmbeddingModel:
        def __init__(self, model_name): pass
        def encode(self, texts):
            rng = np.random.default_rng(0)
            return rng.standard_normal((len(texts), 8)).astype(np.float32)
    monkeypatch.setattr(ctrl, "EmbeddingModel", _DummyEmbeddingModel, raising=True)

    #Fake FAISS helpers via the FX module bound in controller
    monkeypatch.setattr(ctrl.FX, "buildFlatIP", lambda embs: _FakeIndex(), raising=True)
    monkeypatch.setattr(ctrl.FX, "search", lambda index, q, k: index.search(q, k), raising=True)
    monkeypatch.setattr(ctrl.FX, "build_ivf_flat", lambda embs, nlist: _FakeIndex(), raising=True)
    monkeypatch.setattr(ctrl.FX, "build_ivf_pq", lambda embs, nlist, m, nbits, use_opq: _FakeIndex(), raising=True)
    monkeypatch.setattr(ctrl.FX, "pickMforPq", lambda d: 2, raising=True)

    def _save_index(idx, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"idx")
    def _index_size_mb(path):
        return os.path.getsize(path) / (1024 * 1024)

    monkeypatch.setattr(ctrl.FX, "saveIndex", _save_index, raising=True)
    monkeypatch.setattr(ctrl.FX, "indexSizeMb", _index_size_mb, raising=True)

    class _FakeFaissNS:
        @staticmethod
        def downcast_index(x): return x
    monkeypatch.setattr(ctrl.FX, "faiss", _FakeFaissNS(), raising=False)


def test_run_benchmark_creates_results_and_indexes(tmp_path, monkeypatch):
    _apply_fast_fakes(monkeypatch, tmp_path)

    captured = {}
    def _view(results):
        captured["results"] = results

    ctrl.runBenchmark(viewPrint=_view)

    assert os.path.exists(ctrl.RESULTS_CSV)
    df = pd.read_csv(ctrl.RESULTS_CSV)
    expected_cols = ["index_name", "params", "ms_per_query", "recall_at_k", "index_mb"]
    assert list(df.columns) == expected_cols
    assert len(df) >= 1
    assert "FlatIP (exact)" in set(df["index_name"])
    faiss_files = [p for p in os.listdir(ctrl.ARTIFACT_DIR) if p.endswith(".faiss")]
    assert len(faiss_files) >= 1

    assert "results" in captured and len(captured["results"]) >= 1


def test_saves_numpy_arrays(tmp_path, monkeypatch):
    _apply_fast_fakes(monkeypatch, tmp_path)

    ctrl.runBenchmark(viewPrint=lambda _r: None)

    assert os.path.exists(os.path.join(ctrl.ARTIFACT_DIR, "corpus.npy"))
    assert os.path.exists(os.path.join(ctrl.ARTIFACT_DIR, "queries.npy"))
