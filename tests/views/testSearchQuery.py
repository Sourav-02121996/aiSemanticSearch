import sys
import numpy as np
import pytest
import app.searchQuery as sq


def test_requires_index_arg(monkeypatch):
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(sys, "argv", ["prog"])  # no args

    with pytest.raises(SystemExit) as excinfo:
        sq.main()
    assert excinfo.value.code == 2


def test_interactive_query_happy_path(monkeypatch, capsys):

    #Fake dataset loader (no downloads)
    def fake_loader(n_corpus, _n_queries):
        # 3 docs so returned ids 1 and 0 are valid
        corpus = ["alpha document", "beta document", "gamma document"]
        return corpus, []
    monkeypatch.setattr(sq, "load_ag_news_texts", fake_loader)

    #Fake embedding model (no transformers/torch)
    class DummyModel:
        def __init__(self, model_name):  # keep signature
            self.model_name = model_name
        def encode(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(sq, "EmbeddingModel", DummyModel)

    monkeypatch.setattr(sq, "normalize", lambda x: x)

    #Fake FAISS index and read_index()
    class FakeIndex:
        def search(self, q_emb, k):
            I = np.array([[1, 0]], dtype=np.int64)
            D = np.array([[0.9, 0.8]], dtype=np.float32)
            return D[:, :k], I[:, :k]

    def fake_read_index(path):
        assert isinstance(path, str) and len(path) > 0
        return FakeIndex()

    monkeypatch.setattr(sq.faiss, "read_index", fake_read_index)

    monkeypatch.setattr(sys, "argv", ["prog", "--index", "artifacts/dummy.faiss", "--k", "2"])

    inputs = iter(["test query", ""])  # one query, then exit
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    sq.main()
    out = capsys.readouterr().out

    assert "Loaded corpus texts: 3" in out
    assert "Loading index: artifacts/dummy.faiss" in out
    assert "Type a query" in out
    assert "Top results:" in out

    assert "beta document" in out
    assert "alpha document" in out
    assert "score=0.9000" in out or "score=0.900" in out
    assert "score=0.8000" in out or "score=0.800" in out
