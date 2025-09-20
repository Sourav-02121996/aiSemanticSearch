import pytest
from app.data import dataloader

class FakeSplit:
    def __init__(self, texts):
        self._texts = list(texts)

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return {"text": self._texts[idx]}
        return {"text": self._texts[idx]}

def test_load_ag_news_texts_returns_expected(monkeypatch):
    train_texts = [f"doc{i}" for i in range(10)]
    test_texts  = [f"query{i}" for i in range(5)]

    def fake_load_dataset(name, split):
        assert name == "ag_news"
        return FakeSplit(train_texts) if split == "train" else FakeSplit(test_texts)

    monkeypatch.setattr(dataloader, "load_dataset", fake_load_dataset)

    corpus, queries = dataloader.load_ag_news_texts(n_corpus=5, n_queries=3)

    assert corpus == ["doc0", "doc1", "doc2", "doc3", "doc4"]
    assert queries == ["query0", "query1", "query2"]

def test_load_ag_news_texts_caps_to_available(monkeypatch):
    train_texts = [f"doc{i}" for i in range(3)]
    test_texts  = [f"query{i}" for i in range(2)]

    def fake_load_dataset(name, split):
        return FakeSplit(train_texts) if split == "train" else FakeSplit(test_texts)

    monkeypatch.setattr(dataloader, "load_dataset", fake_load_dataset)

    corpus, queries = dataloader.load_ag_news_texts(n_corpus=10, n_queries=10)

    assert corpus == ["doc0", "doc1", "doc2"]   # capped to available
    assert queries == ["query0", "query1"]      # capped to available

def test_load_ag_news_texts_handles_zero(monkeypatch):
    train_texts = [f"doc{i}" for i in range(4)]
    test_texts  = [f"query{i}" for i in range(3)]

    def fake_load_dataset(name, split):
        return FakeSplit(train_texts) if split == "train" else FakeSplit(test_texts)

    monkeypatch.setattr(dataloader, "load_dataset", fake_load_dataset)

    corpus, queries = dataloader.load_ag_news_texts(n_corpus=0, n_queries=0)

    assert corpus == []
    assert queries == []
