from typing import List, Tuple
from datasets import load_dataset

def load_ag_news_texts(n_corpus: int, n_queries: int):
    ds_train = load_dataset("ag_news", split="train")
    ds_test  = load_dataset("ag_news", split="test")

    n_corpus = min(n_corpus, len(ds_train))
    n_queries = min(n_queries, len(ds_test))

    corpus = ds_train[:n_corpus]["text"]
    queries = ds_test[:n_queries]["text"]
    return corpus, queries

