from typing import List, Tuple
from datasets import load_dataset

def load_ag_news_texts(n_corpus: int, n_queries: int) -> Tuple[List[str], List[str]]:
    ds_train = load_dataset("ag_news", split="train")
    ds_test = load_dataset("ag_news", split="test")

    def to_text(x):
        if isinstance(x, dict):
            if "text" in x: return x["text"]
            return f'{x.get("title","")}. {x.get("description","")}'
        return str(x)

    corpus = [to_text(x) for x in ds_train[:n_corpus]]
    queries = [to_text(x) for x in ds_test[:n_queries]]
    return corpus, queries
