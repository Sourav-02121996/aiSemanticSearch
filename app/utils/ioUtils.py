import os, json, pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class RunResult:
    index_name: str
    params: Dict
    ms_per_query: float
    recall_at_k: float
    index_mb: float

def save_results_csv(results, path: str):
    df = pd.DataFrame([asdict(r) for r in results])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df
