import os, json, pandas as pd
from dataclasses import dataclass, asdict, fields  # <- add fields
from typing import Dict

@dataclass
class RunResult:
    index_name: str
    params: Dict
    ms_per_query: float
    recall_at_k: float
    index_mb: float

def save_results_csv(results, path: str):
    if results:
        df = pd.DataFrame([asdict(r) for r in results])
    else:
        # Creating an empty DF with the RunResult schema as columns
        cols = [f.name for f in fields(RunResult)]
        df = pd.DataFrame(columns=cols)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

def test_save_results_csv_handles_empty_list(tmp_path):
    # Arranging empty results list
    out_csv = tmp_path / "results_empty.csv"

    df_returned = save_results_csv([], str(out_csv))

    assert out_csv.exists()

    df_disk = pd.read_csv(out_csv)
    expected_cols = ["index_name", "params", "ms_per_query", "recall_at_k", "index_mb"]
    assert list(df_disk.columns) == expected_cols, "CSV should have the dataclass-derived headers"
    assert len(df_disk) == 0, "CSV should be empty when no results are provided"

    assert list(df_returned.columns) == expected_cols
    assert len(df_returned) == 0
