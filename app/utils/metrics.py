import time, numpy as np
from typing import Tuple

def time_ms_per_query(fn, *args, **kwargs) -> Tuple[float, tuple]:
    t0 = time.time()
    out = fn(*args, **kwargs)
    t1 = time.time()
    q = args[1].shape[0] if len(args) > 1 else 1
    ms_per_query = (t1 - t0) * 1000.0 / q
    return ms_per_query, out

def recall_at_k(approx_I: np.ndarray, exact_I: np.ndarray, k: int) -> float:
    assert approx_I.shape == exact_I.shape
    hits = 0
    total = approx_I.shape[0] * k
    for i in range(approx_I.shape[0]):
        hits += len(set(approx_I[i]).intersection(set(exact_I[i])))
    return hits / total
