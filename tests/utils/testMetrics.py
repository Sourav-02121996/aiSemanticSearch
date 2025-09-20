import numpy as np
import math
import pytest

from app.utils import metrics

def test_time_ms_per_query_divides_by_num_queries(monkeypatch):
    # Fake time: t0=100.0s, t1=100.1s
    calls = [100.0, 100.1]
    monkeypatch.setattr(metrics.time, "time", lambda: calls.pop(0))

    # Dummy function returns
    def dummy_search(_index, queries, _k):
        return ("D", "I")

    queries = np.zeros((2, 3), dtype=np.float32)  # 2 queries
    ms_per_q, out = metrics.time_ms_per_query(dummy_search, None, queries, 5)

    assert math.isclose(ms_per_q, 50.0, rel_tol=1e-3, abs_tol=1e-3)
    assert out == ("D", "I")


def test_time_ms_per_query_single_arg_function(monkeypatch):
    calls = [10.0, 10.05]
    monkeypatch.setattr(metrics.time, "time", lambda: calls.pop(0))

    def only_one_arg(x):
        return x * 2
    ms_per_q, out = metrics.time_ms_per_query(only_one_arg, 7)
    assert math.isclose(ms_per_q, 50.0, rel_tol=1e-3, abs_tol=1e-3)
    assert out == 14

def test_recall_at_k_perfect_match():
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    approx = np.array([[1, 2, 3], [4, 5, 6]])
    r = metrics.recall_at_k(approx, exact, k=3)
    assert math.isclose(r, 1.0)


def test_recall_at_k_partial_match():
    exact = np.array([[4, 7, 1], [5, 0, 3]])
    approx = np.array([[4, 2, 1], [9, 0, 3]])
    r = metrics.recall_at_k(approx, exact, k=3)
    assert math.isclose(r, 4/6, rel_tol=1e-6)


def test_recall_at_k_no_match():
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    approx = np.array([[10, 11, 12], [13, 14, 15]])
    r = metrics.recall_at_k(approx, exact, k=3)
    assert math.isclose(r, 0.0)


def test_recall_at_k_shape_mismatch_asserts():
    exact = np.array([[1, 2, 3]])
    approx = np.array([[1, 2]])
    with pytest.raises(AssertionError):
        _ = metrics.recall_at_k(approx, exact, k=3)
