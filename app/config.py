import os

SEED = 42

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
N_CORPUS = 120000
N_QUERIES = 2000

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(ARTIFACT_DIR, "results.csv")

IVF_FLAT_NLIST = [256, 512, 1024, 2048]
IVF_FLAT_NPROBE = [1, 4, 8, 16, 32]
IVF_PQ_NLIST = [512, 1024, 2048]
IVF_PQ_NPROBE = [4, 8, 16, 32, 64]
