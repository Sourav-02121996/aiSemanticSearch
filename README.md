# AI Semantic Search – Compression & Optimization with FAISS

This project implements a semantic search pipeline with a modular MVC architecture.  
It benchmarks exact search vs approximate nearest neighbor (ANN) methods  using FAISS,  
and provides an interactive CLI to query saved indexes.

## Features

- Semantic embeddings with Sentence-Transformers.
- Indexing via FAISS:
  - FlatIP (exact baseline)
  - IVF-Flat (coarse quantization)
  - IVF-PQ (+ optional OPQ compression)
- Benchmarking framework**: latency, recall@k, and memory footprint
- CSV results** + saved FAISS indexes
- Interactive CLI for live querying
- MVC-style architecture for clean separation of concerns

---

## Project Structure
```
aiSemanticSearch/
├─ app/
│  ├─ controllers/
│  │  └─ benchMarkController.py      # orchestrates benchmark pipeline
│  ├─ data/
│  │  └─ dataloader.py               # loads AG News dataset
│  ├─ models/
│  │  ├─ embeddingModel.py           # wraps Sentence-Transformers
│  │  └─ faissIndexes.py             # FAISS index builders & search
│  ├─ utils/
│  │  ├─ ioUtils.py                  # RunResult dataclass, CSV writer
│  │  └─ metrics.py                  # timing + recall@k helpers
│  ├─ views/
│  │  └─ cliView.py                  # CLI output (pretty tables)
│  ├─ config.py                      # parameters & paths
│  ├─ main.py                        # run benchmarks
│  └─ searchQuery.py                 # interactive query CLI
├─ artifacts/                        # generated indexes + results.csv
├─ tests/                            # unit tests
└─ README.md
```

---

## Installation

1. Clone repo & create venv:
   ```bash
   git clone <repo-url>
   cd aiSemanticSearch
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install sentence-transformers faiss-cpu datasets numpy pandas tabulate tqdm psutil matplotlib pytest
   ```

   > If FAISS fails, try:  
   > `pip install faiss-cpu==1.7.4`

---

### Run Benchmark
Build embeddings, train indexes, and evaluate performance:
```bash
python -m app.main
```
Outputs:
- FAISS indexes in `artifacts/*.faiss`
- Results in `artifacts/results.csv`

### Query a Saved Index
```bash
python -m app.searchQuery --index artifacts/ivf_flat_nlist1024_nprobe16.faiss --k 5
```

Example:
```
Query> Apple launches new iPhone with better camera

Top results:
 1. [id=9321  score=0.8123]  Apple unveils the new iPhone with improved optics…
 2. [id=12044 score=0.8077]  New smartphone release highlights camera upgrades…
```

---

## Configuration

Edit `app/config.py`:

- Dataset size: `N_CORPUS`, `N_QUERIES`
- Retrieval: `TOP_K`
- IVF sweeps:
  ```python
  IVF_FLAT_NLIST = [256, 512, 1024, 2048]
  IVF_FLAT_NPROBE = [1, 4, 8, 16, 32]
  IVF_PQ_NLIST = [512, 1024, 2048]
  IVF_PQ_NPROBE = [4, 8, 16, 32, 64]
  ```

---

## Results

The benchmarking script logs metrics into `results.csv`:

| Index       | Params                         | ms/query | Recall@10 | Size (MB) |
|-------------|--------------------------------|----------|-----------|-----------|
| FlatIP      | {}                             | 12.8     | 1.000     | 175.8     |
| IVF-Flat    | {"nlist":1024,"nprobe":16}     | 0.27     | 0.971     | 178.2     |
| IVF-PQ+OPQ  | {"nlist":1024,"m":16,"nprobe":32} | 0.05  | 0.519     | 4.6       |

Interpretation:
- **FlatIP** = perfect recall, heavy memory, slower
- **IVF-Flat** = near-perfect recall, faster, similar size
- **IVF-PQ** = tiny memory footprint, blazing fast, lower recall

---

## Tests

Run all tests:
```bash
pytest tests/
```
## Testing

- This repo uses pytest with lightweight fakes/stubs and monkeypatch so tests are fast and deterministic (no downloads, no native FAISS required).
- What I tested
- metrics.py — timing math & recall@k
- ioUtils.py — CSV writing (including empty results)
- dataloader.py — slicing/limits via a fake HF dataset
- searchQuery.py — interactive CLI flow (fake index + input)
- benchMarkController.py — end-to-end smoke with fake data/model/index

## Tools & patterns

- pytest — simple test runner
- monkeypatch — swap functions/classes at runtime (e.g., load_ag_news_texts, EmbeddingModel, FAISS builders)
- fakes/stubs — tiny classes like _FakeIndex that mimic .search()
- capsys — capture and assert CLI output
---

## Troubleshooting

- Argparse says `--index` is required 
  Run `searchQuery.py` without arguments. Pass `--index` (relative to project root) or modify the script to auto-pick the newest file from `artifacts/`.

- FAISS: could not open artifacts  
  Wrong working directory. In PyCharm, set the working directory to the project root; in Terminal, run from the root. Or pass an absolute path to `--index`.

- OMP: pthread_mutex_init failed`
  Set the environment variables at the top of scripts (see Installation step 3) and use `faiss.omp_set_num_threads(1)`.

- FAISS training error `nx >= k`  
  Your `nlist` is larger than the number of available training vectors. Increase `N_CORPUS` or shrink `nlist` (e.g., `[1, 2]`) for tiny runs.

- Slow first run  
  Model & dataset are downloaded once subsequent runs are faster.

---

## Skills Demonstrated

- Python, FAISS, Sentence-Transformers  
- Approximate Nearest Neighbor (ANN) search  
- MVC architecture** & modular design  
- Benchmarking & Evaluation: latency, recall, memory  
- NumPy, Pandas, Matplotlib  
- PyTest unit testing

## Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
