from tabulate import tabulate
import json

def print_table(results):
    rows = []
    for r in results:
        rows.append([
            r.index_name,
            json.dumps(r.params),
            f"{r.ms_per_query:.2f}",
            f"{r.recall_at_k:.3f}",
            f"{r.index_mb:.1f}",
        ])
    print("\n=== Results ===")
    print(tabulate(rows, headers=["Index","Params","ms/query","Recall@k","Size (MB)"], tablefmt="github"))