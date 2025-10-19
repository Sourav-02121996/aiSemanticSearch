
import os
import requests
from tqdm import tqdm

ARTIFACT_FILES = {
    "corpus.npy": "https://your-url.com/corpus.npy",
    "queries.npy": "https://your-url.com/queries.npy",
    "ivf_flat_nlist1024_nprobe16.faiss": "https://your-url.com/index.faiss",
    # Add more files as needed
}

ARTIFACT_DIR = "/mnt/disk/artifacts" if os.path.exists("/mnt/disk") else "artifacts"

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def setup_artifacts():
    """Download artifacts if they don't exist."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    for filename, url in ARTIFACT_FILES.items():
        dest_path = os.path.join(ARTIFACT_DIR, filename)
        if not os.path.exists(dest_path):
            print(f"Downloading {filename}...")
            download_file(url, dest_path)
        else:
            print(f"✓ {filename} already exists")

if __name__ == "__main__":
    setup_artifacts()
