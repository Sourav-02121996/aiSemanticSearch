
import os
import sys
import streamlit as st
import faiss
import glob

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set thread environment variables for compatibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from app.config import MODEL_NAME, N_CORPUS, ARTIFACT_DIR
from app.data.dataloader import load_ag_news_texts
from app.models.embeddingModel import EmbeddingModel
from app.models.faissIndexes import normalize

@st.cache_resource
def load_model(model_name):
    """Loads and caches the embedding model."""
    return EmbeddingModel(model_name)

@st.cache_resource
def load_corpus(n_corpus):
    """Loads and caches the corpus texts."""
    corpus_texts, _ = load_ag_news_texts(n_corpus, 0)
    return corpus_texts

@st.cache_resource
def load_faiss_index(index_path):
    """Loads and caches the FAISS index."""
    return faiss.read_index(index_path)

def find_faiss_indexes(artifact_dir):
    """Finds all .faiss files in the artifact directory."""
    return glob.glob(os.path.join(artifact_dir, "*.faiss"))

def main():
    """Streamlit-based UI for semantic search."""
    st.set_page_config(page_title="Semantic Search Engine", layout="wide")
    st.title("🔎 Semantic Search with FAISS")
    st.markdown(
        "This app allows you to perform semantic search on a corpus of news articles. "
        "Select an index, adjust parameters, and type your query below."
    )

    # --- Load resources ---
    with st.spinner("Loading embedding model..."):
        model = load_model(MODEL_NAME)
    with st.spinner(f"Loading {N_CORPUS} corpus documents..."):
        corpus_texts = load_corpus(N_CORPUS)

    # --- Sidebar for configuration ---
    with st.sidebar:
        st.header("Configuration")
        
        index_files = find_faiss_indexes(ARTIFACT_DIR)
        if not index_files:
            st.error(f"No FAISS index files found in `{ARTIFACT_DIR}`. Please run the benchmark first.")
            return
            
        index_paths = {os.path.basename(f): f for f in index_files}
        selected_index_name = st.selectbox(
            "Choose a FAISS Index:",
            options=sorted(index_paths.keys()),
            index=0,
            help="Select the pre-built FAISS index to query against."
        )
        
        k_value = st.slider(
            "Top-K Results:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top results to retrieve."
        )

    selected_index_path = index_paths[selected_index_name]
    with st.spinner(f"Loading index: {selected_index_name}..."):
        index = load_faiss_index(selected_index_path)

    st.info(f"**Ready to search!** Using index `{selected_index_name}` with `{len(corpus_texts)}` documents.")

    # --- Search query input ---
    query = st.text_input("Enter your search query:", "")

    if query:
        # --- Perform search ---
        q_emb = model.encode([query])
        q_emb = normalize(q_emb)
        
        D, I = index.search(q_emb, k_value)
        
        # --- Display results ---
        st.subheader("Top Results")
        if not I.size:
            st.warning("No results found.")
            return

        results = zip(I[0], D[0])
        for rank, (doc_id, score) in enumerate(results, start=1):
            with st.container():
                st.markdown(f"**{rank}. Score: `{score:.4f}`** (Document ID: `{doc_id}`)")
                text = corpus_texts[int(doc_id)]
                st.markdown(f"> {text}")
                st.divider()

if __name__ == "__main__":
    main()
