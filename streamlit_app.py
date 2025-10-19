
import streamlit as st
import faiss
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.models.embeddingModel import EmbeddingModel
from app.models.faissIndexes import normalize
from app.data.dataloader import load_ag_news_texts
from app.config import MODEL_NAME, N_CORPUS

# Configuration
ARTIFACTS_DIR = "artifacts"
DEFAULT_INDEX = "ivf_flat_nlist1024_nprobe16.faiss"
DEFAULT_K = 5

# Set page config
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Cache functions to load heavy resources only once
@st.cache_resource
def load_model():
    """Load the sentence transformer model."""
    return EmbeddingModel(MODEL_NAME)

@st.cache_resource
def load_corpus():
    """Load the corpus texts."""
    corpus_texts, _ = load_ag_news_texts(N_CORPUS, 0)
    return corpus_texts

@st.cache_resource
def load_index(index_name):
    """Load a FAISS index."""
    index_path = os.path.join(ARTIFACTS_DIR, index_name)
    if not os.path.exists(index_path):
        st.error(f"Index file not found: {index_path}")
        return None
    return faiss.read_index(index_path)

def get_available_indexes():
    """Get list of available FAISS indexes."""
    if not os.path.exists(ARTIFACTS_DIR):
        return []
    return [f for f in os.listdir(ARTIFACTS_DIR) if f.endswith('.faiss')]

# Main app
def main():
    st.title("🔍 Semantic Search Engine")
    st.markdown("""
    Search through 120,000 news articles using semantic similarity.
    Powered by Sentence-Transformers and FAISS.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Index selection
        available_indexes = get_available_indexes()
        if not available_indexes:
            st.error("No FAISS indexes found in artifacts directory!")
            st.stop()
        
        selected_index = st.selectbox(
            "Select FAISS Index:",
            options=available_indexes,
            index=available_indexes.index(DEFAULT_INDEX) if DEFAULT_INDEX in available_indexes else 0,
            help="Different indexes offer different speed/accuracy tradeoffs"
        )
        
        # Top-K selection
        k_value = st.slider(
            "Number of results (k):",
            min_value=1,
            max_value=20,
            value=DEFAULT_K,
            help="How many similar documents to retrieve"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - **Model**: all-MiniLM-L6-v2
        - **Corpus**: AG News (120k articles)
        - **Index**: FAISS (various configurations)
        """)
    
    # Load resources
    with st.spinner("Loading model and data... (this may take a moment on first load)"):
        model = load_model()
        corpus_texts = load_corpus()
        index = load_index(selected_index)
    
    if index is None:
        st.stop()
    
    st.success(f"✅ Loaded {len(corpus_texts):,} documents with index: `{selected_index}`")
    
    # Search interface
    st.markdown("---")
    st.subheader("🔎 Search")
    
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., technology news, sports updates, political events...",
        help="Type any natural language query"
    )
    
    if st.button("Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                # Encode query
                query_embedding = model.encode([query])
                query_embedding = normalize(query_embedding)
                
                # Search index
                distances, indices = index.search(query_embedding, k_value)
                
                # Display results
                st.markdown("### 📊 Search Results")
                
                for rank, (doc_id, score) in enumerate(zip(indices[0], distances[0]), start=1):
                    if doc_id == -1:  # FAISS returns -1 for no result
                        continue
                    
                    doc_id = int(doc_id)
                    text = corpus_texts[doc_id]
                    
                    with st.container():
                        col1, col2 = st.columns([0.1, 0.9])
                        
                        with col1:
                            st.markdown(f"### {rank}")
                        
                        with col2:
                            st.markdown(f"**Similarity Score:** `{float(score):.4f}`")
                            st.markdown(f"**Document ID:** `{doc_id}`")
                            
                            # Show snippet
                            snippet = text[:300] + "..." if len(text) > 300 else text
                            st.info(snippet)
                            
                            # Expandable full text
                            with st.expander("Show full text"):
                                st.text(text)
                        
                        st.markdown("---")

if __name__ == "__main__":
    main()
