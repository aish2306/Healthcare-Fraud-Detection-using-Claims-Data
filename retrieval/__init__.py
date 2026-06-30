import os

from .tfidf_retriever import TfidfRetriever
from .base import active_on


def get_retriever():
    """Return the configured retriever. Default TF-IDF (always available);
    set RETRIEVER=dense for sentence-transformers + FAISS (must be installed)."""
    if os.environ.get("RETRIEVER", "tfidf").lower() == "dense":
        from .dense_retriever import DenseRetriever
        return DenseRetriever()
    return TfidfRetriever()
