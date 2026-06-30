"""Optional dense retriever: sentence-transformers embeddings + FAISS index.

This is the production-grade semantic retrieval path. It is NOT required -- the
TF-IDF retriever is the default and runs everywhere. Enable this by installing
the extras and setting RETRIEVER=dense:

    pip install sentence-transformers faiss-cpu
    export RETRIEVER=dense

Kept import-light: heavy libs are imported lazily inside index() so the module
can be imported even when they are absent.
"""

from typing import List, Dict, Any

import numpy as np

from .base import Retriever
from .tfidf_retriever import _doc_text


class DenseRetriever(Retriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index_obj = None
        self.embeddings = None
        self.policies: List[Dict[str, Any]] = []

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def index(self, policies: List[Dict[str, Any]]):
        import faiss
        self._ensure_model()
        self.policies = policies
        emb = self.model.encode([_doc_text(p) for p in policies], normalize_embeddings=True)
        self.embeddings = np.asarray(emb, dtype="float32")
        self.index_obj = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine via normalized inner product
        self.index_obj.add(self.embeddings)
        return self

    def query(self, text: str, candidates: List[Dict[str, Any]], k: int):
        if not candidates:
            return []
        self._ensure_model()
        qv = np.asarray(self.model.encode([text], normalize_embeddings=True), dtype="float32")
        cand_idx = [self.policies.index(c) for c in candidates]
        cand_emb = self.embeddings[cand_idx]
        sims = (qv @ cand_emb.T).ravel()
        order = sims.argsort()[::-1][:k]
        return [(candidates[i], float(sims[i])) for i in order]
