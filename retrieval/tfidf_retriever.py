"""Default retriever: real sparse vector retrieval with scikit-learn TF-IDF +
cosine similarity. Runs anywhere (no heavy model download), and is genuine vector
retrieval -- documents and the query are embedded as TF-IDF vectors and ranked by
cosine similarity. For dense semantic embeddings, see dense_retriever.py.
"""

from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Retriever


def _doc_text(p: Dict[str, Any]) -> str:
    # index on title + body + the procedure codes it governs
    return f"{p['title']} {p['text']} codes: {' '.join(p['procedure_codes'])}"


class TfidfRetriever(Retriever):
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.policies: List[Dict[str, Any]] = []

    def index(self, policies: List[Dict[str, Any]]):
        self.policies = policies
        self.matrix = self.vec.fit_transform([_doc_text(p) for p in policies])
        return self

    def query(self, text: str, candidates: List[Dict[str, Any]], k: int):
        """Rank only the (temporally-filtered) candidates by cosine similarity."""
        if not candidates:
            return []
        idx = [self.policies.index(c) for c in candidates]
        qv = self.vec.transform([text])
        sims = cosine_similarity(qv, self.matrix[idx]).ravel()
        order = sims.argsort()[::-1][:k]
        return [(candidates[i], float(sims[i])) for i in order]
