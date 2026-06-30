"""Retriever interface + the temporal filter that eliminates 'time drift'.

The temporal filter is the core of Step 3: before ranking by similarity, we keep
only policies whose [effective_date, end_date] window contains the claim's Date
of Service. This guarantees we never retrieve a guideline that was not in force
on the DOS -- the defensibility property the blueprint calls for.
"""

from datetime import date
from typing import List, Dict, Any


def active_on(policies: List[Dict[str, Any]], dos: date) -> List[Dict[str, Any]]:
    """Return only policies in force on the date of service."""
    out = []
    for p in policies:
        eff = date.fromisoformat(p["effective_date"])
        end = date.fromisoformat(p["end_date"])
        if eff <= dos <= end:
            out.append(p)
    return out


class Retriever:
    """Interface: index a set of policy docs, then retrieve top-k for a query."""
    def index(self, policies: List[Dict[str, Any]]):  # pragma: no cover
        raise NotImplementedError

    def query(self, text: str, candidates: List[Dict[str, Any]], k: int):  # pragma: no cover
        raise NotImplementedError
