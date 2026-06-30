"""Step 3 - Temporal GraphRAG Verification (Medical Necessity Agent).

Graph-structured retrieval: for each procedure on the claim it walks
procedure <- GOVERNS <- policy in the claim graph and keeps only policies active on
the Date of Service (temporal edges). When several policies govern, it ranks them by
TF-IDF similarity to the claim context (graph traversal + vector hybrid). It then
checks whether the governing policy covers the claim's diagnoses; if not, it emits a
necessity finding whose evidence is the graph path + the in-force policy text.

This is graph-structured retrieval over claims/providers/policies with temporal
edges -- defensible and time-drift-free. (A full knowledge-graph expansion with
learned entity/relation embeddings is the documented next step.)
"""

from typing import List, Dict

from .base import Finding
from retrieval import get_retriever

CODE_DESC = {
    "70553": "MRI brain advanced imaging", "99215": "high complexity office visit E/M level 5",
    "99213": "office visit", "80053": "comprehensive metabolic panel", "80048": "basic metabolic panel",
    "93000": "electrocardiogram ECG", "97110": "therapeutic exercise physical therapy",
    "27447": "total knee arthroplasty", "20610": "joint injection", "11042": "debridement wound",
}


def _covered(dx_list, prefixes) -> bool:
    if "*" in prefixes:
        return True
    return any(dx.upper().startswith(pfx.upper()) for dx in dx_list for pfx in prefixes)


class NecessityAgent:
    name = "MedicalNecessity"

    def __init__(self, policies: List[Dict], graph, k_deep: int = 5, k_fast: int = 2):
        self.policies = policies
        self.graph = graph
        self.k_deep, self.k_fast = k_deep, k_fast
        self.retriever = get_retriever().index(policies)

    def _rank(self, query: str, govs: List[Dict], k: int):
        if len(govs) <= 1:
            return [(p, 1.0) for p in govs]
        return self.retriever.query(query, govs, k)

    def run(self, claim: dict, ctx: dict) -> List[Finding]:
        findings: List[Finding] = []
        dx = claim["diagnosis_codes"]
        dos = claim["dos_date"]
        k = self.k_deep if ctx.get("route") == "DEEP" else self.k_fast

        for ln in claim["lines"]:
            code = ln["procedure_code"]
            govs = self.graph.governing_policies(code, dos)   # GRAPH RETRIEVAL + TEMPORAL FILTER
            if not govs:
                continue
            query = f"{code} {CODE_DESC.get(code, '')} diagnoses {' '.join(dx)}"
            ranked = self._rank(query, govs, k)
            gov, sim = ranked[0]

            if not _covered(dx, gov["covered_dx_prefixes"]):
                findings.append(Finding(
                    self.name, claim["claim_id"], "necessity", "high", code,
                    round(ln["billed"], 2),
                    f"{code} not supported for {dx} under {gov['policy_id']} (in force on {dos}).",
                    f"Policy {gov['policy_id']} (in force {gov['effective_date']} to "
                    f"{gov['end_date']}): {gov['text']}"))
            elif code == "99215" and any(d.upper().startswith("Z00") for d in dx):
                findings.append(Finding(
                    self.name, claim["claim_id"], "upcoding", "medium", code,
                    round(ln["billed"] * 0.5, 2),
                    "Level-5 E/M (99215) billed against a routine/preventive diagnosis.",
                    f"Policy {gov['policy_id']} (in force {gov['effective_date']} to "
                    f"{gov['end_date']}): {gov['text']}"))
        return findings
