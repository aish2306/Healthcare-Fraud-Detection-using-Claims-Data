"""Graph store for graph-structured retrieval (the 'GraphRAG' core).

Builds a heterogeneous NetworkX graph and answers retrieval by TRAVERSAL, not just
vector similarity:

  nodes:  ("provider", npi) ("procedure", cpt) ("policy", id) ("patient", pid)
          ("facility", fid)
  edges:  policy   -GOVERNS->   procedure       (policy carries effective/end dates)
          provider -AT->        facility
          provider -SAW->       patient         (enables shared-patient networks)

Key retrieval method: governing_policies(code, dos) walks procedure -> GOVERNS ->
policy and applies the TEMPORAL filter (policy active on the date of service), so a
guideline that was not in force on the DOS can never be retrieved. That is the
time-drift-free, defensible retrieval the blueprint asks for.

The graph also exposes provider-network structure (shared patients / facility) used
for collusion-style risk -- the same graph serves multiple agents.
"""

from datetime import date
from typing import List, Dict, Any

import networkx as nx
import pandas as pd


class ClaimGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    # ---- construction -------------------------------------------------------
    def build(self, claims: pd.DataFrame, providers: pd.DataFrame, policies: List[Dict[str, Any]]):
        for p in policies:
            pid = ("policy", p["policy_id"])
            self.g.add_node(pid, kind="policy", **p)
            for code in p["procedure_codes"]:
                self.g.add_node(("procedure", code), kind="procedure")
                # policy GOVERNS procedure
                self.g.add_edge(pid, ("procedure", code), rel="GOVERNS")

        for _, pr in providers.iterrows():
            pnode = ("provider", str(pr["NPI"]))
            self.g.add_node(pnode, kind="provider", provider_id=pr["Provider_ID"],
                            specialty=pr.get("Specialty"), facility=pr.get("Facility_ID"))
            self.g.add_edge(pnode, ("facility", pr.get("Facility_ID")), rel="AT")

        for _, c in claims.iterrows():
            pnode = ("provider", str(c["NPI"]))
            patient = ("patient", c["Patient_ID"])
            self.g.add_edge(pnode, patient, rel="SAW", claim=c["Claim_ID"])
        return self

    # ---- graph retrieval ----------------------------------------------------
    def governing_policies(self, code: str, dos: date) -> List[Dict[str, Any]]:
        """Policies that GOVERN this procedure AND were active on the DOS."""
        node = ("procedure", code)
        if node not in self.g:
            return []
        out = []
        for policy_node in self.g.predecessors(node):
            # predecessors via GOVERNS edges (policy -> procedure)
            if not any(d.get("rel") == "GOVERNS" for d in self.g.get_edge_data(policy_node, node).values()):
                continue
            attrs = self.g.nodes[policy_node]
            eff = date.fromisoformat(attrs["effective_date"])
            end = date.fromisoformat(attrs["end_date"])
            if eff <= dos <= end:                       # TEMPORAL edge filter
                out.append(attrs)
        return out

    def shared_patient_providers(self, npi: str) -> List[str]:
        """Other providers who share >=1 patient (2-hop provider->patient->provider)."""
        pnode = ("provider", str(npi))
        if pnode not in self.g:
            return []
        peers = set()
        for patient in self.g.successors(pnode):
            if patient[0] != "patient":
                continue
            for prov in self.g.predecessors(patient):
                if prov[0] == "provider" and prov != pnode:
                    peers.add(prov[1])
        return sorted(peers)

    def facility_peers(self, npi: str) -> List[str]:
        pnode = ("provider", str(npi))
        if pnode not in self.g:
            return []
        peers = set()
        for fac in self.g.successors(pnode):
            if fac[0] != "facility":
                continue
            for prov in self.g.predecessors(fac):
                if prov[0] == "provider" and prov != pnode:
                    peers.add(prov[1])
        return sorted(peers)

    def stats(self) -> Dict[str, int]:
        kinds = {}
        for _, a in self.g.nodes(data=True):
            kinds[a.get("kind", "other")] = kinds.get(a.get("kind", "other"), 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), **kinds}
