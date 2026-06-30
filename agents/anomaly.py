"""Step 4 - Unsupervised Behavioral Profiling (Anomaly Agent).

Profiles each line's charge against peer baselines built from the CMS Part B-style
provider x HCPCS table: for each (specialty, HCPCS) it computes the peer mean and
std of submitted charge, then z-scores the claim's charge. Unsupervised -- no fraud
labels used -- which is the point: confirmed labels are scarce and schemes evolve.

`self.z_threshold` is a tunable outlier-sensitivity threshold.
"""

from typing import List, Dict

import pandas as pd

from .base import Finding


class AnomalyAgent:
    name = "AnomalyFWA"

    def __init__(self, baselines: pd.DataFrame, providers: pd.DataFrame, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self.prov_specialty = dict(zip(providers["Provider_ID"], providers["Specialty"]))
        b = baselines.merge(providers[["NPI", "Specialty"]], on="NPI", how="left",
                            suffixes=("", "_p"))
        spec_col = "Specialty" if "Specialty" in b.columns else "Specialty_p"
        stats = b.groupby([spec_col, "HCPCS_Code"])["Avg_Submitted_Charge"].agg(["mean", "std"]).reset_index()
        stats.columns = ["Specialty", "HCPCS_Code", "peer_mean", "peer_std"]
        self.peer = {(r["Specialty"], r["HCPCS_Code"]): (r["peer_mean"], r["peer_std"])
                     for _, r in stats.iterrows()}

    def run(self, claim: dict, ctx: dict) -> List[Finding]:
        findings: List[Finding] = []
        specialty = self.prov_specialty.get(claim["provider_id"])
        for ln in claim["lines"]:
            key = (specialty, ln["procedure_code"])
            if key not in self.peer:
                continue
            mu, sd = self.peer[key]
            if not sd or sd <= 0:
                continue
            z = (ln["billed"] - mu) / sd
            if z > self.z_threshold:
                findings.append(Finding(
                    self.name, claim["claim_id"], "anomaly", "medium", ln["procedure_code"],
                    round(max(ln["billed"] - mu, 0.0), 2),
                    f"{ln['procedure_code']}: ${ln['billed']:.0f} is {z:.1f} SD above the "
                    f"{specialty} peer mean (${mu:.0f}).",
                    f"Peer baseline ({specialty}/{ln['procedure_code']}): mean=${mu:.0f}, "
                    f"std=${sd:.0f}, z-threshold={self.z_threshold}."))
        return findings
