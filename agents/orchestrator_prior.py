"""Step 1 - Prior-Guided Risk Assessment (Master Orchestrator).

Computes each provider's base fraud propensity with a Beta-Bernoulli conjugate
model (exact Bayesian posterior), overrides to 1.0 for OIG-excluded providers,
adds a facility-network risk term, and routes the claim:
  - high propensity -> DEEP  (deep multi-document retrieval, full anomaly scan)
  - low propensity  -> FAST  (shallow retrieval, skip expensive checks)

Honest scoping: the blueprint specifies a *Variational* Bayesian network prior.
For a tractable, exact, and fully-testable POC this uses the conjugate
Beta-Bernoulli posterior (which is exact for this Bernoulli-outcome model, so no
variational approximation is needed) plus a simple facility-network bump. The
production upgrade -- a full variational network model over the provider graph --
is documented in the README.
"""

from typing import Dict

import pandas as pd

from .base import RiskAssessment


class PriorOrchestrator:
    def __init__(self, a0: float = 1.0, b0: float = 11.0, route_threshold: float = 0.15,
                 network_weight: float = 0.3):
        # Beta(a0,b0): prior mean a0/(a0+b0) ~ 0.083, matching the ~8% base fraud rate.
        self.a0, self.b0 = a0, b0
        self.route_threshold = route_threshold
        self.network_weight = network_weight
        self.post_mean: Dict[str, float] = {}   # provider_id -> posterior mean
        self.alpha: Dict[str, float] = {}        # provider_id -> Beta alpha (for online update)
        self.beta: Dict[str, float] = {}         # provider_id -> Beta beta
        self.excluded_npis: set = set()
        self.facility_risk: Dict[str, float] = {}
        self._prov_facility: Dict[str, str] = {}
        self._prov_npi: Dict[str, str] = {}

    def fit(self, history: pd.DataFrame, providers: pd.DataFrame, leie: pd.DataFrame):
        """Estimate provider posteriors from labeled historical claims."""
        self.excluded_npis = set(leie["NPI"].astype(str)) if len(leie) else set()
        self._prov_facility = dict(zip(providers["Provider_ID"], providers["Facility_ID"]))
        self._prov_npi = dict(zip(providers["Provider_ID"], providers["NPI"].astype(str)))

        grp = history.groupby("Provider_ID")["Is_Fraud"].agg(["sum", "count"])
        for pid, row in grp.iterrows():
            k, n = float(row["sum"]), float(row["count"])
            a, b = self.a0 + k, self.b0 + (n - k)
            self.alpha[pid], self.beta[pid] = a, b
            self.post_mean[pid] = a / (a + b)
        # providers with no history fall back to the global prior mean
        for pid in providers["Provider_ID"]:
            if pid not in self.post_mean:
                self.alpha[pid], self.beta[pid] = self.a0, self.b0
                self.post_mean[pid] = self.a0 / (self.a0 + self.b0)

        # facility-network risk = mean provider posterior within each facility
        fac = pd.DataFrame({"pid": list(self.post_mean.keys()),
                            "post": list(self.post_mean.values())})
        fac["facility"] = fac["pid"].map(self._prov_facility)
        self.facility_risk = fac.groupby("facility")["post"].mean().to_dict()
        return self

    def assess(self, claim: dict) -> RiskAssessment:
        pid = claim["provider_id"]
        npi = str(claim["npi"])
        base = self.post_mean.get(pid, self.a0 / (self.a0 + self.b0))

        if npi in self.excluded_npis:
            return RiskAssessment(pid, npi, 1.0, "DEEP",
                                  "Provider is on the OIG LEIE (excluded); maximal prior, deep route.")

        facility = self._prov_facility.get(pid)
        net = self.facility_risk.get(facility, 0.0)
        prior = min(1.0, base + self.network_weight * net)
        route = "DEEP" if prior >= self.route_threshold else "FAST"
        rationale = (f"Bayesian posterior {base:.3f} (Beta({self.alpha.get(pid, self.a0):.0f},"
                     f"{self.beta.get(pid, self.b0):.0f})), facility-network risk {net:.3f} "
                     f"-> prior {prior:.3f} -> {route} route.")
        return RiskAssessment(pid, npi, round(prior, 3), route, rationale)
