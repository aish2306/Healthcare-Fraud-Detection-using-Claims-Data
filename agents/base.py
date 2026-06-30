"""Shared primitives for the five-stage fraud-detection multi-agent system."""

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List, Optional, Dict, Any

SEVERITY_WEIGHT = {"info": 0.0, "low": 0.3, "medium": 0.6, "high": 0.9}


@dataclass
class Finding:
    agent: str
    claim_id: str
    error_type: str          # exclusion|unbundling|duplicate|necessity|upcoding|anomaly|high_dollar
    severity: str            # info|low|medium|high
    line_ref: Optional[str]
    amount_at_risk: float
    message: str
    evidence: str            # the rule / policy text / statistic that justifies it (the citation)

    def to_dict(self):
        return asdict(self)


@dataclass
class RiskAssessment:
    """Output of Step 1 (the Bayesian prior orchestrator)."""
    provider_id: str
    npi: str
    prior_mean: float        # posterior fraud propensity in [0,1]
    route: str               # "DEEP" or "FAST"
    rationale: str


def parse_claim(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw claims.csv row (dict) into the structure agents consume.

    Splits the ';'-separated CPT/ICD/Units fields into aligned lists and parses
    the date of service. Keeps the raw fields too.
    """
    cpts = str(row["CPT_Codes"]).split(";")
    units = [int(u) for u in str(row["Units"]).split(";")]
    icds = str(row["ICD10_Codes"]).split(";")
    # distribute submitted amount across lines proportional to nothing fancy: even split
    n = max(len(cpts), 1)
    per_line = float(row["Submitted_Amount"]) / n
    lines = [{"procedure_code": c, "units": u, "billed": round(per_line, 2)}
             for c, u in zip(cpts, units)]
    dos = str(row["Date_Of_Service"])
    return {
        "claim_id": row["Claim_ID"],
        "provider_id": row["Provider_ID"],
        "npi": str(row["NPI"]),
        "diagnosis_codes": icds,
        "lines": lines,
        "date_of_service": dos,
        "dos_date": date.fromisoformat(dos),
        "submitted": float(row["Submitted_Amount"]),
        "insurance": row.get("Insurance_Type", ""),
        # label is carried only for offline evaluation; agents must not use it
        "_is_fraud": int(row.get("Is_Fraud", 0)) if row.get("Is_Fraud", "") != "" else None,
    }


@dataclass
class Agent:
    name: str = "agent"
    cfg: dict = field(default_factory=dict)

    def run(self, claim: dict, ctx: dict) -> List[Finding]:  # pragma: no cover - interface
        raise NotImplementedError
