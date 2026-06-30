"""Step 5 - Budget-Constrained Adjudication (Reviewer Agent).

Turns findings + the Step-1 prior into a decision. Risk score blends the provider
prior with the worst finding severity. Dollars at risk and expected recovery
(risk * dollars * recovery_rate) drive the batch-level ROI ranking under a review
budget (applied in pipeline.py). High-dollar denials route to a human.

`self.risk_cutoff` is a tunable decision threshold (precision/recall trade-off).
"""

from typing import List, Dict, Any

from .base import Finding, SEVERITY_WEIGHT


class ReviewerAgent:
    name = "Reviewer"

    def __init__(self, recovery_rate: float = 0.7, hitl_dollar: float = 10000.0,
                 risk_cutoff: float = 0.5, prior_weight: float = 0.35):
        self.rho = recovery_rate
        self.hitl = hitl_dollar
        self.risk_cutoff = risk_cutoff
        self.prior_weight = prior_weight

    def adjudicate(self, claim: dict, findings: List[Finding], prior_mean: float) -> Dict[str, Any]:
        claim_total = sum(ln["billed"] for ln in claim["lines"])
        sev = max((SEVERITY_WEIGHT[f.severity] for f in findings), default=0.0)
        # blend provider prior with evidence severity
        risk = min(1.0, (1 - self.prior_weight) * sev + self.prior_weight * prior_mean)
        # dollars at risk: overlapping findings can hit the same line, so take the
        # largest amount flagged per procedure line (not the sum), then cap at the
        # claim total -- you can never be at risk of more than was billed.
        per_line = {}
        for f in findings:
            key = f.line_ref
            per_line[key] = max(per_line.get(key, 0.0), f.amount_at_risk)
        dollars_at_risk = round(min(sum(per_line.values()), claim_total), 2)
        expected_recovery = round(risk * dollars_at_risk * self.rho, 2)

        has_hard = any(f.severity == "high" for f in findings)
        if has_hard:
            action = "deny"
        elif findings or risk >= self.risk_cutoff:
            action = "review"
        else:
            action = "pay"

        hitl = False
        if action == "deny" and claim_total >= self.hitl:
            action, hitl = "review", True
        if action == "pay" and claim_total >= self.hitl:
            action, hitl = "review", True

        return {
            "claim_id": claim["claim_id"], "provider_id": claim["provider_id"],
            "action": action, "risk_score": round(risk, 3), "prior_mean": round(prior_mean, 3),
            "claim_total": round(claim_total, 2), "dollars_at_risk": dollars_at_risk,
            "expected_recovery": expected_recovery, "human_in_the_loop": hitl,
            "n_findings": len(findings),
            "findings": [f.to_dict() for f in findings],
            "_is_fraud": claim.get("_is_fraud"),
        }
