"""Step 2 - Deterministic Pre-Filtering (Coding Validation Agent).

Rules-based first line of defense: NCCI mutually-exclusive / bundled pairs,
duplicate same-day lines, and an 'impossible units' sanity check. Deterministic,
so these are high-confidence findings that need no model.
"""

from collections import Counter
from typing import List

from .base import Finding

# Illustrative NCCI-style pairs; in production load the official CMS PTP edit tables.
NCCI_PAIRS = {
    ("80053", "80048"): "80048 is a component of 80053 (comprehensive metabolic panel); billing both unbundles a single panel.",
    ("99215", "99213"): "Two established-patient E/M codes for the same encounter are mutually exclusive.",
}
# crude per-code plausible daily unit ceiling
MAX_UNITS = {"99213": 1, "99214": 1, "99215": 1, "70553": 2, "27447": 1, "93000": 3}


class CodingAgent:
    name = "CodingValidation"

    def run(self, claim: dict, ctx: dict) -> List[Finding]:
        findings: List[Finding] = []
        lines = claim["lines"]
        codes = [ln["procedure_code"] for ln in lines]
        billed_by_code, units_by_code = {}, {}
        for ln in lines:
            billed_by_code[ln["procedure_code"]] = billed_by_code.get(ln["procedure_code"], 0.0) + ln["billed"]
            units_by_code[ln["procedure_code"]] = units_by_code.get(ln["procedure_code"], 0) + ln["units"]

        # duplicates
        for code, n in Counter(codes).items():
            if n > 1:
                per = billed_by_code[code] / n
                findings.append(Finding(self.name, claim["claim_id"], "duplicate", "high", code,
                                        round(per * (n - 1), 2),
                                        f"Procedure {code} billed {n}x same day without a distinguishing modifier.",
                                        "Duplicate same-day line; redundant copies are non-reimbursable."))

        # NCCI pairs
        cs = set(codes)
        for (c1, c2), why in NCCI_PAIRS.items():
            if c1 in cs and c2 in cs:
                findings.append(Finding(self.name, claim["claim_id"], "unbundling", "high", c2,
                                        round(billed_by_code.get(c2, 0.0), 2),
                                        f"Codes {c1} and {c2} billed together; {c2} should not be separately reported.",
                                        f"NCCI PTP edit (illustrative): {why}"))

        # impossible units
        for code, u in units_by_code.items():
            ceiling = MAX_UNITS.get(code)
            if ceiling and u > ceiling + 3:  # generous to avoid noise
                findings.append(Finding(self.name, claim["claim_id"], "excessive_units", "medium", code,
                                        round(billed_by_code.get(code, 0.0) * 0.5, 2),
                                        f"{code}: {u} units exceeds the plausible daily ceiling ({ceiling}).",
                                        f"Per-code daily unit ceiling = {ceiling}; implausible volume suggests phantom billing."))
        return findings
