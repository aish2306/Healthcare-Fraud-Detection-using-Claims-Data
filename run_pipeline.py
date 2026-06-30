"""Console runner for the five-stage pipeline (uses pandas/sklearn/networkx).

  python run_pipeline.py            batch summary + top of the ROI-ranked queue
  python run_pipeline.py --claim CLM-100030    full five-stage trace for one claim
"""

import argparse
from collections import Counter

from pipeline import Pipeline

TAG = {"deny": "[DENY]  ", "review": "[REVIEW]", "pay": "[PAY]   "}


def trace_one(p: Pipeline, claim_id: str):
    row = p.incoming[p.incoming["Claim_ID"] == claim_id]
    if len(row) == 0:
        row = p.history[p.history["Claim_ID"] == claim_id]
    if len(row) == 0:
        print(f"claim {claim_id} not found"); return
    d = p.process(row.iloc[0].to_dict())
    print(f"\n=== {d['claim_id']} ===")
    print(f"Step 1 prior/route : {d['route']}  | {d['route_rationale']}")
    print(f"Step 2-4 agents    : {d['agent_trace']}")
    print(f"Step 5 decision    : {d['action'].upper()}  risk={d['risk_score']}  "
          f"prior={d['prior_mean']}  at_risk=${d['dollars_at_risk']:.0f}  "
          f"E[recovery]=${d['expected_recovery']:.0f}  HITL={d['human_in_the_loop']}")
    for f in d["findings"]:
        print(f"   - {f['error_type']:10} [{f['severity']}] {f['line_ref']}: {f['message']}")
        print(f"       evidence: {f['evidence'][:130]}")
    if not d["findings"]:
        print("   (no findings)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default=None)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()

    p = Pipeline(review_budget=args.budget)
    if args.claim:
        trace_one(p, args.claim)
        return

    decisions = p.run_incoming()
    actions = Counter(d["action"] for d in decisions)
    routes = Counter(d["route"] for d in decisions)
    at_risk = sum(d["dollars_at_risk"] for d in decisions)
    exp = sum(d["expected_recovery"] for d in decisions)
    print("=" * 74)
    print("  FIVE-STAGE FRAUD-DETECTION PIPELINE  (synthetic stand-in data)")
    print("=" * 74)
    print(f"incoming claims : {len(decisions)}")
    print(f"actions         : {dict(actions)}")
    print(f"routing         : {dict(routes)}  (DEEP runs the anomaly stage)")
    print(f"dollars at risk : ${at_risk:,.0f}   expected recovery ${exp:,.0f}")
    print(f"\nTop of the ROI-ranked review queue (budget {args.budget}):")
    print("-" * 74)
    ranked = sorted([d for d in decisions if d["action"] != "pay"],
                    key=lambda d: d["expected_recovery"], reverse=True)[:8]
    for d in ranked:
        hitl = " HITL" if d["human_in_the_loop"] else ""
        print(f"  #{d['queue_rank']:<3} {TAG[d['action']]} {d['claim_id']}  "
              f"risk={d['risk_score']:.2f} route={d['route']:4} "
              f"E[rec]=${d['expected_recovery']:,.0f}{hitl}")
    print("\nRun a single claim:  python run_pipeline.py --claim CLM-100030")


if __name__ == "__main__":
    main()
