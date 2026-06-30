r"""LangGraph orchestration of the five-stage pipeline.

The control flow IS a state graph: Step 1 sets a route, and a conditional edge
branches DEEP (run the anomaly node) vs FAST (skip it) before adjudication.

    START -> prior -> coding -> necessity --(DEEP)--> anomaly -> reviewer -> END
                                          \--(FAST)--------------/

Nodes reuse the already-tested agent objects from pipeline.Pipeline, so LangGraph is
a thin, legible orchestration layer over verified components.

NOTE: requires `pip install langgraph`. It is written against the current LangGraph
API and is syntax-checked, but run it once locally before recording:
    pip install -r requirements.txt
    python run_graph.py --claim CLM-100459
"""

import argparse
import operator
from typing import Annotated, List, Dict, Any, TypedDict

from pipeline import Pipeline


class ClaimState(TypedDict):
    raw: Dict[str, Any]
    claim: Dict[str, Any]
    route: str
    prior_mean: float
    route_rationale: str
    findings: Annotated[List[Any], operator.add]   # reducer concatenates node outputs
    decision: Dict[str, Any]


def build_app(pipeline: Pipeline):
    from langgraph.graph import StateGraph, START, END

    def prior_node(state: ClaimState) -> Dict[str, Any]:
        from agents import parse_claim
        claim = parse_claim(state["raw"])
        risk = pipeline.orch.assess(claim)
        return {"claim": claim, "route": risk.route, "prior_mean": risk.prior_mean,
                "route_rationale": risk.rationale, "findings": []}

    def coding_node(state: ClaimState) -> Dict[str, Any]:
        ctx = {"route": state["route"], "prior_mean": state["prior_mean"]}
        return {"findings": pipeline.coding.run(state["claim"], ctx)}

    def necessity_node(state: ClaimState) -> Dict[str, Any]:
        ctx = {"route": state["route"], "prior_mean": state["prior_mean"]}
        return {"findings": pipeline.necessity.run(state["claim"], ctx)}

    def anomaly_node(state: ClaimState) -> Dict[str, Any]:
        ctx = {"route": state["route"], "prior_mean": state["prior_mean"]}
        return {"findings": pipeline.anomaly.run(state["claim"], ctx)}

    def reviewer_node(state: ClaimState) -> Dict[str, Any]:
        decision = pipeline.reviewer.adjudicate(state["claim"], state["findings"], state["prior_mean"])
        decision["route"] = state["route"]
        decision["route_rationale"] = state["route_rationale"]
        return {"decision": decision}

    def route_decider(state: ClaimState) -> str:
        return "anomaly" if state["route"] == "DEEP" else "reviewer"

    b = StateGraph(ClaimState)
    b.add_node("prior", prior_node)
    b.add_node("coding", coding_node)
    b.add_node("necessity", necessity_node)
    b.add_node("anomaly", anomaly_node)
    b.add_node("reviewer", reviewer_node)
    b.add_edge(START, "prior")
    b.add_edge("prior", "coding")
    b.add_edge("coding", "necessity")
    b.add_conditional_edges("necessity", route_decider,
                            {"anomaly": "anomaly", "reviewer": "reviewer"})
    b.add_edge("anomaly", "reviewer")
    b.add_edge("reviewer", END)
    return b.compile()


def run(claim_id: str):
    pipeline = Pipeline()
    app = build_app(pipeline)
    row = pipeline.incoming[pipeline.incoming["Claim_ID"] == claim_id]
    if len(row) == 0:
        row = pipeline.history[pipeline.history["Claim_ID"] == claim_id]
    if len(row) == 0:
        print(f"claim {claim_id} not found"); return
    raw = row.iloc[0].to_dict()
    result = app.invoke({"raw": raw})
    d = result["decision"]
    print(f"\n=== LangGraph adjudication: {d['claim_id']} ===")
    print(f"route     : {d['route']}  ({d['route_rationale']})")
    print(f"action    : {d['action'].upper()}   risk={d['risk_score']}  "
          f"E[recovery]=${d['expected_recovery']:.0f}  HITL={d['human_in_the_loop']}")
    for f in d["findings"]:
        print(f"  - {f['error_type']:10} [{f['severity']}] {f['line_ref']}: {f['message']}")
        print(f"      evidence: {f['evidence'][:120]}...")
    if not d["findings"]:
        print("  (no findings)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="CLM-100030")
    run(ap.parse_args().claim)
