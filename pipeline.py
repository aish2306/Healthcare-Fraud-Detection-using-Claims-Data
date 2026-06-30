"""The five-stage orchestrated pipeline.

  Step 1  PriorOrchestrator.assess  -> provider prior + DEEP/FAST route
  Step 2  CodingAgent               -> deterministic coding findings
  Step 3  NecessityAgent            -> temporal-RAG necessity findings (depth by route)
  Step 4  AnomalyAgent              -> peer-baseline anomaly findings (DEEP route only)
  Step 5  ReviewerAgent             -> risk, expected recovery, action, HITL
  (batch) budget selection          -> ROI-ranked review queue under a review budget
"""

import os
from typing import List, Dict, Any

import pandas as pd

from agents import (PriorOrchestrator, CodingAgent, NecessityAgent, AnomalyAgent,
                    ReviewerAgent, parse_claim)
from agents.loaders import load_claims, load_baselines, load_leie, load_policies, load_providers
from retrieval.graph_store import ClaimGraph


class Pipeline:
    def __init__(self, history_frac: float = 0.5, review_budget: int = 60, seed: int = 7):
        claims = load_claims()
        providers = load_providers()
        baselines = load_baselines()
        leie = load_leie()
        policies = load_policies()

        # split: history (fit priors) vs incoming (score). deterministic shuffle.
        claims = claims.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_hist = int(len(claims) * history_frac)
        self.history = claims.iloc[:n_hist].copy()
        self.incoming = claims.iloc[n_hist:].copy()

        self.orch = PriorOrchestrator().fit(self.history, providers, leie)
        self.graph = ClaimGraph().build(claims, providers, policies)
        self.coding = CodingAgent()
        self.necessity = NecessityAgent(policies, self.graph)
        self.anomaly = AnomalyAgent(baselines, providers)
        self.reviewer = ReviewerAgent()
        self.review_budget = review_budget

    def process(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        claim = parse_claim(raw)
        risk = self.orch.assess(claim)                      # Step 1
        ctx = {"route": risk.route, "prior_mean": risk.prior_mean}

        findings = []
        findings += self.coding.run(claim, ctx)             # Step 2
        findings += self.necessity.run(claim, ctx)          # Step 3
        if risk.route == "DEEP":                             # Step 4 (deep track only)
            findings += self.anomaly.run(claim, ctx)

        decision = self.reviewer.adjudicate(claim, findings, risk.prior_mean)  # Step 5
        decision["route"] = risk.route
        decision["route_rationale"] = risk.rationale
        decision["agent_trace"] = {
            "coding": sum(1 for f in findings if f.agent == "CodingValidation"),
            "necessity": sum(1 for f in findings if f.agent == "MedicalNecessity"),
            "anomaly": sum(1 for f in findings if f.agent == "AnomalyFWA"),
        }
        return decision

    def run_incoming(self) -> List[Dict[str, Any]]:
        decisions = [self.process(r) for _, r in self.incoming.iterrows()]
        self._budget_select(decisions)
        return decisions

    def _budget_select(self, decisions: List[Dict[str, Any]]):
        """Batch ROI ranking: among flagged claims, select the top-B by expected
        recovery to work this cycle (the review-budget knapsack, greedy form)."""
        flagged = [d for d in decisions if d["action"] in ("deny", "review")]
        flagged.sort(key=lambda d: d["expected_recovery"], reverse=True)
        for rank, d in enumerate(flagged):
            d["queue_rank"] = rank + 1
            d["selected_this_cycle"] = rank < self.review_budget
        for d in decisions:
            d.setdefault("queue_rank", None)
            d.setdefault("selected_this_cycle", False)
