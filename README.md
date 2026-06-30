# Multi-Agent Healthcare Fraud Detection

A proof-of-concept implementing a five-stage payment-integrity pipeline: a Bayesian
prior-routing orchestrator, deterministic coding checks, **temporal GraphRAG**
medical-necessity verification, unsupervised peer-baseline anomaly profiling, and
budget-constrained adjudication with a human-in-the-loop safeguard. Orchestrated as a
**LangGraph** state graph.

Built for the Cotiviti agentic-AI internship assessment.

---

## The five stages

| # | Stage | Implementation | File |
|---|-------|----------------|------|
| 1 | Prior-guided risk + routing | Beta-Bernoulli conjugate provider posterior, OIG-LEIE override, facility-network term -> DEEP/FAST route | `agents/orchestrator_prior.py` |
| 2 | Deterministic coding | NCCI pairs, duplicates, impossible-units | `agents/coding.py` |
| 3 | Temporal **GraphRAG** necessity | NetworkX graph traversal (procedure<-GOVERNS<-policy) + **date-of-service temporal filter** + TF-IDF ranking | `agents/necessity_rag.py`, `retrieval/graph_store.py` |
| 4 | Unsupervised anomaly | peer (specialty x HCPCS) z-score from CMS Part B-style baselines | `agents/anomaly.py` |
| 5 | Budget-constrained adjudication | risk = blend(prior, severity); expected recovery; ROI-ranked queue; HITL | `agents/reviewer.py`, `pipeline.py` |

Orchestration: **LangGraph** state graph with a conditional DEEP/FAST edge (`run_graph.py`).

## Quick start

```bash
pip install -r requirements.txt
python -m data.generate_data        # writes synthetic stand-ins into data/synth/

# tested pipeline (runs on numpy/pandas/sklearn/networkx)
python run_pipeline.py                       # batch summary + ROI-ranked queue
python run_pipeline.py --claim CLM-100030    # full five-stage trace for one claim

# LangGraph orchestration (needs `pip install langgraph`)
python run_graph.py --claim CLM-100030

# human review console (needs `pip install streamlit`)
streamlit run app.py
```

## Human review console (`app.py`)

A Streamlit UI for the review/adjudication stage:

- Streams every incoming claim, **searchable by Claim ID**, filterable by action/route.
- For each flagged claim it shows the full **agent reasoning** and the **reference**
  behind every finding (in-force policy text + graph path, NCCI rule, peer statistic).
- A reviewer records a **disposition** (Confirm fraud / Not fraud) per claim, and the
  worklist of dispositions can be exported as JSON for the audit team.

## The temporal GraphRAG showcase (your strongest demo moment)

Claim **CLM-100030** bills an MRI brain (70553) for headache (R51.9) on 2025-05-30.
The graph retrieves the policy **in force on that date** — `POL-IMG-001`, which does
*not* cover R51 — and denies. The corpus also contains `POL-IMG-002`, a revision
effective 2025-07-01 that *does* cover R51. Same claim, later date of service =>
different governing policy => different decision. That is time-drift elimination by
construction. Verify it directly:

```python
from datetime import date
from retrieval.graph_store import ClaimGraph
from agents.loaders import load_claims, load_providers, load_policies
g = ClaimGraph().build(load_claims(), load_providers(), load_policies())
g.governing_policies("70553", date(2025, 3, 1))   # -> POL-IMG-001 (old)
g.governing_policies("70553", date(2025, 9, 1))   # -> POL-IMG-002 (revised)
```

## Datasets (synthetic stand-ins matching real open schemas)

`data/generate_data.py` emits files whose columns match real open datasets, so you
can drop a real download in by pointing an env var at it (see `data/schemas.py`,
`agents/loaders.py`).

| Stand-in | Matches (real) | Used by |
|---|---|---|
| `claims.csv` | Kaggle "Healthcare Fraud Detection" / NHIS fraud typologies | Steps 1,2,3,5 |
| `provider_baselines.csv` | CMS Medicare Physician & Other Practitioners (Part B) | Step 4 |
| `leie.csv` | OIG List of Excluded Individuals/Entities | Step 1 |
| `policies.json` | CMS Coverage Database / Synthea-derived guidelines (with effective windows) | Step 3 |

Swap in real CSVs:
```bash
export CLAIMS_CSV=/path/to/kaggle_claims.csv
export BASELINES_CSV=/path/to/cms_partb.csv
export LEIE_CSV=/path/to/leie.csv
```
Claims column aliases (e.g. `PotentialFraud` -> `Is_Fraud`) are normalized
automatically; the LEIE join key `NPI` is identical in the real file.

## Honest scoping (say this in your write-up)

This POC implements the measurable, runnable core of each stage and documents the
heavier research components as the production path:

- **Step 1** uses an exact Beta-Bernoulli conjugate posterior (no approximation
  needed for Bernoulli outcomes) plus a facility-network term. The full *Variational*
  Bayesian network prior over the provider graph is the upgrade.
- **Step 3** does graph traversal + temporal filtering + vector ranking. A full
  knowledge-graph layer with learned entity/relation embeddings (Microsoft-style
  GraphRAG community summaries) is the upgrade; the graph structure is already here.
- Retrieval defaults to **real sklearn TF-IDF vector retrieval**; set `RETRIEVER=dense`
  for sentence-transformers + FAISS.

## Limitations

Synthetic data (10 policies, 1,500 claims); illustrative NCCI edits. Metric truth in
the data is `Is_Fraud`, retained only for offline evaluation; the agents never read it.
This is a concept demonstrator, not a production engine — but every stage is real and
the architecture is data-scale-independent.

## Files

```
data/        generate_data.py, schemas.py, synth/ (generated)
agents/      base, orchestrator_prior, coding, necessity_rag, anomaly, reviewer, loaders
retrieval/   graph_store (NetworkX GraphRAG), tfidf_retriever, dense_retriever
pipeline.py  five-stage orchestration (tested)
run_pipeline.py   CLI (tested)
run_graph.py LangGraph orchestration (run locally)
app.py       Streamlit human-review console (run locally)
```
