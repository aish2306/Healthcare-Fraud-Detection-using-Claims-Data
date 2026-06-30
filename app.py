"""Streamlit Review Console for the five-stage fraud-detection system.

    streamlit run app.py

- Streams every incoming claim, searchable by Claim ID, filterable by action/route.
- Shows each flagged claim's agent REASONING and the REFERENCE behind every finding
  (in-force policy text + graph path, NCCI rule, peer statistic).
- A reviewer records a disposition per claim (Confirm fraud / Not fraud); the worklist
  of dispositions can be exported for the audit team.
"""

import json
import os

import streamlit as st

from pipeline import Pipeline

st.set_page_config(page_title="Fraud Review Console", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- palette
C = {
    "deny": "#DC2626", "review": "#D97706", "pay": "#059669",
    "high": "#DC2626", "medium": "#D97706", "low": "#2563EB", "info": "#6B7280",
    "DEEP": "#7C3AED", "FAST": "#0891B2",
    "ink": "#1F2937", "muted": "#6B7280", "accent": "#4F46E5",
}
DISP_OPTIONS = ["— (no disposition)", "Confirm fraud", "Not fraud"]
DISP_LABEL = {"Confirm fraud": 1, "Not fraud": 0}
ACT_EMOJI = {"deny": "🔴", "review": "🟠", "pay": "🟢"}

CSS = """
<style>
  .block-container { padding-top: 1.4rem; }
  .hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #0891B2 100%);
    color: #fff; padding: 22px 26px; border-radius: 16px; margin-bottom: 14px;
    box-shadow: 0 6px 22px rgba(79,70,229,.25);
  }
  .hero h1 { font-size: 1.7rem; font-weight: 800; margin: 0; letter-spacing: -.3px; }
  .hero p  { font-size: .92rem; margin: 6px 0 0; opacity: .92; }
  .pill {
    display:inline-block; padding:2px 11px; border-radius:999px;
    font-size:.70rem; font-weight:800; letter-spacing:.4px; color:#fff;
    text-transform:uppercase; vertical-align:middle;
  }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .claimid { font-size:1.02rem; font-weight:800; color:#111827; }
  .kv { font-size:.82rem; color:#6B7280; }
  .kv b { color:#1F2937; }
  .rationale {
    font-size:.82rem; color:#374151; background:#EEF2FF;
    border-left:3px solid #4F46E5; padding:7px 11px; border-radius:6px; margin:8px 0 4px;
  }
  .sec { font-size:.74rem; font-weight:800; letter-spacing:.6px; color:#6B7280;
         text-transform:uppercase; margin:10px 0 4px; }
  .finding {
    border-left:5px solid #999; background:#F9FAFB; border:1px solid #EEF0F3;
    border-radius:8px; padding:9px 12px; margin:7px 0;
  }
  .ftitle { font-size:.92rem; font-weight:700; color:#111827; }
  .fmeta  { font-size:.72rem; color:#9CA3AF; font-weight:600; }
  .evidence {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.76rem;
    color:#374151; background:#F3F4F6; border:1px solid #E5E7EB;
    border-radius:6px; padding:7px 10px; margin-top:6px; white-space:pre-wrap;
  }
  .riskwrap { background:#E5E7EB; border-radius:999px; height:9px; width:160px; display:inline-block; vertical-align:middle; overflow:hidden; }
  .riskfill { height:9px; border-radius:999px; }
  .clean { font-size:.85rem; color:#059669; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- html helpers
def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'


def risk_meter(risk):
    col = C["deny"] if risk >= 0.6 else (C["review"] if risk >= 0.35 else C["pay"])
    pct = int(min(max(risk, 0), 1) * 100)
    return (f'<span class="riskwrap" style="background:linear-gradient(to right,'
            f'{col} {pct}%, #E5E7EB {pct}%)"></span> '
            f'<span class="kv"><b>{risk:.2f}</b> risk</span>')


def finding_html(f):
    col = C.get(f["severity"], "#999")
    return (
        f'<div class="finding" style="border-left-color:{col}">'
        f'{pill(f["error_type"], col)} '
        f'<span class="ftitle">&nbsp;{f["message"]}</span><br>'
        f'<span class="fmeta">{f["agent"]} · severity {f["severity"]} · code {f["line_ref"]}</span>'
        f'<div class="evidence">{f["evidence"]}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------- state
@st.cache_resource(show_spinner="Building pipeline (data, graph, priors)…")
def get_pipeline():
    return Pipeline()


def init():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()
    if "decisions" not in st.session_state:
        st.session_state.decisions = st.session_state.pipeline.run_incoming()
    if "dispositions" not in st.session_state:
        st.session_state.dispositions = {}     # claim_id -> 1 (fraud) / 0 (not)


# ---------------------------------------------------------------- UI
init()
p = st.session_state.pipeline
decisions = st.session_state.decisions

st.markdown(
    '<div class="hero"><h1>🛡️ Fraud Detection — Human Review Console</h1>'
    '<p>Bayesian routing · temporal GraphRAG · peer-anomaly profiling · '
    'budget-constrained adjudication · human-in-the-loop review</p></div>',
    unsafe_allow_html=True)

# ---- sidebar ----
with st.sidebar:
    st.markdown("### 🧾 Reviewer worklist")
    disp = st.session_state.dispositions
    n_fraud = sum(1 for v in disp.values() if v == 1)
    n_clear = sum(1 for v in disp.values() if v == 0)
    st.markdown(f"**Dispositions recorded:** {len(disp)}  \n"
                f"✅ Confirmed fraud: {n_fraud}  \n"
                f"⬜ Cleared: {n_clear}")
    if disp:
        export = [{"claim_id": cid, "disposition": ("fraud" if v == 1 else "not_fraud")}
                  for cid, v in disp.items()]
        st.download_button("⬇️ Export worklist (JSON)",
                           data=json.dumps(export, indent=2),
                           file_name="review_worklist.json",
                           mime="application/json", use_container_width=True)
    st.caption("Record a disposition on each flagged claim, then export the worklist "
               "for the audit team. Dispositions are a human-review record.")

# ---- KPI row ----
flagged = [d for d in decisions if d["action"] != "pay"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("📨 Incoming claims", f"{len(decisions):,}")
k2.metric("🚩 Flagged", f"{len(flagged):,}")
k3.metric("💸 Dollars at risk", f"${sum(d['dollars_at_risk'] for d in decisions):,.0f}")
k4.metric("👤 Need human review", f"{sum(1 for d in decisions if d['human_in_the_loop']):,}")

# ---- search + filters ----
st.markdown("#### 🔎 Review queue")
f1, f2, f3 = st.columns([2, 1, 1])
search = f1.text_input("Search by Claim ID", placeholder="e.g. CLM-100030").strip().upper()
action_filter = f2.selectbox("Action", ["flagged only", "all", "deny", "review", "pay"])
topn = f3.slider("Max shown", 5, 100, 25)


def visible(d):
    if search and search not in d["claim_id"].upper():
        return False
    if action_filter == "flagged only":
        return d["action"] != "pay"
    if action_filter == "all":
        return True
    return d["action"] == action_filter


rows = sorted([d for d in decisions if visible(d)],
              key=lambda d: d["dollars_at_risk"], reverse=True)[:topn]
st.caption(f"Showing {len(rows)} claim(s), ranked by dollars at risk.")

# ---- claim cards ----
for d in rows:
    disp = st.session_state.dispositions.get(d["claim_id"])
    vbadge = " ✅" if disp == 1 else (" ⬜" if disp == 0 else "")
    label = (f"{ACT_EMOJI[d['action']]} {d['action'].upper()}  ·  {d['claim_id']}  ·  "
             f"risk {d['risk_score']:.2f}{vbadge}")
    with st.expander(label, expanded=bool(search)):
        hitl = pill("HUMAN-IN-THE-LOOP", C["accent"]) if d["human_in_the_loop"] else ""
        header = (
            f'{pill(d["action"], C[d["action"]])} {hitl}<br>'
            f'<span class="claimid mono">{d["claim_id"]}</span> &nbsp; '
            f'{risk_meter(d["risk_score"])}<br>'
            f'<span class="kv">billed <b>${d["claim_total"]:,.0f}</b> · '
            f'at risk <b>${d["dollars_at_risk"]:,.0f}</b></span>'
        )
        st.markdown(header.replace("$", "&#36;"), unsafe_allow_html=True)

        if d["findings"]:
            st.markdown('<div class="sec">Why this claim is flagged — reasoning &amp; references</div>',
                        unsafe_allow_html=True)
            st.markdown("".join(finding_html(f) for f in d["findings"]).replace("$", "&#36;"),
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="clean">✓ No agent findings — flagged on provider '
                        'risk / high-dollar rule.</div>', unsafe_allow_html=True)

        choice = st.radio("Reviewer disposition:", DISP_OPTIONS, horizontal=True,
                          index=(0 if disp is None else (1 if disp == 1 else 2)),
                          key=f"disp_{d['claim_id']}")
        if choice in DISP_LABEL:
            st.session_state.dispositions[d["claim_id"]] = DISP_LABEL[choice]
        elif d["claim_id"] in st.session_state.dispositions:
            del st.session_state.dispositions[d["claim_id"]]

st.caption("Search a Claim ID to jump to it, read why each claim is flagged and the "
           "policy / coding / peer reference behind each finding, and record your disposition.")
