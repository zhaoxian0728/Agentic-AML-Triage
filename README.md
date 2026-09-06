# Agentic-AML-Triage

## Overview
A junior AML Investigations Analyst at a bank needs a fast, explainable way
to tell which flagged accounts show genuine money-mule behavior, because
90-98% of AML alerts are false positives requiring full manual review.

This is a multi-agent system that does the first-pass investigative work:
a plain-code Detector screens PaySim transactions on `type` and `amount`,
a reasoning Investigation Agent digs into the **recipient** of each flagged
transfer (money-mule accounts are defined by receiving from multiple
already-flagged accounts, not by sending behavior), and confirmed cases are
ranked by urgency and explained in hedged, evidence-based language before
reaching a human investigator. An Output Reviewer Agent gates every
narrative for tone and evidence before it goes out. See
`docs/architecture.md` for full design details.

## Setup
1. Clone the repo and `cd` into it
2. `python -m venv venv`
3. Activate: `.\venv\Scripts\Activate.ps1` (Windows)
4. `pip install -r requirements.txt`
5. `cp .env.example .env`, then fill in your AWS and LangSmith credentials

## How to Run
1. `python sample_data.py` — one-time: samples the full PaySim log down to
   a working subset at `data/paysim_sample.csv`
2. `python evaluate_detector.py` — Detector-only evaluation against ground
   truth across several thresholds, plus a naive-baseline comparison
3. `python evaluate_pipeline.py` — runs the full pipeline end-to-end on a
   balanced batch of fraud + hard-legit cases, prints end-to-end
   recall/precision
4. `python diagnose_pipeline.py` — same as above but with verbose
   per-account diagnostics (evidence, reasoning trail) and error handling
   so one bad account doesn't crash the run
5. `python run_pipeline.py` — runs the full pipeline on a small curated
   batch and prints the ranked, explained shortlist
6. `streamlit run app.py` — launches the interactive dashboard: upload a
   CSV (or use the default sample), run the pipeline on a batch, and
   review the prioritized case queue with a per-account deep-dive

## File Overview
- `src/schemas.py` — Pydantic models for every agent's input/output
  (`TransactionRecord`, `DetectorVerdict`, `InvestigationVerdict`,
  `PriorityRank`, `ExplainerOutput`, `OutputReviewerVerdict`)
- `src/features.py` — Feature engineering; computes `is_transfer_type`
  from `type` only, avoiding the leaky balance columns
- `src/detector.py` — Rule-based mule-likelihood scorer using only `type`
  and `amount` (the fields confirmed safe by the dataset's own docs);
  routes each transaction to auto-close or investigate
- `src/tools.py` — The Investigation Agent's three tools: account history
  (bidirectional, pre-transaction), linked-accounts fan-in ratio, and
  cash-out velocity
- `src/investigation_agent.py` — Real agent: a ReAct loop that
  investigates a flagged recipient using the tools above, then extracts a
  structured verdict
- `src/prioritization_agent.py` — Plain-code, deterministic urgency
  scoring and batch-relative ranking of already-confirmed accounts
- `src/explainer_agent.py` — Real agent: drafts a brief, hedged narrative
  for the investigator, grounded only in gathered evidence
- `src/output_reviewer_agent.py` — Real agent: final checklist and
  safe-wording gate before a narrative reaches a human
- `src/graph.py` — LangGraph wiring that connects Detector → Investigation
  → Prioritization → Explainer → Output Reviewer (with a capped rewrite loop)

## Testing / Evaluation
- **Held-out data:** 40% recall / 53% precision
- **Tuned sample:** 75% recall / 50% precision

Both far exceed the naive baseline rule (0.19% recall) and sit well above
real-world AML precision benchmarks (2-10%). See `docs/architecture.md`
for the full evaluation methodology.
