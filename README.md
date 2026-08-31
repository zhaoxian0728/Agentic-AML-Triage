# Agentic-AML-Triage
This is a multi-agent AI system that helps AML investigators at banks triage flagged accounts for money-mule activity. Instead of a single false-positive-prone score, it reasons step-by-step — checking linked accounts, transaction velocity, and cash-out patterns before handing investigators a ranked, explained shortlist.

# Mule Agent — AML Money-Mule Triage & Investigation System

## Overview
A junior AML Investigations Analyst at a bank needs a fast, explainable
way to tell which flagged accounts show genuine money-mule behavior,
because 90-98% of AML alerts are false positives requiring full manual
review. This system does the first-pass investigative work: a Detector
screens transactions, an Investigation Agent reasons through evidence on
flagged accounts, and confirmed cases are ranked and explained for a
human investigator. See docs/architecture.md for full design details.

## Setup
1. Clone the repo and cd into it
2. python -m venv venv
3. Activate: .\venv\Scripts\Activate.ps1  (Windows)
4. pip install -r requirements.txt
5. cp .env.example .env, then fill in your AWS and LangSmith credentials

## How to Run
1. python sample_data.py          # generates data/paysim_sample.csv
2. python evaluate_detector.py    # Detector evaluation, prints precision/recall
3. python run_pipeline.py         # full pipeline on a 10-account test batch

## File Overview
- src/schemas.py               Pydantic models for all agent inputs/outputs
- src/features.py              Feature engineering (leakage-safe: type + amount only)
- src/detector.py              Plain-code scorer, routes to auto-close or investigate
- src/tools.py                 3 tools: account history, linked accounts, velocity
- src/investigation_agent.py   Real agent: ReAct loop investigating flagged recipients
- src/prioritization_agent.py  Plain-code ranking (Prioritizer, not an agent)
- src/explainer_agent.py       Real agent: drafts hedged narrative for investigators
- src/output_reviewer_agent.py Real agent: final checklist + safe-wording gate
- src/graph.py                 LangGraph wiring connecting all of the above
- sample_data.py               One-time script: samples full PaySim to a working subset
- evaluate_detector.py         Detector-only evaluation against ground truth
- run_pipeline.py              Runs the full pipeline end-to-end on a test batch

## Testing / Evaluation
Full evaluation methodology and results are covered in our presentation
slides. Summary: naive baseline rule catches 0.19% of fraud; our Detector
alone catches 100% recall at 36% auto-clear rate. LangSmith traces the
full agent reasoning loop for cost, iteration count, and accuracy metrics.
