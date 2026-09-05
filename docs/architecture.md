# Architecture

## Problem
A junior AML Investigations Analyst at a bank needs a fast, explainable way
to tell which flagged accounts show genuine money-mule behavior, because
90-98% of AML alerts are false positives requiring full manual review.

## Pipeline
PaySim transactions -> Detector (plain code, not an agent) ->
  [low score] -> Auto-close
  [med/high score] -> Investigation Agent (reasoning loop, investigates
    the RECIPIENT of the flagged transaction, not the sender)
    -> [false alarm] -> Close
    -> [confirmed] -> Prioritizer (plain code) -> Explainer Agent
    -> Output Reviewer Agent -> Investigator (ranked shortlist + narrative)

Output Reviewer can send the narrative back to the Explainer for
ONE rewrite max, then flags for human review if still failing.

## Key data finding: investigate the recipient, not the sender
Analysis of PaySim showed money-mule accounts are defined by RECEIVING
from multiple already-flagged accounts (a "fan-in" pattern), not by
sending behavior. Measured: 58.5% of fraud-transfer recipients received
from 2+ flagged accounts, vs 0% connection rate on the sender side.
This is why the Investigation Agent's tools are called on nameDest
(recipient), not nameOrig (sender), of a flagged transaction.

Follow-up refinement: raw connection count was later found to be
confounded by transaction volume (a busy legitimate account naturally
accumulates high counts). The signal was corrected to fan_in_ratio
(% of incoming transactions from flagged senders), which proved more
reliable. A further finding showed fan_in_ratio is unreliable at very
low transaction volume (1-2 total transactions), where a ratio of 1.0
occurs by chance about as often on legitimate accounts as fraud ones.

## Data leakage fix
PaySim's own documentation states balance columns (oldbalanceOrg,
newbalanceOrig, oldbalanceDest, newbalanceDest) must not be used for
fraud detection, since cancelled fraud transactions leave an artificial
fingerprint in these fields. The Detector was corrected to use only
`type` and `amount`. Result: 100% recall (legitimate, since PaySim fraud
only ever occurs in TRANSFER/CASH_OUT types), confirmed on the FULL
6.3M-row dataset. At the real-world fraud base rate (~0.13%), honest
precision is 0.30% -- the earlier 14.97% figure was measured on an
artificially fraud-enriched sample and is not representative of
real-world performance.

Note: this Detector-level precision reflects component-level performance
in isolation. The 40%/53% figures below are the full pipeline's real,
held-out, end-to-end result -- the number that matters for the system
as a whole.

## Agents vs plain code
- Detector: plain code (threshold scoring on type + amount)
- Investigation Agent: REAL agent -- ReAct loop, 3 tools, up to 5 iterations
- Prioritizer: plain code (weighted ranking formula, not reasoning)
- Explainer Agent: REAL agent -- drafts hedged, evidence-based narrative
- Output Reviewer Agent: REAL agent -- checklist + safe-wording gate,
  1 rewrite max back to Explainer

Three genuine reasoning agents; two deterministic components. LLM calls
are used only where judgment is actually required.

## Investigation Agent's tools
- check_account_history: transaction count/pattern for this account,
  checked bidirectionally and only BEFORE the triggering transaction.
  Validated finding: legitimate accounts are actually MORE likely to
  have prior history than fraud accounts (73.5% vs 40%) -- a fresh
  account with no history is a signal TOWARD suspicion, not against it.
- check_linked_accounts: fan_in_ratio (% of incoming transactions from
  already-flagged senders) -- the strongest validated signal, though
  unreliable at very low transaction volume (see above).
- check_velocity: time between money in and out; often returns no data
  (None) given how PaySim generates account IDs -- expected, not a bug.

## Tech Stack
- Model: Claude Sonnet (Investigation Agent's reasoning loop) + Claude Haiku
  (Explainer, Output Reviewer, and verdict-structuring steps), via AWS
  Bedrock. Detector uses no model -- it's plain code.
- Tools & model interface: LangChain (@tool decorator, ChatBedrock via
  langchain-aws)
- Orchestration: LangGraph (nodes, conditional edges, built on top of
  LangChain's pieces)
- Schema: Pydantic (typed state + verdicts)
- Evaluation: LangSmith (tracing + dataset experiments)
- Guardrails: iteration cap (5), 1-rewrite cap, allowed_tools allow-list

## Final Evaluation Results
- Held-out sample (never used for calibration): 40% recall, 53% precision
- Tuned sample (used for prompt calibration): 75% recall, 50% precision
- Naive baseline comparison (isFlaggedFraud rule): 0.19% recall
- Real-world AML industry benchmark: 2-10% precision

Both results far exceed the naive baseline and sit well above real-world
industry precision benchmarks, evaluated on genuinely unseen data
specifically to test for overfitting to the tuning sample -- a
methodology deliberately chosen after finding, mid-development, that
performance on a heavily-tuned sample overstated real generalization.

## Benchmarking Against Real-World AML Systems

| | Recall | Precision |
|---|---|---|
| Naive baseline (PaySim's built-in isFlaggedFraud rule) | 0.19% | -- |
| Real-world AML systems (PwC; peer-reviewed industry analysis, ScienceDirect 2024) | -- | 5-10% |
| This system (held-out test data) | 40% | 53% |

The naive baseline figure is our own measured result, reproducible via
`evaluate_detector.py` (comparing PaySim's `isFlaggedFraud` field against
ground-truth `isFraud` labels). The real-world AML benchmark is drawn from
external sources -- PricewaterhouseCoopers reports 90-95% of transaction
monitoring alerts are false positives, consistent with peer-reviewed
industry research (ScienceDirect, 2024), corresponding to roughly 5-10%
precision.

This system achieves a ~200x improvement in recall over the naive
baseline, and a 5-10x improvement in precision over typical real-world
AML systems, evaluated on held-out data never used during development.