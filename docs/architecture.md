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

## Data leakage fix
PaySim's own documentation states balance columns (oldbalanceOrg,
newbalanceOrig, oldbalanceDest, newbalanceDest) must not be used for
fraud detection, since cancelled fraud transactions leave an artificial
fingerprint in these fields. The Detector was corrected to use only
`type` and `amount`. Result: 100% recall (legitimate, since PaySim fraud
only ever occurs in TRANSFER/CASH_OUT types), 14.97% precision, 36%
auto-clear rate -- an honest number, not the earlier leaked 100%/38.5%.

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
- check_account_history: transaction count/pattern for this account
- check_linked_accounts: STRONGEST signal -- other flagged accounts
  connected via sent/received funds
- check_velocity: time between money in and out; often returns no data
  (None) given how PaySim generates account IDs -- expected, not a bug

## Tech Stack
- Model: Claude Haiku (Detector-adjacent/fast tasks) + Claude Sonnet
  (Investigation Agent), via AWS Bedrock
- Orchestration: LangGraph (nodes, conditional edges)
- Schema: Pydantic (typed state + verdicts)
- Evaluation: LangSmith (tracing + dataset experiments)
- Guardrails: iteration cap (5), 1-rewrite cap, allowed_tools allow-list