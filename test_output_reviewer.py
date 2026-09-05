from src.schemas import ExplainerOutput, InvestigationVerdict
from src.output_reviewer_agent import review_output

# A deliberately BAD narrative — overly definitive, matching the exact
# "strongly suggest involvement in unlawful activities" over-claiming
# language you caught earlier tonight, on a LOW confidence verdict.
bad_verdict = InvestigationVerdict(
    account_id="TEST123", confirmed=True, confidence=0.35,
    evidence=["Fan-in ratio of 0.4"],
    reasoning_trail="Weak evidence overall.",
    route="escalate",
)
bad_narrative = ExplainerOutput(
    account_id="TEST123",
    narrative="This account is definitely engaged in money laundering. "
              "This is confirmed illegal activity requiring immediate law "
              "enforcement action."
)

result = review_output(bad_narrative, bad_verdict)
print(f"passed: {result.passed}")
print(f"action: {result.action}")
print(f"issues: {result.issues}")

good_verdict = InvestigationVerdict(
    account_id="TEST456", confirmed=True, confidence=0.85,
    evidence=["Fan-in ratio of 1.0 from 4 flagged accounts", "Fresh account, no prior history"],
    reasoning_trail="Multiple corroborating signals.",
    route="escalate",
)
good_narrative = ExplainerOutput(
    account_id="TEST456",
    narrative="Account activity is consistent with known mule patterns — "
              "specifically, a fresh account receiving funds exclusively from "
              "flagged sources warrants investigator review."
)
result2 = review_output(good_narrative, good_verdict)
print(f"passed: {result2.passed}, action: {result2.action}")

print(f"\npassed: {result2.passed}, action: {result2.action}")
print(f"issues: {result2.issues}")