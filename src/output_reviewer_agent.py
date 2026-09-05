import os
from dotenv import load_dotenv
load_dotenv()
from langchain_aws import ChatBedrock
from src.schemas import ExplainerOutput, InvestigationVerdict, OutputReviewerVerdict

REVIEWER_PROMPT = """Review this narrative before it reaches a human AML investigator.
Check: (1) it references real evidence, not generic statements, (2) its tone
matches the numeric confidence — a 0.5 confidence should read as tentative,
not confirmed, (3) wording is hedged, never a definitive accusation like
"this is money laundering", (4) no leaked internal reasoning artifacts.

If it fails any check: passed=false, list the issues, action="request_rewrite".
If it passes: action="pass". You never change the underlying fraud verdict —
only the presentation.
"""


def review_output(narrative: ExplainerOutput, verdict: InvestigationVerdict) -> OutputReviewerVerdict:
    model = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=os.getenv("AWS_REGION"),
    ).with_structured_output(OutputReviewerVerdict)

    prompt = f"{REVIEWER_PROMPT}\n\nConfidence score: {verdict.confidence}\nNarrative:\n{narrative.narrative}"
    result = model.invoke(prompt)
    result.account_id = verdict.account_id
    return result