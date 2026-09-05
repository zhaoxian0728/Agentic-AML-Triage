import os
from dotenv import load_dotenv
load_dotenv()
from langchain_aws import ChatBedrock
# from langchain_openai import ChatOpenAI
from src.schemas import InvestigationVerdict, ExplainerOutput

EXPLAINER_PROMPT = """You are drafting a brief narrative for a bank AML investigator,
based on a completed investigation. Use hedged, non-accusatory language — say
"activity is consistent with known mule patterns," never "this is money
laundering" or other definitive accusations. This mirrors how real Suspicious
Activity Reports must be worded for legal reasons.

Keep it to 2-4 sentences. Reference only the specific evidence provided —
never invent details not present in the evidence list.
"""


def explain(verdict: InvestigationVerdict) -> ExplainerOutput:
    model = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=os.getenv("AWS_REGION"),
    ).with_structured_output(ExplainerOutput)

    evidence_text = "\n".join(f"- {e}" for e in verdict.evidence)
    prompt = (
        f"{EXPLAINER_PROMPT}\n\nAccount: {verdict.account_id}\n"
        f"Confidence: {verdict.confidence}\nEvidence:\n{evidence_text}"
    )
    result = model.invoke(prompt)
    result.account_id = verdict.account_id
    return result