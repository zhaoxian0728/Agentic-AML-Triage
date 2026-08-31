from pydantic import BaseModel, Field
from typing import Literal


class TransactionRecord(BaseModel):
    """One row from PaySim — what comes into the Detector."""
    step: int
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float
    name_orig: str
    oldbalance_org: float
    newbalance_orig: float
    name_dest: str
    oldbalance_dest: float
    newbalance_dest: float


class DetectorVerdict(BaseModel):
    """Output of the Detector — decides auto-close vs investigate."""
    account_id: str
    score: float = Field(ge=0, le=1, description="Likelihood of mule pattern")
    reason: str
    route: Literal["auto_close", "investigate"]


class InvestigationVerdict(BaseModel):
    """Output of the Investigation Agent after its reasoning loop."""
    account_id: str
    confirmed: bool
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(description="Short evidence bullets from tool calls")
    reasoning_trail: str
    route: Literal["close_false_alarm", "escalate"]


class PriorityRank(BaseModel):
    """Output of the Prioritization Agent."""
    account_id: str
    urgency_score: float
    rank: int


class ExplainerOutput(BaseModel):
    """Output of the Explainer/SAR Agent (stretch goal)."""
    account_id: str
    narrative: str

class OutputReviewerVerdict(BaseModel):
    """Output of the Output Reviewer Agent — final gate before investigator."""
    account_id: str
    passed: bool
    issues: list[str] = Field(default_factory=list)
    action: Literal["pass", "request_rewrite", "flag_for_human_review"]