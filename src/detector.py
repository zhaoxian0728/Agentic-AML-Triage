import pandas as pd
from src.schemas import DetectorVerdict


def score_transaction(row: pd.Series) -> float:
    """Rule-based mule-likelihood score, 0-1. Uses only `type` and `amount` —
    both confirmed safe by the dataset's own documentation.

    Amount thresholds are informed by real distribution differences: fraud
    transactions average $1.47M vs $344K for legitimate transfers (median
    $441K vs $181K) — amount is real signal, but deliberately does NOT gate
    the investigate/auto-close decision, since ~25% of fraud cases fall
    below $200K and would be wrongly auto-closed if amount alone decided."""
    if row["is_transfer_type"] != 1:
        return 0.0
    score = 0.4
    if row["amount"] > 200000:
        score += 0.3
    if row["amount"] > 500000:
        score += 0.2
    return min(score, 1.0)


def route(score: float, threshold: float = 0.3) -> str:
    return "investigate" if score >= threshold else "auto_close"


def run_detector(df: pd.DataFrame, threshold: float = 0.3) -> list[DetectorVerdict]:
    verdicts = []
    for _, row in df.iterrows():
        score = score_transaction(row)
        verdicts.append(
            DetectorVerdict(
                account_id=row["nameOrig"], score=score,
                reason=f"type={row['type']}, amount={row['amount']:.0f}",
                route=route(score, threshold),
            )
        )
    return verdicts