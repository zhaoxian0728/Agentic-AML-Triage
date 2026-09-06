from src.schemas import InvestigationVerdict, PriorityRank


def compute_urgency(verdict: InvestigationVerdict, amount: float,
                     is_fast_cashout: bool, fan_in_ratio: float) -> float:
    """Weighted urgency score, 0-1. Deterministic — ranking already-scored
    numeric risk factors doesn't need LLM reasoning, same principle as the Detector.
    Uses fan_in_ratio, not raw linked_count — count is confounded by transaction
    volume, ratio is the validated signal (proven more reliable earlier in
    development for the Investigation Agent's own reasoning)."""
    score = verdict.confidence * 0.4
    score += min(amount / 500000, 1.0) * 0.3
    score += 0.2 if is_fast_cashout else 0.0
    score += fan_in_ratio * 0.1
    return round(min(score, 1.0), 3)


def assign_ranks(scored_accounts: list[tuple[str, float]]) -> list[PriorityRank]:
    """Call this once, after a whole batch has been scored — rank is
    relative to the batch, so it can't be assigned per-account mid-graph."""
    ordered = sorted(scored_accounts, key=lambda x: x[1], reverse=True)
    return [
        PriorityRank(account_id=acc_id, urgency_score=score, rank=i + 1)
        for i, (acc_id, score) in enumerate(ordered)
    ]
