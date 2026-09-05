import pandas as pd
from pydantic import BaseModel


class AccountHistory(BaseModel):
    account_id: str
    found: bool
    transaction_count: int
    avg_amount: float
    typical_type: str


class LinkedAccounts(BaseModel):
    account_id: str
    linked_flagged_accounts: list[str]
    linked_count: int
    total_incoming_count: int
    fan_in_ratio: float


class VelocityResult(BaseModel):
    account_id: str
    hours_between_in_and_out: float | None
    is_fast_cashout: bool


def get_account_history(account_id: str, df: pd.DataFrame, before_step: float | None = None) -> AccountHistory:
    """Look up this account's other transactions to establish a baseline.
        Checks BOTH directions (sent or received) — checking only nameOrig
        (sending history) is ~99.9% empty for every account regardless of
        fraud status in this dataset, since most accounts only ever appear
        once as a recipient, so it carried no signal. Bidirectional history
        is the version that's actually informative: validated at 40% of
        fraud recipients having PRIOR activity vs 73.5% of legit recipients —
        legit accounts are MORE likely to have prior history, not less; a
        fresh account is a real signal toward suspicion, not against it.

        `before_step` should be the step of the transaction that triggered
        this investigation — without it, a recipient's own triggering
        transaction would count as "history" for itself, making every
        account look non-fresh and erasing the signal."""
    account_txns = df[(df["nameOrig"] == account_id) | (df["nameDest"] == account_id)]
    if before_step is not None:
        account_txns = account_txns[account_txns["step"] < before_step]
    if account_txns.empty:
        return AccountHistory(
            account_id=account_id, found=False,
            transaction_count=0, avg_amount=0.0, typical_type="none"
        )
    return AccountHistory(
        account_id=account_id, found=True,
        transaction_count=len(account_txns),
        avg_amount=float(account_txns["amount"].mean()),
        typical_type=account_txns["type"].mode()[0],
    )


def check_linked_accounts(account_id: str, df: pd.DataFrame, flagged_ids: set[str]) -> LinkedAccounts:
    """Ratio of incoming transactions from flagged senders, not raw count —
    raw count is confounded with how many total transactions an account
    receives, which misleads toward flagging busy legitimate accounts and
    clearing fresh low-volume fraud accounts."""
    incoming = df[df["nameDest"] == account_id]
    total_incoming = len(incoming)
    received_from = set(incoming["nameOrig"]) & flagged_ids
    return LinkedAccounts(
        account_id=account_id,
        linked_flagged_accounts=list(received_from),
        linked_count=len(received_from),
        total_incoming_count=total_incoming,
        fan_in_ratio=len(received_from) / total_incoming if total_incoming > 0 else 0.0,
    ) 


def check_velocity(account_id: str, df: pd.DataFrame) -> VelocityResult:
    """How fast did money leave after it arrived — the core mule signature."""
    incoming = df[df["nameDest"] == account_id]
    outgoing = df[(df["nameOrig"] == account_id) & (df["type"].isin(["TRANSFER", "CASH_OUT"]))]

    if incoming.empty or outgoing.empty:
        return VelocityResult(account_id=account_id, hours_between_in_and_out=None, is_fast_cashout=False)

    first_in = incoming["step"].min()
    first_out_after = outgoing[outgoing["step"] >= first_in]["step"].min()

    if pd.isna(first_out_after):
        return VelocityResult(account_id=account_id, hours_between_in_and_out=None, is_fast_cashout=False)

    hours = float(first_out_after - first_in)
    return VelocityResult(account_id=account_id, hours_between_in_and_out=hours, is_fast_cashout=hours <= 2)