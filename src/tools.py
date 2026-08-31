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


class VelocityResult(BaseModel):
    account_id: str
    hours_between_in_and_out: float | None
    is_fast_cashout: bool


def get_account_history(account_id: str, df: pd.DataFrame) -> AccountHistory:
    """Look up this account's other transactions to establish a baseline."""
    account_txns = df[df["nameOrig"] == account_id]
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
    """Find other flagged accounts this one sent money to or received from —
    the basis for spotting a mule ring, not just a single account."""
    sent_to = set(df[df["nameOrig"] == account_id]["nameDest"])
    received_from = set(df[df["nameDest"] == account_id]["nameOrig"])
    connected = (sent_to | received_from) & flagged_ids
    return LinkedAccounts(
        account_id=account_id,
        linked_flagged_accounts=list(connected),
        linked_count=len(connected),
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