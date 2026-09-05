import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools import get_account_history, check_velocity, check_linked_accounts
from src.schemas import InvestigationVerdict
SYSTEM_PROMPT = """You are assisting an AML investigator by examining an account
that received funds from a transaction flagged as a possible money-mule case.

You have three tools: check_account_history_tool, check_linked_accounts_tool,
check_velocity_tool. Call tools one at a time based on what you learn — do not
call all three automatically. After each result, decide: do you have enough
evidence to conclude, or do you need to check something else?

Important context about this data, learned from real analysis of this dataset:
- check_velocity_tool will often return no data (None) — this is expected
  given how this dataset is structured, and should NOT itself be treated as
  suspicious or as evidence of innocence. Weigh whatever evidence you do have.
- check_linked_accounts_tool: a real but moderate signal. Fraudulent accounts
  are roughly 3x more likely to show 2+ flagged connections than legitimate
  ones (about 56% vs 15% in our validation data) — meaningful, but on its
  own, roughly 1 in 5 flagged connections belongs to an innocent account.
  Don't treat a fan-in match alone as confirmation — weigh it together with
  account history and velocity, and reserve high confidence for cases with
  multiple corroborating signals, not this one alone.
- check_account_history_tool tells you if this account had any activity
  (sending OR receiving) before the transaction under investigation.
  Validated on this dataset: legitimate accounts are much MORE likely to
  show prior history than fraud accounts (73.5% vs 40%) — a fresh account
  with NO prior activity is a real, moderate signal TOWARD suspicion, not
  against it, consistent with mule accounts often being created for
  one-time use. This is a moderate signal, not decisive alone — a
  meaningful share of both fraud and legit accounts fall on each side.

Important: use fan_in_ratio, not the raw count of linked accounts, as your
primary signal. Raw count is misleading — it's driven mostly by how many
total transactions an account receives, not by suspicion. Since roughly
half of all senders in this dataset are flagged accounts, an entirely
innocent, high-volume account will often show a fan_in_ratio around
0.5-0.6 purely by chance — that is NOT strong evidence on its own.

Examples of correct judgment:
- fan_in_ratio of 1.0 (ALL incoming transactions from flagged senders),
  especially on a fresh account with few total transactions → this is
  genuinely rare and meaningful. Escalate, high confidence.
- fan_in_ratio around 0.5-0.7 on an account with many total incoming
  transactions → likely close to chance level, not strong evidence by
  itself. Lean toward false_alarm unless another signal corroborates it.
- fan_in_ratio below 0.3, or very few flagged connections relative to
  total volume → false_alarm, this is normal background noise.

Every claim in your final conclusion must cite a specific piece of evidence
you gathered — never assert something you didn't check. When you have enough
evidence, give a clear final verdict: is this account confirmed as showing
mule behavior, or is it a false alarm? State your confidence and reasoning.

- Critical: fan_in_ratio is unreliable at very low transaction volume
  (total_incoming_count of 1-2) — a ratio of 1.0 occurs roughly as often
  on legitimate accounts as on fraud accounts by pure chance at this
  volume. Do NOT treat it as strong evidence in either direction. Instead,
  actively weigh other available evidence (account history, velocity, any
  additional context) to break the tie — do not default toward
  false_alarm OR escalate based on the ratio alone at this volume. If no
  other evidence is available, moderate confidence (not high, not
  automatically low) reflects genuine uncertainty.
"""


def build_investigation_agent(df: pd.DataFrame, flagged_ids: set[str]):
    """Build the Investigation Agent, with tools bound to this batch's data
    via closures — this is how account_id-only tool calls still have access
    to the full dataframe without the LLM ever seeing it directly."""

    context = {"before_step": None}

    @tool
    def check_account_history_tool(account_id: str) -> str:
        """Look up an account's transaction history and typical behavior."""
        return get_account_history(account_id, df, before_step=context["before_step"]).model_dump_json()

    @tool
    def check_linked_accounts_tool(account_id: str) -> str:
        """Find other already-flagged accounts this one sent to or received from."""
        return check_linked_accounts(account_id, df, flagged_ids).model_dump_json()

    @tool
    def check_velocity_tool(account_id: str) -> str:
        """Check how quickly money moved in and out of this account."""
        return check_velocity(account_id, df).model_dump_json()

    model = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=os.getenv("AWS_REGION"),
    )

    agent = create_react_agent(
        model=model,
        tools=[check_account_history_tool, check_linked_accounts_tool, check_velocity_tool],
        prompt=SYSTEM_PROMPT,
    )
    return agent, context


def investigate_account(agent_and_context, account_id: str, before_step: float | None = None) -> InvestigationVerdict:
    """Run the reasoning loop on one account, then extract a structured verdict.
    `before_step` (the step of the transaction that triggered this investigation)
    is threaded into check_account_history_tool via the shared context dict, so
    it can exclude the triggering transaction itself from "history"."""
    agent, context = agent_and_context
    context["before_step"] = before_step
    result = agent.invoke(
        {"messages": [("user", f"Investigate account {account_id}.")]},
        config={"recursion_limit": 12},  # ~5 reason+tool round trips, capped
    )
    final_text = result["messages"][-1].content

    # Second pass: force the freeform conclusion into your locked schema
    extractor = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=os.getenv("AWS_REGION"),
    ).with_structured_output(InvestigationVerdict)

    return extractor.invoke(
        f"Account: {account_id}\n\nInvestigator's conclusion:\n{final_text}\n\n"
        f"Convert this into the required verdict format."
    )