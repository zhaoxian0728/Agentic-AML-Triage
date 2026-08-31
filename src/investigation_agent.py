import pandas as pd
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
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
- check_linked_accounts_tool is your strongest signal: an account receiving
  from 2 or more OTHER accounts that are already flagged is a strong
  indicator of a mule collection point (a "fan-in" pattern).
- check_account_history_tool tells you if this account has a track record
  or is a fresh/one-off account — fresh accounts with no history are more
  typical of mule accounts.

Every claim in your final conclusion must cite a specific piece of evidence
you gathered — never assert something you didn't check. When you have enough
evidence, give a clear final verdict: is this account confirmed as showing
mule behavior, or is it a false alarm? State your confidence and reasoning.
"""


def build_investigation_agent(df: pd.DataFrame, flagged_ids: set[str]):
    """Build the Investigation Agent, with tools bound to this batch's data
    via closures — this is how account_id-only tool calls still have access
    to the full dataframe without the LLM ever seeing it directly."""

    @tool
    def check_account_history_tool(account_id: str) -> str:
        """Look up an account's transaction history and typical behavior."""
        return get_account_history(account_id, df).model_dump_json()

    @tool
    def check_linked_accounts_tool(account_id: str) -> str:
        """Find other already-flagged accounts this one sent to or received from."""
        return check_linked_accounts(account_id, df, flagged_ids).model_dump_json()

    @tool
    def check_velocity_tool(account_id: str) -> str:
        """Check how quickly money moved in and out of this account."""
        return check_velocity(account_id, df).model_dump_json()

    model = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="ap-southeast-1",
    )

    return create_react_agent(
        model=model,
        tools=[check_account_history_tool, check_linked_accounts_tool, check_velocity_tool],
        prompt=SYSTEM_PROMPT,
    )


def investigate_account(agent, account_id: str) -> InvestigationVerdict:
    """Run the reasoning loop on one account, then extract a structured verdict."""
    result = agent.invoke(
        {"messages": [("user", f"Investigate account {account_id}.")]},
        config={"recursion_limit": 12},  # ~5 reason+tool round trips, capped
    )
    final_text = result["messages"][-1].content

    # Second pass: force the freeform conclusion into your locked schema
    extractor = ChatBedrock(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="ap-southeast-1",
    ).with_structured_output(InvestigationVerdict)

    return extractor.invoke(
        f"Account: {account_id}\n\nInvestigator's conclusion:\n{final_text}\n\n"
        f"Convert this into the required verdict format."
    )