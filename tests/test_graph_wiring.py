import pandas as pd
from unittest.mock import patch
from src.schemas import InvestigationVerdict, ExplainerOutput, OutputReviewerVerdict
from src.features import engineer_features
from src.detector import score_transaction
from src.graph import build_graph

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

batch = df[df["score"] >= 0.3].sample(n=5, random_state=1)


def fake_investigate_account(agent, account_id):
    return InvestigationVerdict(
        account_id=account_id, confirmed=True, confidence=0.8,
        evidence=["stub: received from 2 flagged accounts"],
        reasoning_trail="Stub response — testing graph wiring only.",
        route="escalate",
    )


def fake_explain(verdict):
    return ExplainerOutput(account_id=verdict.account_id, narrative="Stub narrative for wiring test.")


def fake_review_output(narrative, verdict):
    return OutputReviewerVerdict(account_id=verdict.account_id, passed=True, issues=[], action="pass")


with patch("src.graph.investigate_account", side_effect=fake_investigate_account), \
     patch("src.graph.explain", side_effect=fake_explain), \
     patch("src.graph.review_output", side_effect=fake_review_output):

    graph = build_graph(df, flagged_ids, investigation_agent=None)

    for _, row in batch.iterrows():
        state = {"transaction": row.to_dict(), "recipient_id": row["nameDest"], "rewrite_count": 0}
        result = graph.invoke(state)
        reached_end = result.get("urgency_score") is not None
        print(f"Account {row['nameDest']}: reached full pipeline = {reached_end}, "
              f"reviewer action = {result.get('reviewer_verdict').action if result.get('reviewer_verdict') else 'N/A'}")