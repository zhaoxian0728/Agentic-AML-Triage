import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent
from prioritizer import assign_ranks
from src.graph import build_graph

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

# Small curated batch for a real test run — NOT the full ~21k flagged set
batch = df[df["score"] >= 0.3].sample(n=10, random_state=1)

agent = build_investigation_agent(df, flagged_ids)
graph = build_graph(df, flagged_ids, agent)

results = []
for _, row in batch.iterrows():
    try:
        state = {"transaction": row.to_dict(), "recipient_id": row["nameDest"], "rewrite_count": 0}
        final_state = graph.invoke(state)
        if final_state.get("urgency_score") is not None:
            results.append((final_state["investigation_verdict"], final_state))
    except Exception as e:
        print(f"Skipped {row['nameDest']}: {e}")
        continue

ranked = assign_ranks([(fs["investigation_verdict"].account_id, fs["urgency_score"]) for _, fs in results])
for rank in ranked:
    matching = next(fs for v, fs in results if v.account_id == rank.account_id)
    reviewer_action = matching.get("reviewer_verdict").action if matching.get("reviewer_verdict") else "N/A"
    print(f"#{rank.rank} | {rank.account_id} | urgency={rank.urgency_score} | reviewer={reviewer_action} | "
          f"{matching['explainer_output'].narrative}")