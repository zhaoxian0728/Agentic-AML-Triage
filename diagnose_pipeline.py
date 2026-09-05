import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent
from src.graph import build_graph

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

fraud_sample = df[df["isFraud"] == 1].sample(n=20, random_state=2)
hard_legit_sample = df[(df["isFraud"] == 0) & (df["score"] >= 0.3)].sample(n=20, random_state=2)
test_batch = pd.concat([fraud_sample, hard_legit_sample])

agent = build_investigation_agent(df, flagged_ids)
graph = build_graph(df, flagged_ids, agent)

counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

for _, row in test_batch.iterrows():
    try:
        state = {"transaction": row.to_dict(), "recipient_id": row["nameDest"], "rewrite_count": 0}
        final_state = graph.invoke(state)
    except Exception as e:
        print(f"\n!!! FAILED on {row['nameDest']}: {e}\n(likely expired token — refresh .env and re-run; earlier results above are safe)")
        break  # stop cleanly instead of losing everything to a traceback

    iv = final_state.get("investigation_verdict")
    predicted_fraud = iv.confirmed if iv else False
    actual_fraud = bool(row["isFraud"])

    if actual_fraud and predicted_fraud: category = "TP"
    elif not actual_fraud and not predicted_fraud: category = "TN"
    elif not actual_fraud and predicted_fraud: category = "FP"
    else: category = "FN"
    counts[category] += 1

    print(f"\n{'='*70}\n{row['nameDest']} — {category} (confidence: {iv.confidence if iv else 'N/A'})\n{'='*70}")
    if iv:
        print("Evidence:")
        for item in iv.evidence:
            print(f"  - {item}")
        print(f"Reasoning: {iv.reasoning_trail}")
    else:
        print("N/A — auto-closed by Detector, never reached agent")

print(f"\n\nSUMMARY SO FAR: {counts}")