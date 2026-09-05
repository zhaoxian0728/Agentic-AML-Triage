import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent
from src.graph import build_graph

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

# Known fraud cases, plus the HARD legit cases — ones that fooled the
# Detector into flagging them, so we're actually testing whether the
# Investigation Agent can correctly clear a false positive, not just
# easy cases that never reach it.
fraud_sample = df[df["isFraud"] == 1].sample(n=20, random_state=3)
hard_legit_sample = df[(df["isFraud"] == 0) & (df["score"] >= 0.3)].sample(n=20, random_state=3)
test_batch = pd.concat([fraud_sample, hard_legit_sample])

agent = build_investigation_agent(df, flagged_ids)
graph = build_graph(df, flagged_ids, agent)

results = []
for _, row in test_batch.iterrows():
    state = {"transaction": row.to_dict(), "recipient_id": row["nameDest"], "rewrite_count": 0}
    final_state = graph.invoke(state)

    iv = final_state.get("investigation_verdict")
    predicted_fraud = iv.confirmed if iv else False  # never reached investigation = auto-closed = predicted not-fraud
    actual_fraud = bool(row["isFraud"])
    stage = "confirmed" if iv and iv.confirmed else ("cleared_by_agent" if iv else "auto_closed")

    results.append({"actual": actual_fraud, "predicted": predicted_fraud})
    print(f"{row['nameDest']}: actual={actual_fraud}, predicted={predicted_fraud}, stage={stage}, "
          f"{'✓' if actual_fraud == predicted_fraud else '✗ WRONG'}")

r = pd.DataFrame(results)
tp = ((r.actual) & (r.predicted)).sum()
fn = ((r.actual) & (~r.predicted)).sum()
fp = ((~r.actual) & (r.predicted)).sum()
tn = ((~r.actual) & (~r.predicted)).sum()

print(f"\nTP={tp} FN={fn} FP={fp} TN={tn}")
print(f"End-to-end recall:    {tp/(tp+fn) if (tp+fn)>0 else 0:.0%}")
print(f"End-to-end precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.0%}")