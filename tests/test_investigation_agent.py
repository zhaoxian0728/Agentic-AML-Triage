import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent, investigate_account

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

agent = build_investigation_agent(df, flagged_ids)

fraud_transfer = df[(df["isFraud"] == 1) & (df["type"] == "TRANSFER")].iloc[0]
recipient = fraud_transfer["nameDest"]

verdict = investigate_account(agent, recipient)
print("\n" + "=" * 50)
print(f"Account:     {verdict.account_id}")
print(f"Confirmed:   {verdict.confirmed}")
print(f"Confidence:  {verdict.confidence:.0%}")
print(f"Route:       {verdict.route}")
print("\nEvidence:")
for item in verdict.evidence:
    print(f"  - {item}")
print(f"\nReasoning:\n{verdict.reasoning_trail}")
print("=" * 50 + "\n")