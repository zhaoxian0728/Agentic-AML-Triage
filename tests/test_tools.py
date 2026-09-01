import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.tools import get_account_history, check_velocity, check_linked_accounts

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)

# Build the flagged set the same way the real pipeline will —
# accounts the Detector sent to investigation (score >= 0.3)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])
print(f"Flagged accounts: {len(flagged_ids)}")

# Test against one real fraud TRANSFER's sender and recipient
fraud_transfer = df[(df["isFraud"] == 1) & (df["type"] == "TRANSFER")].iloc[0]
sender = fraud_transfer["nameOrig"]
recipient = fraud_transfer["nameDest"]

print(f"\n--- Sender: {sender} ---")
print(get_account_history(sender, df))
print(check_velocity(sender, df))
print(check_linked_accounts(sender, df, flagged_ids))

print(f"\n--- Recipient: {recipient} ---")
print(get_account_history(recipient, df))
print(check_velocity(recipient, df))
print(check_linked_accounts(recipient, df, flagged_ids))

# Diagnostic: do flagged accounts actually connect to each other at all?
# (sampling 200 for speed rather than checking all flagged accounts)
connected_count = 0
for acc in list(flagged_ids)[:200]:
    result = check_linked_accounts(acc, df, flagged_ids)
    if result.linked_count > 0:
        connected_count += 1
print(f"\nOf 200 sampled flagged accounts, {connected_count} connect to another flagged account")

recipients = set(df[df["isFraud"] == 1]["nameDest"])
fan_in_count = sum(
    1 for r in list(recipients)[:200]
    if check_linked_accounts(r, df, flagged_ids).linked_count >= 2
)
print(f"Of 200 fraud-transfer recipients, {fan_in_count} received from 2+ flagged accounts")