import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.tools import check_linked_accounts

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])

legit_recipients = set(df[df["isFraud"] == 0]["nameDest"])
false_positive_fan_in = sum(
    1 for r in list(legit_recipients)[:200]
    if check_linked_accounts(r, df, flagged_ids).linked_count >= 2
)
print(f"Of 200 legitimate recipients, {false_positive_fan_in} ALSO show 2+ flagged connections")

for threshold in [2, 3, 4, 5]:
    fraud_hits = sum(1 for r in list(set(df[df["isFraud"]==1]["nameDest"]))[:200]
                      if check_linked_accounts(r, df, flagged_ids).linked_count >= threshold)
    legit_hits = sum(1 for r in list(legit_recipients)[:200]
                      if check_linked_accounts(r, df, flagged_ids).linked_count >= threshold)
    print(f"threshold={threshold}: fraud={fraud_hits}/200, legit={legit_hits}/200")