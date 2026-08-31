import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.features import engineer_features
from src.detector import score_transaction

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)

for threshold in [0.2, 0.3, 0.4, 0.5, 0.8]:
    predicted = (df["score"] >= threshold).astype(int)
    precision = precision_score(df["isFraud"], predicted)
    recall = recall_score(df["isFraud"], predicted)
    f1 = f1_score(df["isFraud"], predicted)
    flagged_pct = predicted.mean() * 100
    print(f"threshold={threshold}: flagged={flagged_pct:.1f}% of txns, "
          f"recall={recall:.2%} (fraud caught), precision={precision:.2%}, f1={f1:.2f}")

naive_recall = recall_score(df["isFraud"], df["isFlaggedFraud"])
naive_precision = precision_score(df["isFraud"], df["isFlaggedFraud"], zero_division=0)
print(f"\nnaive isFlaggedFraud rule: recall={naive_recall:.2%}, precision={naive_precision:.2%}")