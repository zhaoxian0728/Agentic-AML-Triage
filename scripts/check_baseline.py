import pandas as pd

file_path = "data/paysim_sample.csv"

df = pd.read_csv(file_path)

df["isFraud"] = pd.to_numeric(df["isFraud"])
df["isFlaggedFraud"] = pd.to_numeric(df["isFlaggedFraud"])

# Actual fraud cases
actual_fraud = df["isFraud"] == 1

# Cases flagged by PaySim's built-in rule
baseline_flagged = df["isFlaggedFraud"] == 1

# Correctly flagged fraud cases
true_positives = actual_fraud & baseline_flagged

total_actual_fraud = actual_fraud.sum()
correctly_flagged_fraud = true_positives.sum()

recall = correctly_flagged_fraud / total_actual_fraud

print(f"Total transactions: {len(df):,}")
print(f"Total actual fraud cases: {total_actual_fraud:,}")
print(f"Fraud cases correctly flagged: {correctly_flagged_fraud:,}")
print(f"Baseline recall: {recall:.6%}")
print(f"Baseline recall: {recall * 100:.2f}%")