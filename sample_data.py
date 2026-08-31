import pandas as pd

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Keep ALL fraud rows (they're rare — only ~0.13% of the data) plus a
# random sample of legit transactions. This isn't the "real" fraud rate
# anymore, but a pure random sample would only contain ~40 fraud rows
# out of 30k, which is too few to evaluate anything meaningfully against.
fraud = df[df["isFraud"] == 1]
legit = df[df["isFraud"] == 0].sample(n=30000, random_state=42)

sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv("data/paysim_sample.csv", index=False)

print(f"Sample size: {len(sample)}, fraud cases included: {len(fraud)}")