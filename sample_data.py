import pandas as pd

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Every account involved in a known fraud case, as either sender or receiver
fraud_rows = df[df["isFraud"] == 1]
fraud_accounts = set(fraud_rows["nameOrig"]) | set(fraud_rows["nameDest"])

# Pull ALL transactions belonging to those accounts — this is what lets
# get_account_history / check_velocity / check_linked_accounts actually work,
# since they need an account's full transaction trail, not just its one flagged row.
account_history = df[df["nameOrig"].isin(fraud_accounts) | df["nameDest"].isin(fraud_accounts)]

# Random unrelated legit transactions, for baseline volume/comparison
other_legit = df[
    (df["isFraud"] == 0)
    & (~df["nameOrig"].isin(fraud_accounts))
    & (~df["nameDest"].isin(fraud_accounts))
].sample(n=20000, random_state=42)

sample = pd.concat([account_history, other_legit]).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv("data/paysim_sample.csv", index=False)

print(f"Sample size: {len(sample)}")
print(f"Fraud rows: {(sample['isFraud']==1).sum()}")
print(f"Accounts with full history preserved: {len(fraud_accounts)}")