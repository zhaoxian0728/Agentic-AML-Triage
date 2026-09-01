import pandas as pd
from src.features import engineer_features
from src.detector import score_transaction
from src.investigation_agent import build_investigation_agent, investigate_account

df = pd.read_csv("data/paysim_sample.csv")
df = engineer_features(df)
df["score"] = df.apply(score_transaction, axis=1)
flagged_ids = set(df[df["score"] >= 0.3]["nameOrig"])
agent = build_investigation_agent(df, flagged_ids)

wrong_cases = {
    # missed real fraud (false negatives)
    "C447486475": "missed real fraud",
    "C961098321": "missed real fraud",
    "C1047213176": "missed real fraud",
    "C597468231": "missed real fraud",
    "C1882040140": "missed real fraud",
    # wrongly confirmed legit accounts (false positives)
    "C1850704979": "wrongly confirmed legit",
    "C836514715": "wrongly confirmed legit",
    "C1818870755": "wrongly confirmed legit",
    "C1986090456": "wrongly confirmed legit",
    "C1663207814": "wrongly confirmed legit",
}

for account_id, error_type in wrong_cases.items():
    print(f"\n{'='*60}\n{account_id} ({error_type})\n{'='*60}")
    verdict = investigate_account(agent, account_id)
    print(f"Confirmed: {verdict.confirmed}, Confidence: {verdict.confidence:.0%}")
    print("Evidence:")
    for e in verdict.evidence:
        print(f"  - {e}")
    print(f"Reasoning: {verdict.reasoning_trail}")