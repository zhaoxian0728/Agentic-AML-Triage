import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SAFE — uses only `type`, not the contaminated balance columns
    df["is_transfer_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    return df