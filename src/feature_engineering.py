"""
feature_engineering.py
-----------------------
Creates new informative features from the cleaned Telco dataset.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that may improve churn prediction:

    New features:
        - AverageChargePerMonth : TotalCharges / tenure (0 for tenure=0)
        - TenureGroup            : categorical bucket (New / Medium / Long)
        - TenureGroup_encoded    : numeric label for TenureGroup
        - HasProtection          : 1 if customer has OnlineSecurity or TechSupport
        - NumAddonServices       : count of add-on services subscribed
        - ChargePerService       : MonthlyCharges / (NumAddonServices + 1)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (Churn encoded as 0/1).

    Returns
    -------
    pd.DataFrame
        Dataset with additional features.
    """
    df = df.copy()

    # Average monthly charge relative to total tenure
    df["AverageChargePerMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]          # for tenure=0 use monthly rate
    )
    df["AverageChargePerMonth"] = df["AverageChargePerMonth"].round(2)

    # Tenure groups
    def _tenure_group(tenure: int) -> str:
        if tenure <= 12:
            return "New"
        elif tenure <= 36:
            return "Medium"
        else:
            return "Long"

    df["TenureGroup"] = df["tenure"].apply(_tenure_group)

    # Ordinal encoding for tree models
    group_map = {"New": 0, "Medium": 1, "Long": 2}
    df["TenureGroup_encoded"] = df["TenureGroup"].map(group_map)

    # Drop string TenureGroup (will be one-hot encoded separately if desired,
    # but ordinal is more informative here)
    df.drop(columns=["TenureGroup"], inplace=True)

    # Has any protection/support service
    support_cols = ["OnlineSecurity", "TechSupport"]
    existing = [c for c in support_cols if c in df.columns]
    if existing:
        df["HasProtection"] = (
            df[existing].isin(["Yes"]).any(axis=1).astype(int)
        )

    # Number of add-on streaming / security services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_addon = [c for c in addon_cols if c in df.columns]
    if existing_addon:
        df["NumAddonServices"] = (
            df[existing_addon].isin(["Yes"]).sum(axis=1)
        )
        df["ChargePerService"] = (
            df["MonthlyCharges"] / (df["NumAddonServices"] + 1)
        ).round(2)

    print(f"[INFO] Feature engineering complete. Shape: {df.shape}")
    return df
