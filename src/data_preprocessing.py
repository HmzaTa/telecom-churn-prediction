"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and preprocessing of the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# DATA LOADING

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download the Telco Customer Churn dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and place it at: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv\n\n"
            "Alternatively, run: python src/download_data.py"
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def download_sample_data(save_path: str) -> pd.DataFrame:
    """
    Generate a realistic synthetic Telco churn dataset for demo/testing.
    Mirrors the schema of the real Kaggle dataset.

    Parameters
    ----------
    save_path : str
        File path where the synthetic CSV will be saved.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset.
    """
    np.random.seed(42)
    n = 7043  # same size as real dataset

    customer_ids = [f"{''.join(np.random.choice(list('0123456789ABCDEF'), 4))}-{''.join(np.random.choice(list('ABCDEFGHIJ'), 5))}" for _ in range(n)]

    tenure = np.random.randint(0, 73, n)
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Churn logic: month-to-month, high charges, short tenure → higher churn
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    contract_risk = np.where(contracts == "Month-to-month", 0.35,
                    np.where(contracts == "One year", 0.12, 0.05))
    tenure_factor = np.clip(1 - tenure / 72, 0.05, 1.0)
    charge_factor = np.clip((monthly_charges - 18) / 100, 0, 0.3)
    churn_prob = np.clip(contract_risk * tenure_factor + charge_factor, 0, 0.95)
    churn = np.random.binomial(1, churn_prob)

    internet = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n, p=[0.48, 0.52]),
        "Dependents": np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22]),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
        "Contract": contracts,
        "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": np.where(tenure == 0, "", total_charges.astype(str)),  # mimic real dataset quirk
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Synthetic dataset saved → {save_path}  ({n} rows)")
    return df


# DATA CLEANING

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Drop customerID (non-predictive)
      - Fix TotalCharges (may be empty string for new customers)
      - Encode target variable Churn → 0/1
      - Drop any duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Drop customerID – not a predictive feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges: coerce to numeric, fill NaN with 0 (new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    if missing_tc > 0:
        print(f"[INFO] TotalCharges: {missing_tc} missing values filled with 0 (new customers, tenure=0)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} duplicate rows")

    print(f"[INFO] Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Churn rate: {df['Churn'].mean():.2%}")
    return df


# ENCODING

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables:
      - Binary columns (Yes/No, Male/Female) → Label Encoding (0/1)
      - Multi-class columns → One-Hot Encoding

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (after feature engineering).

    Returns
    -------
    (pd.DataFrame, dict)
        Encoded dataframe and a metadata dict for reference.
    """
    df = df.copy()
    encoding_meta = {}

    # Columns that are already numeric – skip
    skip_cols = ["Churn", "tenure", "MonthlyCharges", "TotalCharges",
                 "SeniorCitizen", "AverageChargePerMonth", "TenureGroup_encoded"]

    # Binary Yes/No columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "gender"
    ]

    for col in binary_cols:
        if col not in df.columns:
            continue
        mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        df[col] = df[col].map(mapping)
        encoding_meta[col] = mapping

    # Multi-class columns → One-Hot Encoding
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    multi_cols = [c for c in multi_cols if c in df.columns]

    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)
    encoding_meta["one_hot_encoded"] = multi_cols

    # Convert bool columns to int (pandas get_dummies returns bool in newer versions)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[INFO] After encoding: {df.shape[1]} features")
    return df, encoding_meta


# SCALING

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   cols_to_scale: list[str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to numerical columns.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    cols_to_scale : list[str] | None
        Columns to scale. Defaults to continuous numeric columns.

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    if cols_to_scale is None:
        cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges", "AverageChargePerMonth"]
        cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(f"[INFO] Scaled columns: {cols_to_scale}")
    return X_train, X_test, scaler


# TRAIN / TEST SPLIT

def split_data(df: pd.DataFrame, target: str = "Churn",
               test_size: float = 0.2, random_state: int = 42
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[INFO] Train churn rate: {y_train.mean():.2%}  |  Test churn rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# FULL PREPROCESSING PIPELINE

def full_preprocessing_pipeline(filepath: str, processed_dir: str = "data/processed"
                                  ) -> tuple:
    """
    Execute the complete preprocessing pipeline:
        load → clean → feature_engineer → encode → split → scale

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    from feature_engineering import engineer_features  # local import to avoid circular

    os.makedirs(processed_dir, exist_ok=True)

    # Load
    df_raw = load_data(filepath)

    # Clean
    df_clean = clean_data(df_raw)

    # Feature engineering
    df_feat = engineer_features(df_clean)

    # Encode
    df_encoded, _ = encode_features(df_feat)

    # Save processed
    processed_path = os.path.join(processed_dir, "telco_processed.csv")
    df_encoded.to_csv(processed_path, index=False)
    print(f"[INFO] Processed dataset saved → {processed_path}")

    # Split
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Fill any remaining NaN values before scaling/training
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    feature_names = X_train_scaled.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
