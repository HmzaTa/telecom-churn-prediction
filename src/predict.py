"""
src/predict.py
--------------
Command-line tool for predicting churn on new customer data.

Usage — single customer (interactive):
    python src/predict.py

Usage — batch (CSV file):
    python src/predict.py --input data/new_customers.csv --output data/predictions.csv

Usage — example profiles:
    python src/predict.py --examples
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from train_model import load_model, predict_churn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")


# EXAMPLE PROFILES
EXAMPLE_CUSTOMERS = pd.DataFrame([
    {
        "customerID": "RISK-HIGH-001",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50, "TotalCharges": "191.0",
    },
    {
        "customerID": "RISK-MED-002",
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 14, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 73.45, "TotalCharges": "1028.3",
    },
    {
        "customerID": "RISK-LOW-003",
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 64, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 64.00, "TotalCharges": "4096.0",
    },
])


# INTERACTIVE SINGLE PREDICTION
def _prompt(prompt: str, choices: list[str] | None = None, default: str = "") -> str:
    if choices:
        opts = " / ".join(choices)
        val = input(f"  {prompt} [{opts}] (default: {default or choices[0]}): ").strip()
        return val if val in choices else (default or choices[0])
    val = input(f"  {prompt} (default: {default}): ").strip()
    return val if val else default


def interactive_predict(bundle: dict) -> None:
    print("\n" + "=" * 55)
    print("  CHURN PREDICTION — Customer Profile Entry")
    print("=" * 55)

    try:
        features = {
            "gender":           _prompt("Gender", ["Male", "Female"]),
            "SeniorCitizen":    1 if _prompt("Senior Citizen (65+)?", ["No", "Yes"]) == "Yes" else 0,
            "Partner":          _prompt("Has Partner?", ["Yes", "No"]),
            "Dependents":       _prompt("Has Dependents?", ["Yes", "No"]),
            "tenure":           int(_prompt("Tenure (months)", default="12")),
            "PhoneService":     _prompt("Phone Service?", ["Yes", "No"]),
            "MultipleLines":    _prompt("Multiple Lines?", ["Yes", "No", "No phone service"]),
            "InternetService":  _prompt("Internet Service", ["DSL", "Fiber optic", "No"]),
            "OnlineSecurity":   _prompt("Online Security?", ["Yes", "No", "No internet service"]),
            "OnlineBackup":     _prompt("Online Backup?", ["Yes", "No", "No internet service"]),
            "DeviceProtection": _prompt("Device Protection?", ["Yes", "No", "No internet service"]),
            "TechSupport":      _prompt("Tech Support?", ["Yes", "No", "No internet service"]),
            "StreamingTV":      _prompt("Streaming TV?", ["Yes", "No", "No internet service"]),
            "StreamingMovies":  _prompt("Streaming Movies?", ["Yes", "No", "No internet service"]),
            "Contract":         _prompt("Contract Type", ["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": _prompt("Paperless Billing?", ["Yes", "No"]),
            "PaymentMethod":    _prompt("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "MonthlyCharges":   float(_prompt("Monthly Charges ($)", default="65.00")),
        }
        features["TotalCharges"] = str(features["MonthlyCharges"] * features["tenure"])

    except KeyboardInterrupt:
        print("\n[Cancelled]")
        return

    df = pd.DataFrame([features])
    result = predict_churn(df, bundle)

    prob  = result["ChurnProbability"].iloc[0]
    label = result["ChurnPrediction"].iloc[0]

    bar_len = 30
    filled  = int(prob * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "─" * 55)
    print("  PREDICTION RESULT")
    print("─" * 55)
    print(f"  Churn Probability : {prob:.1%}  [{bar}]")
    print(f"  Risk Assessment   : {label}")
    print()

    if label == "High Risk":
        print("  ⚠  RECOMMENDED ACTIONS:")
        if features["Contract"] == "Month-to-month":
            print("   • Offer discounted annual contract upgrade")
        if features["MonthlyCharges"] > 70:
            print("   • Propose loyalty discount or bundle plan")
        if features["tenure"] < 12:
            print("   • Enrol in new-customer onboarding programme")
        if features.get("TechSupport") == "No":
            print("   • Offer 3-month free TechSupport trial")
    else:
        print("  ✅  Customer appears stable. Maintain regular engagement.")
    print("─" * 55)


# BATCH PREDICTION
def batch_predict(input_path: str, output_path: str, bundle: dict) -> None:
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df)} customers from {input_path}")

    result = predict_churn(df, bundle)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved → {output_path}")

    # Summary
    high  = (result["ChurnPrediction"] == "High Risk").sum()
    total = len(result)
    print(f"\n  Customers analysed : {total:,}")
    print(f"  High Risk          : {high:,}  ({high/total:.1%})")
    print(f"  Low Risk           : {total - high:,}  ({(total - high)/total:.1%})")
    print(f"\n  Top 5 highest-risk customers:")
    top5 = result.nlargest(5, "ChurnProbability")
    id_col = "customerID" if "customerID" in result.columns else result.columns[0]
    for _, row in top5.iterrows():
        print(f"   {row[id_col]}  →  {row['ChurnProbability']:.1%}")


# EXAMPLES
def run_examples(bundle: dict) -> None:
    print("\n" + "=" * 55)
    print("  EXAMPLE PREDICTIONS — 3 Customer Profiles")
    print("=" * 55)

    features_df = EXAMPLE_CUSTOMERS.drop(columns=["customerID"])
    result = predict_churn(features_df, bundle)
    result.insert(0, "customerID", EXAMPLE_CUSTOMERS["customerID"].values)

    display_cols = ["customerID", "tenure", "Contract", "MonthlyCharges",
                    "ChurnProbability", "ChurnPrediction"]
    # Only show columns that exist
    display_cols = [c for c in display_cols if c in result.columns]

    print(result[display_cols].to_string(index=False))
    print()


# MAIN
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telecom Churn Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input",    type=str, help="Path to input CSV (batch mode)")
    parser.add_argument("--output",   type=str, default="data/predictions.csv",
                        help="Path for predictions output CSV")
    parser.add_argument("--model",    type=str, default=MODEL_PATH,
                        help="Path to model .pkl bundle")
    parser.add_argument("--examples", action="store_true",
                        help="Run built-in example customer predictions")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at: {args.model}")
        print("        Please run:  python src/train_model.py")
        sys.exit(1)

    bundle = load_model(args.model)

    if args.examples:
        run_examples(bundle)
    elif args.input:
        batch_predict(args.input, args.output, bundle)
    else:
        interactive_predict(bundle)


if __name__ == "__main__":
    main()
