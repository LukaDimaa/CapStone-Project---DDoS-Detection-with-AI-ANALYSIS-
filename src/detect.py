import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/ddos_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

## nothing, just the main DDoS detection tool and the start and screen and being the banner for the Toool
def print_banner() -> None:
    print("=" * 60)
    print("DDoS Detection and AI Analysis Tool")
    print("=" * 60)

## Meta and Loading the model file and testing if it will load : NOT FOUND, IS FOUND
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f" Model file not found: {MODEL_PATH}")
        print("Train the model first before running detection.")
        sys.exit(1)

    if not os.path.exists(FEATURES_PATH):
        print(f" Feature column file not found: {FEATURES_PATH}")
        print("Train the model first before running detection.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, feature_columns


def load_csv(file_path: str, row_limit: int | None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f" Input file not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, nrows=row_limit, low_memory=False)
        return df
    except Exception as exc:
        print(f" Failed to read CSV file: {exc}")
        sys.exit(1)


def preprocess_data(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Drop columns that were excluded during training
    df = df.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"], errors="ignore")

    # Replace invalid values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # Add missing columns --
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the exact trained feature set and order
    df = df[feature_columns]

    return df


def classify_traffic(attack_count: int, total_rows: int) -> str:
    if total_rows == 0:
        return "No data"

    attack_ratio = attack_count / total_rows

    if attack_count == 0:
        return "NORMAL"
    if attack_ratio < 0.10:
        return "LOW RISK"
    if attack_ratio < 0.50:
        return "SUSPICIOUS"
    return "ATTACK DETECTED"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-powered DDoS detection tool for CSV network traffic data."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional limit on number of rows to read"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Optional output CSV file path (default: results.csv)"
    )

    args = parser.parse_args()

    print_banner()

    model, feature_columns = load_model()
    print(" Model and feature list loaded")

    raw_df = load_csv(args.file, args.rows)
    print(f" Input file: {args.file}")
    print(f" Rows loaded: {len(raw_df)}")
    print(f" Columns loaded: {len(raw_df.columns)}")

    processed_df = preprocess_data(raw_df, feature_columns)

    try:
        predictions = model.predict(processed_df)
    except Exception as exc:
        print(f" Prediction failed: {exc}")
        sys.exit(1)

    result_df = raw_df.copy()
    result_df["Prediction"] = predictions
    result_df["Prediction_Label"] = result_df["Prediction"].map({0: "normal", 1: "attack"})

    total_rows = len(result_df)
    attack_count = int((result_df["Prediction"] == 1).sum())
    normal_count = int((result_df["Prediction"] == 0).sum())

    attack_percent = (attack_count / total_rows * 100) if total_rows else 0
    normal_percent = (normal_count / total_rows * 100) if total_rows else 0

    verdict = classify_traffic(attack_count, total_rows)

    print("\n" + "-" * 60)
    print("Detection Summary")
    print("-" * 60)
    print(f"Normal rows : {normal_count}")
    print(f"Attack rows : {attack_count}")
    print(f"Normal %    : {normal_percent:.2f}%")
    print(f"Attack %    : {attack_percent:.2f}%")
    print(f"Verdict     : {verdict}")

    if verdict == "NORMAL":
        print("\n Traffic appears normal.")
    elif verdict == "LOW RISK":
        print("\n Low amount of suspicious traffic detected.")
    elif verdict == "SUSPICIOUS":
        print("\n Suspicious traffic pattern detected.")
    elif verdict == "ATTACK DETECTED":
        print("\n Potential DDoS attack detected.")
    else:
        print("\nℹ No usable data found.")

    try:
        result_df.to_csv(args.output, index=False)
        print(f"\n Results saved to: {args.output}")
    except Exception as exc:
        print(f"\n Could not save results file: {exc}")


if __name__ == "__main__":
    main()
