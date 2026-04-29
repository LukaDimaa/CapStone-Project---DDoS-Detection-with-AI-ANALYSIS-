import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/ddos_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


def print_banner() -> None:
    print("=" * 128)
    print(r"""
#       ___  ___       ____     __    __          __  _                          __  ___   ____                 __         _   
#      / _ \/ _ \___  / __/ ___/ /__ / /____ ____/ /_(_)__  ___    ___ ____  ___/ / / _ | /  _/ ___ ____  ___ _/ /_ _____ (_)__
#     / // / // / _ \_\ \  / _  / -_) __/ -_) __/ __/ / _ \/ _ \  / _ `/ _ \/ _  / / __ |_/ /  / _ `/ _ \/ _ `/ / // (_-</ (_-<
#    /____/____/\___/___/  \_,_/\__/\__/\__/\__/\__/_/\___/_//_/  \_,_/_//_/\_,_/ /_/ |_/___/  \_,_/_//_/\_,_/_/\_, /___/_/___/
#                                                                                                              /___/            """)
    print("=" * 128)


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f" Model file not found: {MODEL_PATH}")
        print("Train the model first before running detection.")
        sys.exit(2)

    if not os.path.exists(FEATURES_PATH):
        print(f" Feature column file not found: {FEATURES_PATH}")
        print("Train the model first before running detection.")
        sys.exit(2)

    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
    except Exception as exc:
        print(f" Failed to load model artifacts: {exc}")
        sys.exit(2)

    return model, feature_columns


def preprocess_data(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df = df.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"], errors="ignore")

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df


def classify_traffic(attack_count: int, total_rows: int) -> str:
    if total_rows == 0:
        return "NO DATA"

    attack_ratio = attack_count / total_rows

    if attack_count == 0:
        return "NORMAL"
    if attack_ratio < 0.10:
        return "LOW RISK"
    if attack_ratio < 0.50:
        return "SUSPICIOUS"
    return "ATTACK DETECTED"


def safe_output_name(input_file: str) -> str:
    base = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results_{base}_{timestamp}.csv"


def process_large_csv(
    file_path: str,
    model,
    feature_columns: list[str],
    chunk_size: int,
    row_limit: int | None,
    output_file: str | None,
    verbose: bool,
) -> tuple[int, int, int, str | None]:
    if not os.path.exists(file_path):
        print(f" Input file not found: {file_path}")
        sys.exit(2)

    total_rows = 0
    total_attack = 0
    total_normal = 0
    processed_chunks = 0
    rows_remaining = row_limit
    wrote_header = False

    final_output = output_file

    if final_output is None:
        final_output = safe_output_name(file_path)

    try:
        chunk_reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=False
        )
    except Exception as exc:
        print(f" Failed to open CSV file: {exc}")
        sys.exit(2)

    for raw_chunk in chunk_reader:
        if rows_remaining is not None:
            if rows_remaining <= 0:
                break
            raw_chunk = raw_chunk.head(rows_remaining)
            rows_remaining -= len(raw_chunk)

        if raw_chunk.empty:
            continue

        processed_chunks += 1

        processed_chunk = preprocess_data(raw_chunk, feature_columns)

        try:
            predictions = model.predict(processed_chunk)
        except Exception as exc:
            print(f" Prediction failed on chunk {processed_chunks}: {exc}")
            sys.exit(2)

        result_chunk = raw_chunk.copy()
        result_chunk["Prediction"] = predictions
        result_chunk["Prediction_Label"] = result_chunk["Prediction"].map(
            {0: "normal", 1: "attack"}
        )

        chunk_attack = int((result_chunk["Prediction"] == 1).sum())
        chunk_normal = int((result_chunk["Prediction"] == 0).sum())

        total_rows += len(result_chunk)
        total_attack += chunk_attack
        total_normal += chunk_normal

        if verbose:
            print(
                f"[Chunk {processed_chunks}] Rows: {len(result_chunk)} | "
                f"Normal: {chunk_normal} | Attack: {chunk_attack}"
            )

        try:
            result_chunk.to_csv(
                final_output,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False
            )
            wrote_header = True
        except Exception as exc:
            print(f"⚠️ Could not write results for chunk {processed_chunks}: {exc}")
            final_output = None

    return total_rows, total_normal, total_attack, final_output


def print_summary(
    input_file: str,
    total_rows: int,
    total_normal: int,
    total_attack: int,
    output_file: str | None,
) -> str:
    normal_percent = (total_normal / total_rows * 100) if total_rows else 0
    attack_percent = (total_attack / total_rows * 100) if total_rows else 0
    verdict = classify_traffic(total_attack, total_rows)

    print("\n" + "-" * 70)
    print("Detection Summary")
    print("-" * 70)
    print(f"Input file   : {input_file}")
    print(f"Total rows   : {total_rows}")
    print(f"Normal rows  : {total_normal}")
    print(f"Attack rows  : {total_attack}")
    print(f"Normal %     : {normal_percent:.2f}%")
    print(f"Attack %     : {attack_percent:.2f}%")
    print(f"Verdict      : {verdict}")

    if verdict == "NORMAL":
        print("\nTraffic appears normal.")
    elif verdict == "LOW RISK":
        print("\n Low amount of suspicious traffic detected.")
    elif verdict == "SUSPICIOUS":
        print("\n Suspicious traffic pattern detected.")
    elif verdict == "ATTACK DETECTED":
        print("\n Potential DDoS attack detected.")
    else:
        print("\nℹ No usable data found.")

    if output_file:
        print(f"\n Results saved to: {output_file}")

    return verdict

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
        help="Optional limit on number of rows to process"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of rows to process per chunk (default: 10000)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV file path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-chunk processing output"
    )

    args = parser.parse_args()

    print_banner()

    model, feature_columns = load_model()

    print(" Model and feature list loaded")
    print(f" Input file: {args.file}")
    print(f" Chunk size: {args.chunk_size}")

    if args.rows is not None:
        print(f" Row limit : {args.rows}")

    total_rows, total_normal, total_attack, output_file = process_large_csv(
        file_path=args.file,
        model=model,
        feature_columns=feature_columns,
        chunk_size=args.chunk_size,
        row_limit=args.rows,
        output_file=args.output,
        verbose=args.verbose,
    )

    verdict = print_summary(
        input_file=args.file,
        total_rows=total_rows,
        total_normal=total_normal,
        total_attack=total_attack,
        output_file=output_file,
    )

    if verdict == "NORMAL":
        sys.exit(0)
    elif verdict in {"LOW RISK", "SUSPICIOUS", "ATTACK DETECTED"}:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()


    