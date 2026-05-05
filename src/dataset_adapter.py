import pandas as pd

LABEL_CANDIDATES = ["Label", "Class", "Attack", "Category", "target"]

DROP_COLUMNS = [
    "Flow_ID", "Flow ID", "Src_IP", "Src IP",
    "Dst_IP", "Dst IP", "Timestamp", "Unnamed: 0", "Unnamed: 0.1"
]

NORMAL_LABELS = {
    "benign", "normal", "0", "non-attack", "non_attack", "legitimate"
}

ATTACK_LABELS = {
    "attack", "1", "ddos", "dos", "drdos"
}


def find_label_column(df):
    binary_candidates = []

    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue

        unique_values = {
            str(value).strip().lower()
            for value in df[col].dropna().unique()
        }

        if unique_values and unique_values.issubset(NORMAL_LABELS | ATTACK_LABELS):
            binary_candidates.append(col)

    if binary_candidates:
        if "Class" in binary_candidates:
            return "Class"
        return binary_candidates[0]

    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError("No label column found.")


def normalize_label(value):
    value = str(value).strip().lower()

    if value in NORMAL_LABELS:
        return "normal"
    return "attack"


def adapt_dataset(input_file, output_file):
    print("ADAPTER STARTED")
    df = pd.read_csv(input_file, low_memory=False)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    label_col = find_label_column(df)

    df["Label"] = df[label_col].apply(normalize_label)

    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")

    extra_label_columns = [col for col in LABEL_CANDIDATES if col in df.columns and col != "Label"]
    if extra_label_columns:
        df.drop(columns=extra_label_columns, inplace=True, errors="ignore")

    if df["Label"].nunique() < 2:
        print("WARNING: Dataset contains only one class.")

    df.to_csv(output_file, index=False)


    

    print("Saved adapted dataset:", output_file)
    print(df["Label"].value_counts())

if __name__ == "__main__":
    import argparse

    print("MAIN BLOCK RUNNING")

    parser = argparse.ArgumentParser(
        description="Universal dataset adapter for DDoS datasets"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw input dataset CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to adapted output CSV"
    )

    args = parser.parse_args()

    print("Input:", args.input)
    print("Output:", args.output)

    adapt_dataset(args.input, args.output)
