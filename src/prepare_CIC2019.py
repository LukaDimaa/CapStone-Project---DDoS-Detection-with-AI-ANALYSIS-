import pandas as pd

INPUT = "data/raw/cicddos2019_dataset.csv"
OUTPUT = "data/raw/cic2019_preprocessed.csv"

df = pd.read_csv(INPUT, low_memory=False)

source_column = "Class" if "Class" in df.columns else "Label"

df["Label"] = df[source_column].apply(
    lambda x: "normal" if str(x).strip().lower() in {"benign", "normal", "0"} else "attack"
)

df.drop(columns=["Class", "Unnamed: 0"], inplace=True, errors="ignore")

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(df["Label"].value_counts())
