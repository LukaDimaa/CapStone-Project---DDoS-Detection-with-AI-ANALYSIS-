import pandas as pd

df = pd.read_csv("data/raw/final_dataset.csv", nrows=10000)

print(df.head())
print(df.columns.tolist())
print(df.shape)


if "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("No Label column found")
