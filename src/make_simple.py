# src/make_sample.py
import pandas as pd

df = pd.read_csv("data/raw/final_dataset.csv", nrows=1000, low_memory=False)
df.to_csv("data/raw/sample_traffic.csv", index=False)

print("Created data/raw/sample_traffic.csv")

