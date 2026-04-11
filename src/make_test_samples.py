import pandas as pd

FILE_PATH = "data/raw/final_dataset.csv"
CHUNK_SIZE = 500000

attack_sample = None
normal_sample = None

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    attack_rows = chunk[chunk["Label"].astype(str).str.strip().str.lower() != "benign"]
    normal_rows = chunk[chunk["Label"].astype(str).str.strip().str.lower() == "benign"]

    if attack_sample is None and len(attack_rows) >= 1000:
        attack_sample = attack_rows.head(1000)

    if normal_sample is None and len(normal_rows) >= 1000:
        normal_sample = normal_rows.head(1000)

    if attack_sample is not None and normal_sample is not None:
        break

if attack_sample is not None:
    attack_sample.to_csv("data/raw/sample_attack.csv", index=False)
    print("Created data/raw/sample_attack.csv")

if normal_sample is not None:
    normal_sample.to_csv("data/raw/sample_normal.csv", index=False)
    print("Created data/raw/sample_normal.csv")

if attack_sample is not None and normal_sample is not None:
    mixed_sample = pd.concat([attack_sample.head(500), normal_sample.head(500)], ignore_index=True)
    mixed_sample.to_csv("data/raw/sample_mixed.csv", index=False)
    print("Created data/raw/sample_mixed.csv")

    