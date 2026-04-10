import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

FILE_PATH = "data/raw/final_dataset.csv"
TARGET_PER_CLASS = 25000
CHUNK_SIZE = 500000

attack_chunks = []
normal_chunks = []

attack_count = 0
normal_count = 0
chunk_index = 0

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_index += 1
    print(f"\nReading chunk {chunk_index}...")

    chunk.columns = chunk.columns.str.strip().str.replace(" ", "_")

    chunk = chunk.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"], errors="ignore")
    chunk.replace([np.inf, -np.inf], 0, inplace=True)
    chunk.fillna(0, inplace=True)

    chunk["Label"] = chunk["Label"].apply(
        lambda x: "normal" if str(x).strip().upper() == "BENIGN" else "attack"
    )

    chunk_counts = chunk["Label"].value_counts()
    print("Chunk labels:")
    print(chunk_counts)

    if attack_count < TARGET_PER_CLASS:
        attack_rows = chunk[chunk["Label"] == "attack"]
        needed_attack = TARGET_PER_CLASS - attack_count
        if not attack_rows.empty:
            attack_take = attack_rows.head(needed_attack)
            attack_chunks.append(attack_take)
            attack_count += len(attack_take)

    if normal_count < TARGET_PER_CLASS:
        normal_rows = chunk[chunk["Label"] == "normal"]
        needed_normal = TARGET_PER_CLASS - normal_count
        if not normal_rows.empty:
            normal_take = normal_rows.head(needed_normal)
            normal_chunks.append(normal_take)
            normal_count += len(normal_take)

    print(f"Collected so far -> attack: {attack_count}, normal: {normal_count}")

    if attack_count >= TARGET_PER_CLASS and normal_count >= TARGET_PER_CLASS:
        break

if attack_count == 0 or normal_count == 0:
    raise ValueError(
        f"Could not collect both classes. attack={attack_count}, normal={normal_count}"
    )

df = pd.concat(attack_chunks + normal_chunks, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced dataset:")
print(df["Label"].value_counts())
print("Shape:", df.shape)

X = df.drop("Label", axis=1)
y = df["Label"].map({"normal": 0, "attack": 1})

feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ddos_model.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("\nModel saved to models/ddos_model.pkl")
print("Feature columns saved to models/feature_columns.pkl")