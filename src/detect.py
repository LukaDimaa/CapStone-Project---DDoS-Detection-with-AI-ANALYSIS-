import pandas as pd
import numpy as np
import argparse
import joblib
import os

# ---------------------------
# CLI ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="DDoS Detection Tool")
parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
args = parser.parse_args()

# ---------------------------
# LOAD MODEL + FEATURES
# ---------------------------
MODEL_PATH = "models/ddos_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

if not os.path.exists(MODEL_PATH):
    print("------- Model not found. Train model first.-------")
    exit()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print("------- Model loaded -------")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(args.file)

print(f"📂 Loaded file: {args.file}")
print(f"Shape: {df.shape}")

# ---------------------------
# CLEAN DATA (SAME AS TRAINING)
# ---------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")

df = df.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"], errors="ignore")

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# ---------------------------
# ALIGN FEATURES
# ---------------------------
# Add missing columns
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0

# Remove extra columns
df = df[feature_columns]

# ---------------------------
# PREDICT
# ---------------------------
predictions = model.predict(df)

df["Prediction"] = predictions

# ---------------------------
# RESULTS
# ---------------------------
attack_count = sum(predictions)
normal_count = len(predictions) - attack_count

print("\n📊 Results:")
print(f"Normal: {normal_count}")
print(f"Attack: {attack_count}")

# ---------------------------
# ALERT SYSTEM
# ---------------------------
if attack_count > 0:
    print("\n ------- ALERT: Potential DDoS Attack Detected! -------")
else:
    print("\n ------- Traffic looks normal. -------")

# Optional: save results
output_file = "results.csv"
df.to_csv(output_file, index=False)

print(f"\n💾 Results saved to {output_file}")   
