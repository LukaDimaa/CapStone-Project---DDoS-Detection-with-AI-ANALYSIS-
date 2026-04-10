import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1. LOAD DATA (larger sample)
# -------------------------------
df = pd.read_csv("data/raw/final_dataset.csv", nrows=200000, low_memory=False)

# -------------------------------
# 2. CLEAN DATA
# -------------------------------

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Drop useless columns
df = df.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"], errors="ignore")

# Fix bad values
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Convert labels to binary
df["Label"] = df["Label"].apply(lambda x: "normal" if str(x).upper() == "BENIGN" else "attack")

# -------------------------------
# 3. SHUFFLE + BALANCE DATA
# -------------------------------

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Take equal samples of each class
df = df.groupby("Label").head(25000)

print("\nBalanced dataset:")
print(df["Label"].value_counts())
print("Shape:", df.shape)

# -------------------------------
# 4. PREPARE FOR ML
# -------------------------------

X = df.drop("Label", axis=1)
y = df["Label"].map({"normal": 0, "attack": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. TRAIN MODEL
# -------------------------------

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 6. EVALUATE MODEL
# -------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))