import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/traffic.csv")

# Encode categorical column
le_protocol = LabelEncoder()
df["protocol_type"] = le_protocol.fit_transform(df["protocol_type"])

# Encode label column
le_label = LabelEncoder()
df["label"] = le_label.fit_transform(df["label"])  # normal=0, attack=1 (example)

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ddos_model.pkl")
joblib.dump(le_protocol, "models/protocol_encoder.pkl")
joblib.dump(le_label, "models/label_encoder.pkl")

print("\nModel saved successfully.")