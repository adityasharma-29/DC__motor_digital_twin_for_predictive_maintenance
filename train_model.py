import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json

# --- Load data ---
df = pd.read_csv("simulated_dc_motor_data.csv")
X = df[['Voltage (V)', 'Current (A)', 'RPM']]
y = df['Fault']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict on test set ---
y_pred = model.predict(X_test)

# --- Calculate accuracy ---
accuracy = accuracy_score(y_test, y_pred)

# --- Save model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "dc_motor_fault_model.pkl")

# --- Save accuracy to metrics.json ---
with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print(f"✅ Model trained and saved with accuracy: {accuracy:.4f}")

# --- 🔽 NEW: Generate and save classification report ---
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, target_names=["Healthy", "Faulty"])
report_path = "model/classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f"📄 Classification report saved to {report_path}")
