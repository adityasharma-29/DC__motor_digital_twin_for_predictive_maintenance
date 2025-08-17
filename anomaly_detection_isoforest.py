import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- Paths ---
DATA_PATH = "realtime_dc_motor_data.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "iso_forest_model.pkl")

# --- Load Data ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Please generate the real-time data first.")

data = pd.read_csv(DATA_PATH)

# --- Features for Anomaly Detection ---
features = data[['Voltage (V)', 'Current (A)', 'RPM']]

# --- Train Isolation Forest ---
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(features)

# --- Predict Anomalies (-1 is anomaly, 1 is normal) ---
data['Anomaly'] = model.predict(features)
data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})  # Convert: 1 = normal → 0, -1 = anomaly → 1

# --- Save the model ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Isolation Forest model saved to {MODEL_PATH}")

# --- Save data with anomaly labels to preview results ---
data.to_csv("anomaly_labeled_data.csv", index=False)
print("📄 Data with anomaly labels saved to anomaly_labeled_data.csv")

# --- 🔽 NEW: Classification report (if Fault column exists) ---
from sklearn.metrics import classification_report

if 'Fault' in data.columns:
    report = classification_report(data['Fault'], data['Anomaly'], target_names=["Healthy", "Faulty"])
    report_path = os.path.join(MODEL_DIR, "anomaly_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📊 Anomaly Classification Report saved to {report_path}")
else:
    print("⚠️ No 'Fault' column found. Skipping anomaly classification report.")
