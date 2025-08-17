import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
import os
from motor_3d_view import render_motor_3d_view

st.set_page_config(layout="wide")

import streamlit_authenticator as stauth


credentials = {
    "usernames": {
        "aditya": {
            "name": "Technip Intern-Aditya",
            "password": stauth.Hasher(['1234@TEN']).generate()[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "dc_motor_dashboard",  # cookie name
    "abcdef123",           # cookie key
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("❌ Incorrect username or password")
    st.stop()
elif authentication_status is None:
    st.warning("⚠️ Please enter your credentials")
    st.stop()
else:
    st.success(f"✅ Welcome, {name}!")




# --- Load model ---
model_path = "dc_motor_fault_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found.")
    st.stop()
model = joblib.load(model_path)

# --- Load anomaly detection model ---
anomaly_model_path = "iso_forest_model.pkl"
if not os.path.exists(anomaly_model_path):
    st.warning("Anomaly model not found. Skipping anomaly detection.")
    anomaly_model = None
else:
    anomaly_model = joblib.load(anomaly_model_path)

# --- Load real-time data ---
csv_path = "realtime_dc_motor_data.csv"
if not os.path.exists(csv_path):
    st.warning("⏳ Waiting for real-time data file to be generated...")
    st.stop()

MAX_SAMPLES = 300 # or whatever your data generation limit is
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True


data = pd.read_csv(csv_path)


if len(data) >= MAX_SAMPLES:
    st.session_state.auto_refresh = False
    st.sidebar.warning("🛑 Data generation completed. Auto-refresh stopped. "
                       "Please find the classification reports at the bottom")

# Load accuracy metric dynamically
import json

accuracy_display = None
metrics_path = "model/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        accuracy_display = metrics.get("accuracy")

# --- Run anomaly detection if model is available ---
if anomaly_model:
    anomaly_features = data[['Voltage (V)', 'Current (A)', 'RPM']]
    data['Anomaly'] = anomaly_model.predict(anomaly_features)
    data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly -> convert to 1 = anomaly
else:
    data['Anomaly'] = 0

AUTO_REFRESH_INTERVAL = 5 #seconds

# --- Self-healing logic ---
def suggest_action(row):
    suggestions = []

    if row["Current (A)"] > 2.5:
        suggestions.append("⚠️ High current: Reduce motor load or check for blockage.")
    if row["RPM"] < 1150:
        suggestions.append("⚠️ Low RPM: Inspect motor for wear or shaft issues.")
    if row["Voltage (V)"] > 12.2 and row["Current (A)"] > 2.0:
        suggestions.append("⚠️ Voltage-Current stress: Evaluate power supply or motor resistance.")
    if row["Anomaly"] == 1:
        suggestions.append("🚨 Anomaly detected: Review recent operational changes.")

    if not suggestions:
        return "✅ System appears stable."
    return " | ".join(suggestions)


# --- Auto-refresh ---
st.sidebar.markdown("### 🔄 Auto-Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
st.session_state.auto_refresh = auto_refresh  # Keep it in sync
refresh_interval = 20  # or 5 or 10 as you prefer

# --- Title ---
st.title("🧠 DC Motor Digital Twin Dashboard")
st.markdown("**Real-Time Predictive Maintenance with Fault Detection and RUL Estimation**")



# --- Show elapsed time ---
elapsed_time = int(data['Time (s)'].iloc[-1])
st.markdown(f"⏱️ **Elapsed Time:** `{elapsed_time}` seconds")


# --- Non-blocking auto-refresh spinner ---
if auto_refresh:
    refresh_placeholder = st.empty()
    countdown_seconds = refresh_interval

    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()

    elapsed_since_refresh = time.time() - st.session_state.last_refresh_time

    if elapsed_since_refresh >= refresh_interval:
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    else:
        remaining = int(refresh_interval - elapsed_since_refresh)
        with refresh_placeholder.container():
            if auto_refresh:
                st.info("🔄 Auto-refresh is enabled")
            else:
                st.warning("🔴 Auto-refresh is disabled")

    with refresh_placeholder.container():
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <div style="height: 12px; width: 12px; background-color: red; border-radius: 50%; 
                            animation: blink 1s infinite;"></div>
                <span style="margin-left: 10px; color: red; font-weight: bold;">Live update</span>
            </div>
            <style>
            @keyframes blink {
                0%   { opacity: 1; }
                50%  { opacity: 0; }
                100% { opacity: 1; }
            }
            </style>
        """, unsafe_allow_html=True)

# --- Predict on data ---
features = data[['Voltage (V)', 'Current (A)', 'RPM']]
data['Predicted Fault'] = model.predict(features)
if 'Fault' in data.columns:
    from sklearn.metrics import accuracy_score
    true_labels = data['Fault']
    predicted_labels = data['Predicted Fault']
    dynamic_accuracy = accuracy_score(true_labels, predicted_labels)
else:
    dynamic_accuracy = None

# --- Estimate RUL based on recent window ---
recent_window = data.tail(60)
# --- Anomaly-based RUL Estimation ---
recent_anomalies = recent_window['Anomaly'].sum()
anomaly_risk = recent_anomalies / len(recent_window)  # Proportion of anomalies
anomaly_rul_estimate = max(0, int(100 * (1 - anomaly_risk)**2))  # Same formula style

fault_probs = model.predict_proba(recent_window[['Voltage (V)', 'Current (A)', 'RPM']])[:, 1]
fault_risk = fault_probs.mean()
rul_estimate = max(0, int(100 * (1 - fault_risk)**2))

# --- Sidebar Input Sliders for Simulation ---
st.sidebar.header("Manual Input Simulation")
sim_voltage = st.sidebar.slider("Voltage (V)", 10.0, 14.0, 12.0, step=0.1)
sim_current = st.sidebar.slider("Current (A)", 0.0, 5.0, 2.0, step=0.1)
sim_rpm = st.sidebar.slider("RPM", 1000, 1600, 1400, step=50)

sim_input = pd.DataFrame([[sim_voltage, sim_current, sim_rpm]], columns=['Voltage (V)', 'Current (A)', 'RPM'])
sim_fault = model.predict(sim_input)[0]
sim_proba = model.predict_proba(sim_input)[0][1]
sim_rul = max(0, int(100 * (1 - sim_proba)**2))

st.sidebar.markdown("### Manual Simulation Result:")
if sim_fault == 1:
    st.sidebar.error(f"⚠️ Fault Risk: {sim_proba*100:.1f}%")
else:
    st.sidebar.success(f"✅ Operating Normally ({sim_proba*100:.1f}% risk)")
st.sidebar.progress(sim_rul)

# --- Display RUL ---
st.subheader("📉 RUL Estimation & Fault Overview")
col1, col2 = st.columns(2)

with col1:
    st.metric("RUL (Fault Model)", f"{rul_estimate} %")
    st.progress(rul_estimate)

    st.metric("RUL (Anomaly Model)", f"{anomaly_rul_estimate} %")
    st.progress(anomaly_rul_estimate)

with col2:
    fault_count = data['Predicted Fault'].sum()
    st.metric("Detected Fault Points", f"{fault_count}")
    if fault_count > 0:
        st.error("⚠️ Faults detected – consider scheduling maintenance.")
    else:
        st.success("✅ No predicted faults in the current data.")

    anomaly_count = data['Anomaly'].sum()
    st.metric("Anomaly Count", f"{anomaly_count}")
    if anomaly_count > 0:
        st.warning("🚨 Anomalies detected in current data.")

# --- Generate maintenance suggestions for each row ---
data['Suggestion'] = data.apply(suggest_action, axis=1)


# --- Simulate Automated Corrective Action ---
def simulate_corrective_action(row):
    actions = []

    if row['Predicted Fault'] == 1:
        if row['Current (A)'] > 2.5:
            actions.append("🔧 Reduce current draw (lower torque or load).")
        if row['RPM'] < 1150:
            actions.append("⚙️ Increase RPM (boost motor voltage or recalibrate load).")

    if row['Anomaly'] == 1 and row['Predicted Fault'] == 0:
        actions.append("🛠️ Anomaly detected — inspect for irregular sensor drift or noise.")

    return actions if actions else ["✅ No corrective action needed."]


# Apply on latest row
latest_row = data.iloc[-1]
corrective_actions = simulate_corrective_action(latest_row)

st.subheader("🤖 Automated Corrective Action")

for action in corrective_actions:
    if "✅" in action:
        st.success(action)
    elif "🛠️" in action:
        st.warning(action)
    else:
        st.error(action)

# --- Fault Timeline Bar ---
st.markdown("### 🕒 Fault Timeline Bar")
fig_faults, ax_faults = plt.subplots(figsize=(6, 0.6))
colors = ['red' if f == 1 else 'green' for f in data['Predicted Fault']]
ax_faults.bar(data['Time (s)'], height=1, width=1, color=colors)
ax_faults.set_yticks([])
ax_faults.set_xlabel("Time (s)")
st.pyplot(fig_faults)
st.markdown("### 🚨 Anomaly Timeline Bar")
fig_anom, ax_anom = plt.subplots(figsize=(6, 0.6))
colors_anom = ['red' if a == 1 else 'green' for a in data['Anomaly']]
ax_anom.bar(data['Time (s)'], height=1, width=1, color=colors_anom)
ax_anom.set_yticks([])
ax_anom.set_xlabel("Time (s)")
st.pyplot(fig_anom)

# --- Time-Series Plots ---
st.markdown("### 📊 Live Motor Sensor Trends")
sensor_cols = ['Voltage (V)', 'Current (A)', 'RPM']
colors = ['blue', 'orange', 'green']

for col, color in zip(sensor_cols, colors):
    st.subheader(f"{col}")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(data['Time (s)'], data[col], color=color, label=col)
    ax.scatter(data[data['Predicted Fault'] == 1]['Time (s)'],
               data[data['Predicted Fault'] == 1][col],
               color='red', label='Fault Detected', s=20)
    ax.legend()
    st.pyplot(fig)

# --- 3D Digital Twin Motor Visualization ---
st.subheader("🔩 3D Digital Twin Motor View")
render_motor_3d_view(data)

# --- Show fault timestamps ---
if fault_count > 0:
    st.markdown("### 🕓 Fault Timestamps")
    fault_times = data[data['Predicted Fault'] == 1]['Time (s)']
    st.write(fault_times.reset_index(drop=True), height=100)
else:
    st.markdown("### 🕓 Fault Timestamps")
    st.info("No faults detected in the current data.")
if anomaly_count > 0:
    st.markdown("### 🕓 Anomaly Timestamps")
    anomaly_times = data[data['Anomaly'] == 1]['Time (s)']
    st.write(anomaly_times.reset_index(drop=True), height=100)

st.markdown("### 🛠️ Maintenance Suggestions Summary")
recent_suggestions = data[data["Suggestion"] != "✅ System appears stable."]["Suggestion"].unique()

if len(recent_suggestions) > 0:
    for suggestion in recent_suggestions:
        st.warning(suggestion)
else:
    st.success("✅ No maintenance action required at this time.")



# --- Optional: Show raw data ---
with st.expander("🔍 Show raw data"):
    st.write(data.tail(100))

st.caption("🔁 This dashboard updates live from `realtime_dc_motor_data.csv`. Refresh to see new data.")

if dynamic_accuracy is not None:
    st.info(f"🎯 Model Accuracy on Current Data: `{dynamic_accuracy * 100:.2f}%`")
else:
    st.warning("⚠️ Ground truth not available to compute accuracy.")

authenticator.logout('Logout', 'sidebar')

if st.session_state.auto_refresh:
    st.rerun()


#classification report section

def classification_report_to_html(report_text, title, icon):
    lines = report_text.strip().split("\n")
    headers = lines[0].split()
    html = f"<h4>{icon} {title}</h4>"
    html += "<table style='border-collapse: collapse; width: 100%;'>"
    html += "<tr>" + "".join(f"<th style='padding: 6px; text-align: left;'>{h}</th>" for h in ["Label"] + headers) + "</tr>"

    for line in lines[1:]:
        if line.strip() == "":
            continue
        parts = line.split()
        label = parts[0]
        rest = parts[1:]
        html += "<tr>" + f"<td style='padding: 6px;'>{label}</td>" + "".join(
            f"<td style='padding: 6px;'>{val}</td>" for val in rest
        ) + "</tr>"
    html += "</table><br>"
    return html


st.subheader("📋 Classification Reports")

rf_report_html, iso_report_html = "", ""

rf_report_path = "model/classification_report.txt"
if os.path.exists(rf_report_path):
    with open(rf_report_path, "r") as f:
        rf_text = f.read()
    rf_report_html = classification_report_to_html(rf_text, "Random Forest Fault Classification Report", "🎯")

iso_report_path = "model/anomaly_classification_report.txt"
if os.path.exists(iso_report_path):
    with open(iso_report_path, "r") as f:
        iso_text = f.read()
    iso_report_html = classification_report_to_html(iso_text, "Isolation Forest Anomaly Report", "🧠")

# Side-by-side view using columns
col1, col2 = st.columns(2)
with col1:
    if rf_report_html:
        st.markdown(rf_report_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Random Forest classification report not found.")
with col2:
    if iso_report_html:
        st.markdown(iso_report_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Isolation Forest classification report not found.")
