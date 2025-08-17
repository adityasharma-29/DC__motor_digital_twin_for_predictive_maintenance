import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load("dc_motor_fault_model.pkl")

# Title
st.title("DC Motor Digital Twin - Predictive Maintenance Dashboard")
st.markdown("Simulated predictive maintenance monitoring with ML and RUL estimation.")

# Load data
data = pd.read_csv("simulated_dc_motor_data.csv")

# Sidebar: Input sliders for simulation
st.sidebar.header("Simulate Motor Input")
sim_voltage = st.sidebar.slider("Voltage (V)", 11.0, 13.0, 12.0, step=0.05)
sim_current = st.sidebar.slider("Current (A)", 1.0, 3.0, 1.5, step=0.05)
sim_rpm = st.sidebar.slider("RPM", 1000, 1600, 1500, step=10)

# Sidebar: Select time range for historical data
time_range = st.slider("Select Time Range (seconds)", 0, len(data)-1, (0, len(data)-1), step=60)
filtered_data = data.iloc[time_range[0]:time_range[1]]

# --- ML Predictions for historical data ---
features = filtered_data[['Voltage (V)', 'Current (A)', 'RPM']]
filtered_data['Predicted Fault'] = model.predict(features)

# --- Predict Fault & RUL for simulated input ---
st.subheader("Simulation: Predict Fault & Remaining Useful Life (RUL)")

sim_input = pd.DataFrame([[sim_voltage, sim_current, sim_rpm]],
                         columns=['Voltage (V)', 'Current (A)', 'RPM'])

sim_fault = model.predict(sim_input)[0]
sim_proba = model.predict_proba(sim_input)[0][1]  # Probability of fault

# Simple RUL estimation: inverse of fault probability squared for more sensitivity
rul_estimate = max(0, int(100 * (1 - sim_proba)**2))

st.write(f"🔌 Input Voltage: **{sim_voltage:.2f} V**, ⚡ Current: **{sim_current:.2f} A**, 🔄 RPM: **{sim_rpm}**")

if sim_fault == 1:
    st.error(f"⚠️ Fault likely (Risk: {sim_proba*100:.1f}%)")
else:
    st.success(f"✅ Operating Normally (Risk: {sim_proba*100:.1f}%)")

st.info(f"🧮 Estimated Remaining Useful Life: **{rul_estimate} %**")

# Progress bar
progress_bar = st.progress(rul_estimate)

# Conditional warning based on RUL %
if rul_estimate < 20:
    st.warning("⚠️ Remaining Useful Life is critically low! Schedule maintenance soon.")
elif rul_estimate < 50:
    st.info("ℹ️ Remaining Useful Life is moderate. Monitor closely.")
else:
    st.success("✅ Remaining Useful Life is healthy.")

# --- Fault Points Explanation and Count ---
st.subheader("Predictive Maintenance Status (Historical Data)")

st.markdown("""
**What are fault points?**

Fault points are moments in time where the machine learning model predicts a high likelihood of failure based on sensor readings such as Voltage, Current, and RPM.  
A high number of predicted fault points may indicate the motor is under stress or approaching failure.
""")

pred_fault_count = filtered_data['Predicted Fault'].sum()
st.write(f"Predicted fault points in selected time range: **{pred_fault_count}**")

if pred_fault_count > 0:
    st.error("⚠️ Potential fault conditions detected. Consider scheduling maintenance.")
else:
    st.success("✅ Motor predicted to be operating normally in the selected range.")

# Display timestamps where faults are predicted
fault_times = filtered_data[filtered_data['Predicted Fault'] == 1]['Time (s)'].values
st.markdown("### 🕒 Fault Occurrence Timeline")

if len(fault_times) > 0:
    # Create a series with 1 where fault occurred, 0 otherwise, indexed by time
    fault_series = pd.Series(0, index=filtered_data['Time (s)'])
    fault_series.loc[fault_times] = 1

    # Plot fault occurrence as bars (spikes) over time
    fig_fault, ax_fault = plt.subplots(figsize=(10, 2))
    ax_fault.bar(fault_series.index, fault_series.values, width=1.0, color='red')
    ax_fault.set_ylabel('Fault')
    ax_fault.set_xlabel('Time (seconds)')
    ax_fault.set_yticks([0, 1])
    ax_fault.set_yticklabels(['No Fault', 'Fault'])
    ax_fault.set_title('Fault Occurrence Over Time')
    ax_fault.grid(True, axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig_fault)
else:
    st.success("✅ No predicted faults in selected time range.")

# --- Plots for historical data ---
st.subheader("Sensor Data Over Time")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax1.plot(filtered_data['Time (s)'], filtered_data['Voltage (V)'], color='blue')
ax1.set_ylabel('Voltage (V)')
ax1.grid(True)

ax2.plot(filtered_data['Time (s)'], filtered_data['Current (A)'], color='orange')
ax2.set_ylabel('Current (A)')
ax2.grid(True)

ax3.plot(filtered_data['Time (s)'], filtered_data['RPM'], color='green')
ax3.set_ylabel('RPM')
ax3.set_xlabel('Time (seconds)')
ax3.grid(True)

st.pyplot(fig)

# Display actual faults recorded in the data
st.subheader("Actual Faults Recorded")
actual_faults = filtered_data['Fault'].sum()
st.write(f"Number of actual faults recorded in selected range: **{actual_faults}**")

# Show raw data option
if st.checkbox("Show raw data with prediction"):
    st.write(filtered_data)