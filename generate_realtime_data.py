import pandas as pd
import numpy as np
import time
import os

# --- Configuration ---
csv_file = "realtime_dc_motor_data.csv"
duration_minutes = 5
sampling_interval = 1  # seconds
total_samples = duration_minutes * 60

# --- Remove previous file ---
if os.path.exists(csv_file):
    os.remove(csv_file)

print("🟢 Generating realistic real-time motor data with early fault chain...")

# --- Generate live data ---
for i in range(total_samples):
    timestamp = i  # in seconds

    # PHASE 1: Healthy (0–60s)
    if i < 60:
        rpm = 1500 + np.random.normal(0, 10)
        current = 1.5 + np.random.normal(0, 0.05)

    # PHASE 2: Degradation begins (60–120s)
    elif 60 <= i < 120:
        rpm = 1450 - 0.5 * (i - 60) + np.random.normal(0, 20)
        current = 1.6 + 0.01 * (i - 60) + np.random.normal(0, 0.05)

    # PHASE 3: Faulty behavior increases (120s+)
    else:
        rpm = 1300 - 0.3 * (i - 120) + np.random.normal(0, 25)
        current = 2.2 + 0.015 * (i - 120) + np.random.normal(0, 0.07)

    voltage = np.random.normal(loc=12.0, scale=0.2)

    # Fault rule: Current > 2.5 A or RPM < 1150
    fault = int((current > 2.5) or (rpm < 1150))

    row = {
        "Time (s)": timestamp,
        "Voltage (V)": voltage,
        "Current (A)": current,
        "RPM": rpm,
        "Fault": fault
    }

    # Save to file
    df = pd.DataFrame([row])
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', index=False, header=False)

    print(f"[{timestamp}s] V: {voltage:.2f}, I: {current:.2f}, RPM: {rpm:.0f}, Fault: {fault}")
    time.sleep(sampling_interval)


print("✅ Real-time data generation finished.")

