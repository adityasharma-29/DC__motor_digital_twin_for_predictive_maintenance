import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
duration_minutes = 60       # 1 hour of data
sampling_rate = 1           # samples per second
total_samples = duration_minutes * 60 * sampling_rate

# Time array
time = np.arange(total_samples)

# Simulate voltage (mostly stable)
voltage = np.random.normal(loc=12.0, scale=0.2, size=total_samples)

# Simulate RPM (slightly varies, drops slightly over time)
rpm = 1500 - 0.2 * time + np.random.normal(0, 20, total_samples)

# Simulate current (increases slowly as motor ages)
current = 1.5 + 0.0005 * time + np.random.normal(0, 0.05, total_samples)

# Improved Fault label logic: create more fault cases based on stress conditions
fault = (
    (current > 2.5) |
    (rpm < 1200) |
    ((current > 2.2) & (rpm < 1300)) |
    ((voltage > 12.2) & (current > 2.0))
).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'Time (s)': time,
    'Voltage (V)': voltage,
    'Current (A)': current,
    'RPM': rpm,
    'Fault': fault
})

# Save to CSV
data.to_csv("simulated_dc_motor_data.csv", index=False)
print("Data saved to 'simulated_dc_motor_data.csv'.")

# Fault label counts to check distribution
print("Fault Label Counts:\n", data['Fault'].value_counts())

# Plot sensor data for visualization
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(time, voltage, label="Voltage")
plt.ylabel("Voltage (V)")
plt.subplot(3,1,2)
plt.plot(time, current, label="Current", color='orange')
plt.ylabel("Current (A)")
plt.subplot(3,1,3)
plt.plot(time, rpm, label="RPM", color='green')
plt.ylabel("RPM")
plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()