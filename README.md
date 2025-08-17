This project is a real-time predictive maintenance system for a DC motor, built with Python and Streamlit. 
It integrates IIoT simulation, AI models, anomaly detection, and 3D digital twin visualization to monitor motor health, predict failures, and estimate Remaining Useful Life (RUL).

Features:
Real-time Data Simulation – Generates continuous Voltage, Current, and RPM readings with fault injection.
Machine Learning Fault Prediction – Random Forest model trained on simulated motor data to classify motor health.
Unsupervised Anomaly Detection – Isolation Forest model detects abnormal sensor behavior.
RUL Estimation – Predicts motor’s remaining operational life based on degradation patterns.
Interactive Dashboard – Built in Streamlit, showing live plots, faults, and anomaly detection results.
3D Digital Twin – Real-time motor visualization with rotating shaft, fan blades, and live indicators (Plotly).
Secure & Modular – Clean structure for easy expansion, edge deployment, or cloud hosting.

Tech Stack:
Programming: Python
Frontend/Dashboard: Streamlit, Plotly
Machine Learning: Scikit-learn (Random Forest, Isolation Forest)
Visualization: Plotly 3D
Data Handling: Pandas, NumPy
