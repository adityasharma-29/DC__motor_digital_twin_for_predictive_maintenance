def render_motor_3d_view(df):
    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st

    latest = df.iloc[-1]
    rpm = latest['RPM']
    time_s = latest['Time (s)']
    fault = latest['Fault'] if 'Fault' in latest else 0

    # Shaft calculation
    angle = (rpm * time_s / 60.0) % 360
    angle_rad = np.radians(angle)
    shaft_length = 2.5
    x1 = shaft_length * 0.5 * np.cos(angle_rad)
    y1 = shaft_length * 0.5 * np.sin(angle_rad)
    z1 = 1.5
    shaft_color = 'red' if fault else 'black'
    glow_color = 'rgba(255,0,0,0.3)' if fault else 'rgba(0,255,0,0.1)'

    # --- Extract latest reading ---
    latest = df.iloc[-1]
    rpm = latest['RPM']
    time_s = latest['Time (s)']
    fault = latest['Fault'] if 'Fault' in latest else 0

    # --- Shaft math ---
    angle = (rpm * time_s / 60.0) % 360
    angle_rad = np.radians(angle)
    shaft_length = 2.5
    x1 = shaft_length * 0.5 * np.cos(angle_rad)
    y1 = shaft_length * 0.5 * np.sin(angle_rad)
    z1 = 1.5

    # --- Colors ---
    shaft_color = 'red' if fault else 'black'
    glow_color = 'rgba(255,0,0,0.3)' if fault else 'rgba(0,255,0,0.1)'

    # --- View 1: Digital Twin ---
    fig1 = go.Figure()

    # Transparent glass-like casing
    fig1.add_trace(go.Mesh3d(
        x=[1, -1, -1, 1, 1, -1, -1, 1],
        y=[1, 1, -1, -1, 1, 1, -1, -1],
        z=[0, 0, 0, 0, 3, 3, 3, 3],
        opacity=0.2,
        color='lightblue',
        name="Glass Casing",
        alphahull=0
    ))

    # Main rotating shaft
    fig1.add_trace(go.Scatter3d(
        x=[0, x1],
        y=[0, y1],
        z=[1.5, z1],
        mode='lines+markers',
        line=dict(color=shaft_color, width=8),
        marker=dict(size=5, color='orange'),
        name='Shaft'
    ))

    # Rotation arc (dashed trail showing shaft rotation path)
    arc_angles = np.linspace(0, angle_rad, 50)
    arc_radius = 1.25
    arc_x = arc_radius * np.cos(arc_angles)
    arc_y = arc_radius * np.sin(arc_angles)
    arc_z = np.full_like(arc_x, 1.5)

    fig1.add_trace(go.Scatter3d(
        x=arc_x,
        y=arc_y,
        z=arc_z,
        mode='lines',
        line=dict(color='orange', width=3, dash='dash'),
        name="Shaft Rotation Path"
    ))

    # Layout settings
    fig1.update_layout(
        title=f"Enhanced Digital Twin View (RPM: {rpm:.0f})",
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True),
            zaxis=dict(title='Z', backgroundcolor="rgb(240, 240,240)", gridcolor="white", showbackground=True)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.02, y=0.98)
    )

    # --- View 2: Realistic Motor ---
    fig2 = go.Figure()

    # --- Motor body (outer casing) ---
    fig2.add_trace(go.Mesh3d(
        x=[1.5, -1.5, -1.5, 1.5, 1.5, -1.5, -1.5, 1.5],
        y=[1.2, 1.2, -1.2, -1.2, 1.2, 1.2, -1.2, -1.2],
        z=[0, 0, 0, 0, 3, 3, 3, 3],
        color='gray',
        opacity=0.4,
        name="Motor Body"
    ))

    # --- Rotor (cylinder-like shape) ---
    fig2.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0.3, 2.7],
        mode='lines',
        line=dict(color=shaft_color, width=20),
        name='Rotor Core'
    ))

    # --- Coils (symbolic winding coils around motor) ---
    coil_positions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for (x, y) in coil_positions:
        fig2.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[1.5],
            mode='markers',
            marker=dict(size=20, color='gold'),
            name='Coil'
        ))

    # --- Fan Blades (at back) ---
    blade_radius = 1.2
    for angle_deg in range(0, 360, 90):
        angle_rad = np.radians(angle_deg)
        x_end = blade_radius * np.cos(angle_rad)
        y_end = blade_radius * np.sin(angle_rad)
        fig2.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[3.2, 3.2],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Fan Blade'
        ))

    # --- Glow / Fault Indicator ---
    fig2.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[1.5],
        mode='markers',
        marker=dict(size=50, color=glow_color, opacity=0.4),
        name='Glow'
    ))

    # --- Shaft ---
    fig2.add_trace(go.Scatter3d(
        x=[0, x1],
        y=[0, y1],
        z=[1.5, z1],
        mode='lines+markers',
        line=dict(color=shaft_color, width=8),
        marker=dict(size=4, color='orange'),
        name='Shaft'
    ))

    # --- Labels for major components ---
    fig2.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[3.4],
        mode='text',
        text=["Fan Blades"],
        textposition="top center",
        textfont=dict(color="blue", size=14),
        showlegend=False
    ))

    fig2.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0.2],
        mode='text',
        text=["Rotor"],
        textposition="bottom center",
        textfont=dict(color="black", size=14),
        showlegend=False
    ))

    fig2.add_trace(go.Scatter3d(
        x=[1.2], y=[0], z=[1.5],
        mode='text',
        text=["Coils"],
        textfont=dict(color="goldenrod", size=14),
        showlegend=False
    ))

    fig2.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.5],
        mode='text',
        text=["Glow / Fault"],
        textfont=dict(color="red" if fault else "green", size=14),
        showlegend=False
    ))

    # --- Static Bearings (front and back)
    bearing_radius = 0.4
    bearing_zs = [0.1, 2.9]
    bearing_segments = 30
    angles = np.linspace(0, 2 * np.pi, bearing_segments)

    for z in bearing_zs:
        x_circle = bearing_radius * np.cos(angles)
        y_circle = bearing_radius * np.sin(angles)
        z_circle = [z] * bearing_segments
        fig2.add_trace(go.Scatter3d(
            x=x_circle,
            y=y_circle,
            z=z_circle,
            mode='lines',
            line=dict(color='darkgray', width=3),
            name='Bearing'
        ))

    # --- Live Text Labels (RPM, Time, Fault Status) ---
    info_labels = [
        f"RPM: {rpm:.0f}",
        f"Time: {int(time_s)} s",
        f"Fault: {'Yes' if fault else 'No'}"
    ]

    info_positions_z = [3.6, 3.4, 3.2]
    info_colors = ['blue', 'black', 'red' if fault else 'green']

    for text, z_pos, color in zip(info_labels, info_positions_z, info_colors):
        fig2.add_trace(go.Scatter3d(
            x=[-2],  # offset left of motor
            y=[0],
            z=[z_pos],
            mode='text',
            text=[text],
            textfont=dict(size=14, color=color),
            showlegend=False
        ))

    fig2.update_layout(
        title="🧿 Realistic Motor View",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )


    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.caption("🔁 This motor updates live every second based on real-time sensor data.")

