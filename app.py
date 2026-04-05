"""
Run:
    streamlit run app.py
"""

import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_parsing import parse_telemetry


#  PAGE CONFIG
st.set_page_config(
    page_title="UAV Telemetry Analyzer",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CUSTOM CSS  — dark military-tech aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* global */
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #060d14;
    color: #b0d8e8;
}
.main { background: #060d14; }
section[data-testid="stSidebar"] {
    background: #091520;
    border-right: 1px solid #0d3a52;
}

/* headings */
h1 { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
     color: #00e5ff !important; letter-spacing: 4px; text-transform: uppercase; }
h2 { font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
     color: #90caf9 !important; letter-spacing: 2px; text-transform: uppercase;
     border-bottom: 1px solid #0d3a52; padding-bottom: 6px; }
h3 { font-family: 'Rajdhani', sans-serif !important; color: #64b5f6 !important;
     letter-spacing: 1px; }

/* metric cards */
[data-testid="metric-container"] {
    background: #0d1b2a;
    border: 1px solid #0d3a52;
    border-radius: 0;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #607d8b !important;
    font-size: 11px !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #0d3a52 !important;
    background: #091520 !important;
    border-radius: 0 !important;
}

/* selectbox / radio */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] > div {
    background: #0d1b2a !important;
    border: 1px solid #0d3a52 !important;
    color: #b0d8e8 !important;
}

/* divider */
hr { border-color: #0d3a52 !important; }

/* tabs */
[data-baseweb="tab-list"] { background: #091520 !important; border-bottom: 1px solid #0d3a52; }
[data-baseweb="tab"] { color: #607d8b !important; font-family: 'Share Tech Mono', monospace !important;
                        letter-spacing: 2px; font-size: 12px; }
[aria-selected="true"] { color: #00e5ff !important; border-bottom: 2px solid #00e5ff !important; }

/* sidebar labels */
.sidebar-label {
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: #00e5ff; opacity: 0.7; margin-bottom: 6px; display: block;
}

/* tag badge */
.tag {
    display: inline-block; font-size: 10px; padding: 2px 10px;
    border: 1px solid #00e5ff; color: #00e5ff; letter-spacing: 2px;
    text-transform: uppercase; margin-right: 6px;
}
.tag-red { border-color: #ff4d6d; color: #ff4d6d; }
.tag-green { border-color: #39ff14; color: #39ff14; }

/* sampling table */
.samp-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.samp-table td { padding: 5px 8px; border-bottom: 1px solid #0d3a52; }
.samp-table td:last-child { color: #39ff14; text-align: right; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060d14; }
::-webkit-scrollbar-thumb { background: #0d3a52; }
</style>
""", unsafe_allow_html=True)

#  HELPERS


def to_enu(gps_df):
    """WGS-84 → local ENU (metres from launch point)."""
    R = 6_371_000
    lat0 = np.radians(gps_df['Lat'].iloc[0])
    lon0 = np.radians(gps_df['Lng'].iloc[0])
    alt0 = gps_df['Alt'].iloc[0]
    E = R * np.cos(lat0) * np.radians(gps_df['Lng'].values - gps_df['Lng'].iloc[0])
    N = R * np.radians(gps_df['Lat'].values - gps_df['Lat'].iloc[0])
    U = gps_df['Alt'].values - alt0
    return E, N, U


def haversine_distance(gps_df):
    """Total flight distance via Haversine formula."""
    R = 6_371_000
    lat = np.radians(gps_df['Lat'].values)
    lon = np.radians(gps_df['Lng'].values)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat/2)**2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon/2)**2
    return float(np.sum(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))))


def trapezoid_velocity(imu_df):
    """
    Trapezoid integration of IMU accelerations → velocity.
    Bias is removed using the first 50 static samples (drone on ground).
    Returns max horizontal speed, max vertical speed, max acceleration.
    """
    dt = imu_df['time_s'].diff().fillna(0).values

    ax = imu_df['AccX'].values - imu_df['AccX'].iloc[:50].mean()
    ay = imu_df['AccY'].values - imu_df['AccY'].iloc[:50].mean()
    az = imu_df['AccZ'].values - imu_df['AccZ'].iloc[:50].mean()

    # trapezoid: v[i] = v[i-1] + (a[i]+a[i-1])/2 * dt
    vx = np.cumsum((ax[1:] + ax[:-1]) / 2 * dt[1:])
    vy = np.cumsum((ay[1:] + ay[:-1]) / 2 * dt[1:])
    vz = np.cumsum((az[1:] + az[:-1]) / 2 * dt[1:])

    h_spd = np.sqrt(vx**2 + vy**2)
    v_spd = np.abs(vz)
    total_acc = np.sqrt(ax**2 + ay**2 + az**2)

    return float(h_spd.max()), float(v_spd.max()), float(total_acc.max())


PLOTLY_DARK = dict(
    paper_bgcolor='#060d14',
    plot_bgcolor='#0d1b2a',
    font=dict(color='#b0d8e8', family='Share Tech Mono'),
    margin=dict(l=10, r=10, t=40, b=10),
)


#  SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:20px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;
                  color:#00e5ff;letter-spacing:4px">UAV<span style="color:#ff4d6d">::</span>TELEMETRY</div>
      <div style="font-size:10px;letter-spacing:3px;color:#607d8b">FLIGHT ANALYZER v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Upload Flight Log</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["BIN"], label_visibility="collapsed")

    st.divider()
    st.markdown('<span class="sidebar-label">3D Color Mode</span>', unsafe_allow_html=True)
    color_mode = st.radio(
        "", ["Speed", "Time", "Altitude"],
        label_visibility="collapsed",
        horizontal=False,
    )

    st.divider()
    st.markdown('<span class="sidebar-label">Animation</span>', unsafe_allow_html=True)
    show_anim = st.checkbox("Show animated playback", value=True)
    anim_speed = st.slider("Frame duration (ms)", 50, 500, 150, 50,
                           disabled=not show_anim)

    st.divider()
    st.markdown('<span class="sidebar-label">Display Options</span>', unsafe_allow_html=True)
    show_shadow = st.checkbox("Ground shadow", value=True)
    show_drops  = st.checkbox("Vertical drop lines", value=True)

    st.divider()
    st.markdown("""
    <div style="font-size:10px;color:#607d8b;line-height:2">
    <b style="color:#00e5ff">COORDS</b>  WGS-84 → ENU<br>
    <b style="color:#00e5ff">DIST</b>    Haversine formula<br>
    <b style="color:#00e5ff">SPEED</b>   Trapezoid integration<br>
    </div>
    """, unsafe_allow_html=True)

#  MAIN


st.markdown("# 🛸  UAV Telemetry Analyzer")
st.markdown('<span class="tag">ARDUPILOT</span><span class="tag tag-green">WGS-84 → ENU</span><span class="tag tag-red">HAVERSINE</span>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;border:1px dashed #0d3a52;
                    background:#091520;margin-top:40px">
          <div style="font-size:48px;margin-bottom:16px">📡</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:20px;
                      color:#00e5ff;letter-spacing:3px">AWAITING FLIGHT LOG</div>
          <div style="font-size:12px;color:#607d8b;margin-top:8px;letter-spacing:1px">
            Upload an ArduPilot .BIN file using the sidebar
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


#  PARSE

with st.spinner("Parsing flight log…"):
    with tempfile.NamedTemporaryFile(suffix=".BIN", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        data   = parse_telemetry(tmp_path)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()

gps  = data['gps']
imu  = data['imu']
baro = data['baro']
att  = data['att']
srates = data['sampling']

if gps.empty:
    st.error("No GPS data found in this log file.")
    st.stop()

# ENU
E, N, U = to_enu(gps)
t   = gps['time_s'].values
spd = gps['Spd'].values

# Metrics
total_dist = haversine_distance(gps)
max_alt_gain = float(U.max() - U.min())
duration = float(t[-1])
max_gps_spd = float(spd.max())

if not imu.empty:
    max_h_spd, max_v_spd, max_acc = trapezoid_velocity(imu)
else:
    max_h_spd = max_v_spd = max_acc = float('nan')


#  METRICS ROW

st.markdown("---")
st.markdown("## Mission Metrics")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Distance",    f"{total_dist:.1f} m")
c2.metric("Max GPS Speed",     f"{max_gps_spd:.1f} m/s")
c3.metric("Max Horiz. Speed",  f"{max_h_spd:.1f} m/s" if not np.isnan(max_h_spd) else "—")
c4.metric("Max Vert. Speed",   f"{max_v_spd:.1f} m/s"  if not np.isnan(max_v_spd) else "—")
c5.metric("Max Alt. Gain",     f"{max_alt_gain:.0f} m")
c6.metric("Duration",          f"{duration:.1f} s")

st.markdown("---")

#  TABS
tab3d, tab_time, tab_imu, tab_baro, tab_att, tab_info = st.tabs([
    "  3D TRAJECTORY  ",
    "  SPEED & ALT  ",
    "  IMU  ",
    "  BAROMETER  ",
    "  ATTITUDE  ",
    "  LOG INFO  ",
])


#  TAB 1 — 3D trajectory
with tab3d:
    # colour values
    if color_mode == "Speed":
        c_vals  = spd
        c_label = "Speed (m/s)"
        cmap    = "plasma"
    elif color_mode == "Time":
        c_vals  = t
        c_label = "Time (s)"
        cmap    = "viridis"
    else:
        c_vals  = U
        c_label = "Altitude AGL (m)"
        cmap    = "thermal"

    fig3d = go.Figure()

    # ── ground shadow ──
    if show_shadow:
        ground = U.min() - 5
        fig3d.add_trace(go.Scatter3d(
            x=E, y=N, z=np.full_like(U, ground),
            mode='lines',
            line=dict(color='#1e3a4a', width=2),
            opacity=0.4,
            name='Shadow',
            showlegend=False,
            hoverinfo='skip',
        ))

    # ── vertical drops ──
    if show_drops:
        step = max(1, len(E) // 12)
        for i in range(0, len(E), step):
            fig3d.add_trace(go.Scatter3d(
                x=[E[i], E[i]], y=[N[i], N[i]], z=[U[i], ground],
                mode='lines',
                line=dict(color='#1e3a4a', width=1),
                opacity=0.4,
                showlegend=False,
                hoverinfo='skip',
            ))

    # ── main coloured trajectory ──
    fig3d.add_trace(go.Scatter3d(
        x=E, y=N, z=U,
        mode='lines+markers',
        line=dict(color=c_vals, colorscale=cmap, width=4,
                  colorbar=dict(
                                title=dict(text=c_label,
                                           font=dict(color='#b0d8e8', size=11)),
                                thickness=12,
                                tickfont=dict(color='#b0d8e8', size=10),
                                bgcolor='#091520', bordercolor='#0d3a52')),
        marker=dict(size=2, color=c_vals, colorscale=cmap, opacity=0.7),
        name='Trajectory',
        hovertemplate=(
            "E: %{x:.1f} m<br>"
            "N: %{y:.1f} m<br>"
            "U: %{z:.1f} m<br>"
            f"{c_label}: %{{marker.color:.2f}}<extra></extra>"
        ),
    ))

    # ── start / end markers ──
    fig3d.add_trace(go.Scatter3d(
        x=[E[0]], y=[N[0]], z=[U[0]],
        mode='markers+text',
        marker=dict(size=10, color='#39ff14', symbol='circle',
                    line=dict(color='white', width=1)),
        text=['LAUNCH'], textposition='top center',
        textfont=dict(color='#39ff14', size=11),
        name='Launch', showlegend=True,
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[E[-1]], y=[N[-1]], z=[U[-1]],
        mode='markers+text',
        marker=dict(size=10, color='#ff4d6d', symbol='circle',
                    line=dict(color='white', width=1)),
        text=['LAND'], textposition='top center',
        textfont=dict(color='#ff4d6d', size=11),
        name='Land', showlegend=True,
    ))

    # ── ANIMATION frames ──
    if show_anim:
        frames = []
        step_anim = max(1, len(E) // 60)   # ~60 frames max
        for i in range(0, len(E), step_anim):
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=[E[i]], y=[N[i]], z=[U[i]],
                    mode='markers',
                    marker=dict(size=10, color='white',
                                symbol='circle',
                                line=dict(color='#00e5ff', width=2)),
                )],
                name=str(i),
                traces=[len(fig3d.data)],
            ))

        # placeholder trace for the animated marker
        fig3d.add_trace(go.Scatter3d(
            x=[E[0]], y=[N[0]], z=[U[0]],
            mode='markers',
            marker=dict(size=10, color='white',
                        line=dict(color='#00e5ff', width=2)),
            name='Position',
            showlegend=True,
        ))

        fig3d.frames = frames
        fig3d.update_layout(
            updatemenus=[dict(
                type='buttons', showactive=False,
                y=0.02, x=0.5, xanchor='center',
                bgcolor='#0d1b2a', bordercolor='#0d3a52',
                font=dict(color='#00e5ff', family='Share Tech Mono', size=11),
                buttons=[
                    dict(label='▶  PLAY',
                         method='animate',
                         args=[None, dict(frame=dict(duration=anim_speed, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸  PAUSE',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')]),
                ],
            )],
            sliders=[dict(
                active=0, currentvalue=dict(visible=False),
                pad=dict(t=10, b=5),
                bgcolor='#0d1b2a', bordercolor='#0d3a52',
                tickcolor='#0d3a52',
                font=dict(color='#607d8b'),
                steps=[dict(
                    method='animate',
                    args=[[str(i)], dict(mode='immediate',
                                         frame=dict(duration=anim_speed, redraw=True))],
                    label='',
                ) for i in range(0, len(E), max(1, len(E)//60))],
            )],
        )

    fig3d.update_layout(
        **PLOTLY_DARK,
        height=620,
        scene=dict(
            bgcolor='#060d14',
            xaxis=dict(title='E (m)', color='#ff4d6d',
                       gridcolor='#1e3a4a', showbackground=False),
            yaxis=dict(title='N (m)', color='#00e5ff',
                       gridcolor='#1e3a4a', showbackground=False),
            zaxis=dict(title='U (m)', color='#39ff14',
                       gridcolor='#1e3a4a', showbackground=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        legend=dict(bgcolor='#091520', bordercolor='#0d3a52',
                    font=dict(color='#b0d8e8', size=11)),
        title=dict(text=f'3D Flight Trajectory — {uploaded.name}',
                   font=dict(color='white', size=14, family='Rajdhani')),
    )

    st.plotly_chart(fig3d, use_container_width=True)


#  TAB 2 — Speed & Altitude

with tab_time:
    fig_sa = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.55, 0.45],
                           vertical_spacing=0.06)

    # speed
    fig_sa.add_trace(go.Scatter(
        x=t, y=spd, name='GPS Speed',
        line=dict(color='#00e5ff', width=2),
        fill='tozeroy', fillcolor='rgba(0,229,255,0.07)',
        hovertemplate='t=%{x:.1f}s  spd=%{y:.2f} m/s<extra></extra>',
    ), row=1, col=1)

    # altitude AGL
    fig_sa.add_trace(go.Scatter(
        x=t, y=U, name='Altitude AGL',
        line=dict(color='#ff9800', width=2),
        fill='tozeroy', fillcolor='rgba(255,152,0,0.07)',
        hovertemplate='t=%{x:.1f}s  alt=%{y:.1f} m<extra></extra>',
    ), row=2, col=1)

    fig_sa.update_layout(
        **PLOTLY_DARK,
        height=480,
        title=dict(text='Speed & Altitude Profile',
                   font=dict(color='white', size=14, family='Rajdhani')),
        legend=dict(bgcolor='#091520', bordercolor='#0d3a52',
                    font=dict(color='#b0d8e8')),
        xaxis2=dict(title='Time (s)', color='#607d8b',
                    gridcolor='#1e3a4a', zerolinecolor='#1e3a4a'),
        yaxis=dict(title='Speed (m/s)', color='#00e5ff',
                   gridcolor='#1e3a4a', zerolinecolor='#1e3a4a'),
        yaxis2=dict(title='Alt AGL (m)', color='#ff9800',
                    gridcolor='#1e3a4a', zerolinecolor='#1e3a4a'),
    )
    st.plotly_chart(fig_sa, use_container_width=True)


#  TAB 3 — IMU

with tab_imu:
    if imu.empty:
        st.warning("No IMU data in this log.")
    else:
        fig_imu = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Accelerometer (m/s²)', 'Gyroscope (rad/s)'),
                                vertical_spacing=0.1)

        colors_acc = ['#ff4d6d', '#39ff14', '#00e5ff']
        colors_gyr = ['#ff9800', '#ab47bc', '#26c6da']

        for col, color, label in zip(['AccX','AccY','AccZ'], colors_acc, ['X','Y','Z']):
            fig_imu.add_trace(go.Scatter(
                x=imu['time_s'], y=imu[col], name=f'Acc {label}',
                line=dict(color=color, width=1.2),
                hovertemplate=f'Acc{label}=%{{y:.3f}}<extra></extra>',
            ), row=1, col=1)

        for col, color, label in zip(['GyrX','GyrY','GyrZ'], colors_gyr, ['X','Y','Z']):
            fig_imu.add_trace(go.Scatter(
                x=imu['time_s'], y=imu[col], name=f'Gyr {label}',
                line=dict(color=color, width=1.2),
                hovertemplate=f'Gyr{label}=%{{y:.3f}}<extra></extra>',
            ), row=2, col=1)

        fig_imu.update_layout(
            **PLOTLY_DARK,
            height=500,
            title=dict(text='IMU Sensor Data',
                       font=dict(color='white', size=14, family='Rajdhani')),
            legend=dict(bgcolor='#091520', bordercolor='#0d3a52',
                        font=dict(color='#b0d8e8', size=10)),
            xaxis2=dict(title='Time (s)', color='#607d8b', gridcolor='#1e3a4a'),
            yaxis=dict(color='#607d8b', gridcolor='#1e3a4a'),
            yaxis2=dict(color='#607d8b', gridcolor='#1e3a4a'),
        )
        for ann in fig_imu.layout.annotations:
            ann.font.color = '#90caf9'
            ann.font.family = 'Rajdhani'

        st.plotly_chart(fig_imu, use_container_width=True)


#  TAB 4 — Barometer

with tab_baro:
    if baro.empty:
        st.warning("No barometer data in this log.")
    else:
        fig_baro = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=('Altitude (m)', 'Climb Rate (m/s)'),
                                 vertical_spacing=0.1)

        fig_baro.add_trace(go.Scatter(
            x=baro['time_s'], y=baro['Alt'], name='Baro Alt',
            line=dict(color='#ff9800', width=2),
            fill='tozeroy', fillcolor='rgba(255,152,0,0.07)',
        ), row=1, col=1)

        if 'CRt' in baro.columns:
            fig_baro.add_trace(go.Scatter(
                x=baro['time_s'], y=baro['CRt'], name='Climb Rate',
                line=dict(color='#ab47bc', width=1.5),
            ), row=2, col=1)

        fig_baro.update_layout(
            **PLOTLY_DARK,
            height=460,
            title=dict(text='Barometer Data',
                       font=dict(color='white', size=14, family='Rajdhani')),
            legend=dict(bgcolor='#091520', bordercolor='#0d3a52',
                        font=dict(color='#b0d8e8')),
            xaxis2=dict(title='Time (s)', color='#607d8b', gridcolor='#1e3a4a'),
            yaxis=dict(color='#607d8b', gridcolor='#1e3a4a'),
            yaxis2=dict(color='#607d8b', gridcolor='#1e3a4a'),
        )
        for ann in fig_baro.layout.annotations:
            ann.font.color = '#90caf9'
            ann.font.family = 'Rajdhani'

        st.plotly_chart(fig_baro, use_container_width=True)


#  TAB 5 — Attitude

with tab_att:
    if att.empty:
        st.warning("No attitude data in this log.")
    else:
        fig_att = go.Figure()
        angle_cols = [('Roll','#ff4d6d'), ('Pitch','#39ff14'), ('Yaw','#00e5ff')]
        for col, color in angle_cols:
            if col in att.columns:
                fig_att.add_trace(go.Scatter(
                    x=att['time_s'], y=att[col], name=col,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'{col}=%{{y:.2f}}°<extra></extra>',
                ))

        fig_att.update_layout(
            **PLOTLY_DARK,
            height=420,
            title=dict(text='Attitude — Roll / Pitch / Yaw',
                       font=dict(color='white', size=14, family='Rajdhani')),
            xaxis=dict(title='Time (s)', color='#607d8b', gridcolor='#1e3a4a'),
            yaxis=dict(title='Angle (°)', color='#607d8b', gridcolor='#1e3a4a'),
            legend=dict(bgcolor='#091520', bordercolor='#0d3a52',
                        font=dict(color='#b0d8e8')),
        )
        st.plotly_chart(fig_att, use_container_width=True)


#  TAB 6 — Log Info

with tab_info:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Sampling Rates")
        rows = ""
        for sensor, info in srates.items():
            rows += f"<tr><td>{sensor}</td><td>{info['freq_hz']} Hz</td></tr>"
        st.markdown(f"""
        <table class="samp-table">
          <tr><td><b>Sensor</b></td><td><b>Frequency</b></td></tr>
          {rows}
        </table>
        """, unsafe_allow_html=True)

        st.markdown("### Message Types Found")
        types = data['msg_types']
        badges = " ".join(
            f'<span style="display:inline-block;margin:2px;padding:2px 8px;'
            f'border:1px solid #1e3a4a;font-size:10px;color:#607d8b">{m}</span>'
            for m in types
        )
        st.markdown(badges, unsafe_allow_html=True)

    with col_r:
        st.markdown("### GPS Summary")
        st.markdown(f"""
        <table class="samp-table">
          <tr><td>Launch Lat</td><td>{gps['Lat'].iloc[0]:.6f}°</td></tr>
          <tr><td>Launch Lng</td><td>{gps['Lng'].iloc[0]:.6f}°</td></tr>
          <tr><td>Launch Alt (MSL)</td><td>{gps['Alt'].iloc[0]:.1f} m</td></tr>
          <tr><td>Max Satellites</td><td>{int(gps['NSats'].max())}</td></tr>
          <tr><td>Min HDop</td><td>{gps['HDop'].min():.2f}</td></tr>
          <tr><td>GPS Points</td><td>{len(gps)}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        if not imu.empty:
            st.markdown("### IMU Summary")
            st.markdown(f"""
            <table class="samp-table">
              <tr><td>IMU Points</td><td>{len(imu)}</td></tr>
              <tr><td>Max |AccX|</td><td>{imu['AccX'].abs().max():.3f} m/s²</td></tr>
              <tr><td>Max |AccY|</td><td>{imu['AccY'].abs().max():.3f} m/s²</td></tr>
              <tr><td>Max |AccZ|</td><td>{imu['AccZ'].abs().max():.3f} m/s²</td></tr>
              <tr><td>Avg Temp</td><td>{imu['T'].mean():.1f} °C</td></tr>
            </table>
            """, unsafe_allow_html=True)
