# UAV Telemetry Analyzer
### BEST::HACKath0n 2026 — "Система аналізу телеметрії та 3D-візуалізації польотів БПЛА"

---

## Overview

A web-based tool for automated parsing, analysis, and 3D visualization of ArduPilot binary flight logs (`.BIN`). Upload any flight log and instantly get a full mission breakdown — trajectory, metrics, sensor data — with no manual setup required.

---

## Tech Stack & Rationale

| Layer | Technology | Why |
|---|---|---|
| **Parsing** | Python + `struct` | Direct binary unpacking of ArduPilot FMT/message protocol — no external MAVLink dependency needed |
| **Data** | `pandas` + `numpy` | Fast vectorized operations for Haversine distance and trapezoid integration |
| **Web UI** | Streamlit | Zero-boilerplate Python web app — file upload, reactive state, and layout in one file |
| **3D Charts** | Plotly | GPU-accelerated WebGL rendering; smooth interactive 3D with native animation frames — no lag |
| **Styling** | Custom CSS in Streamlit | Gives the dashboard a distinctive look beyond Streamlit defaults |

**Why not matplotlib?** It renders on the CPU and lags on large 3D datasets. Plotly renders in the browser via WebGL and supports animated playback natively.

**Why not a standalone HTML file?** Static HTML requires manually embedding data on every new file. Streamlit parses and visualizes any `.BIN` file on the fly.

---

## Project Structure

```
.
├── app.py              # Streamlit dashboard (main entry point)
├── data_parsing.py     # ArduPilot .BIN parser — GPS, IMU, BARO, ATT extraction
├── analytics.py        # Mission metrics: Haversine distance, trapezoid integration
├── data_convert.py     # Utility wrapper for parse_telemetry()
├── README.md
└── requirements.txt
```

---

## Installation & Setup:

### 1. Clone the repository

```bash
git clone [https://github.com/khrystynaa-sh/HACKath0n-2026.git]
cd HACKath0n-2026
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## requirements.txt

```
streamlit>=1.32.0
plotly>=5.20.0
pandas>=2.0.0
numpy>=1.26.0
```

---

## How to Use:

1. Open the app in your browser (`http://localhost:8501`)
2. In the **sidebar**, click **"Browse files"** or drag & drop a `.BIN` ArduPilot log file
3. Results appear instantly — no need to run any other script

### Sidebar controls
| Control | Description |
|---|---|
| File Upload | Accepts any ArduPilot `.BIN` log file |
| 3D Color Mode | Color trajectory by **Speed**, **Time**, or **Altitude** |
| Animated Playback | Toggle moving position marker on the 3D plot |
| Frame Duration | Control animation speed (50–500 ms per frame) |
| Ground Shadow | Show projected shadow below trajectory |
| Vertical Drop Lines | Show altitude reference lines |

---

## Features:

### MVP (Core Requirements):

**Data Parsing**
- Full ArduPilot binary `.BIN` format parser (no external libraries)
- Extracts GPS, IMU, BARO, ATT, VIBE message streams
- Auto-detects sampling frequency for each sensor
- Resolves physical units from UNIT/MULT/FMTU metadata tables
- Outputs structured `pandas` DataFrames

**Analytics Core**
- Total distance via **Haversine formula** (accounts for Earth's curvature)
- Horizontal & vertical speed via **trapezoid integration** of IMU accelerations
- IMU bias removal using first 50 static samples (drone on ground)
- Max acceleration, max altitude gain, total flight duration

**3D Visualization Panel**
- WGS-84 → **ENU** (East-North-Up) coordinate conversion
  - `E = R · cos(lat₀) · Δlon`
  - `N = R · Δlat`
  - `U = alt − alt₀`
- Interactive 3D Plotly chart with orbit, pan, zoom
- Dynamic trajectory coloring by speed / time / altitude
- Animated playback marker with scrubber slider
- Ground shadow projection & vertical drop lines
- Launch / Land markers

### Nice-to-Have:

**Interactive Dashboard (Streamlit)**
- File upload widget — any `.BIN` file, no code changes needed
- 6 dashboard tabs: 3D Trajectory, Speed & Altitude, IMU, Barometer, Attitude, Log Info
- Mission metrics cards (distance, max speed, altitude gain, duration)
- Fully responsive layout

---

## Algorithm Notes:

### Haversine Distance
Used to compute total path length over GPS coordinates. Accounts for Earth's spherical geometry unlike simple Euclidean distance.

```
a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
d = 2R · arctan2(√a, √(1−a))
```

### Trapezoid Integration (IMU → Velocity)
Raw IMU accelerations are integrated into velocity using the trapezoid rule:

```
v[i] = v[i-1] + (a[i] + a[i-1]) / 2 · Δt
```

Static bias (gravity component + sensor offset) is removed by subtracting the mean of the first 50 samples, recorded while the drone is stationary on the ground.

>  **Note on double integration:** Integrating IMU acceleration twice to get position accumulates error over time (drift). This is a known limitation of dead-reckoning from IMU alone. GPS remains the primary source for position data in this tool.

### WGS-84 → ENU Conversion
Global GPS coordinates are converted to a local flat-Earth Cartesian frame centered at the launch point. This linearization is accurate to ~0.1% for distances under 100 km.

### Why not Euler angles for attitude?
Euler angles (roll/pitch/yaw) suffer from **gimbal lock** — a singularity where two rotation axes align and one degree of freedom is lost. ArduPilot internally uses **quaternions** to represent attitude, which avoid this problem entirely. The ATT log messages expose pre-converted Euler angles for readability, but quaternion-based attitude estimation runs under the hood.

---

## Running the Analytics Script Standalone

If you want to run just the metrics without the web UI:

```bash
python analytics.py
```

Make sure `FILE_PATH` inside `analytics.py` points to your `.BIN` file:

```python
FILE_PATH = r"your_flight.BIN"
```

---

## 👥 Team

**BEST::HACKath0n 2026**
📧 hack@best-lviv.org.ua
