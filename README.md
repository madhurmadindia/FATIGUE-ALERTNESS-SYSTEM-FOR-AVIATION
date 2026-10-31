# FATIGUE ALERTNESS SYSTEM FOR AVIATION (AeroVigil)

An advanced, real-time pilot alertness and postural evaluation platform engineered for flight decks, flight simulators, and aviation safety monitoring. AeroVigil fuses high-frequency facial landmark tracking with upper-body skeletal pose estimation to detect fatigue, micro-sleep episodes, and postural degradation before flight performance is compromised.

---

## ✈️ Key Capabilities

- **Independent Ocular Tracking (EAR)**: Continuously evaluates left and right Eye Aspect Ratios to identify blinks, micro-sleeps, and unilateral ptosis (asymmetric eye droop > 32%).
- **Standardized PERCLOS (P80)**: Rolling 60-second time-windowed calculation measuring the proportion of time eyes remain closed, aligned with FAA and NASA aviation fatigue research standards.
- **Adaptive Yawn Filter (MAR)**: Real-time Mouth Aspect Ratio detection equipped with temporal duration filtering (> 1.8s) to distinguish true fatigue yawns from pilot speech and ATC communications.
- **Biometric Head Pose Telemetry**:
  - **Pitch (Forward Nodding)**: Calibrated vertical facial foreshortening detection (> 7.0% ratio drop with 500ms debounce).
  - **Yaw (Inattention / Distraction)**: 2D ocular-nasal projection tracking sustained head rotation (> 17° for > 2.8s) away from forward instrument scan.
  - **Roll (Lateral Slump)**: Symmetric outer canthi angle tracking ear-to-shoulder listing (> 8.5° roll).
- **Upper-Body Skeletal Posture (MediaPipe Pose)**:
  - Clavicle-to-chin vertical clearance for spinal compression and slouch detection.
  - Signed shoulder listing angle to detect asymmetrical pilot fatigue slump.
  - Automatic seat-sink fallback using vertical nose coordinates for close-up or laptop webcams where shoulders are out of frame.
- **Cockpit Annunciator Panel**: High-contrast, aeronautical HUD cells (`LEVEL`, `NOD`, `SLUMP`, `SLOUCH`, `YAW DRIFT`, `MICRO-SLEEP`) with real-time status indication.
- **Automatic Grounding Interlock**: Configurable safety threshold (alertness score ≤ 25) that flags incapacitation and grounds the simulated flight.
- **Flight Telemetry Logging & JSON Export**: Records timestamped fatigue events, baseline metrics, and alertness graphs for post-flight debriefing and CRM analysis.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Computer Vision & ML**: [@mediapipe/tasks-vision](https://developers.google.com/mediapipe/solutions/vision) (`FaceLandmarker` 478 points & `PoseLandmarker` 33 upper-body joints)
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas, CSS3 Cockpit Dark Mode, [Chart.js](https://www.chartjs.org/)

---

## 📁 Repository Structure

```text
├── main.py              # FastAPI server serving cockpit interface and assets
├── requirements.txt      # Python dependencies (fastapi, uvicorn)
├── static/
│   ├── app.js           # MediaPipe heuristic engine, calibration, HUD telemetry
│   ├── index.html       # Aviation cockpit HUD layout & annunciator panel
│   └── style.css        # Aviation cockpit theme, gauge graphics, and styling
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or later
- A functional webcam (built-in or USB external)

### 2. Installation

Clone this repository:
```bash
git clone https://github.com/madhurmadindia/FATIGUE-ALERTNESS-SYSTEM-FOR-AVIATION.git
cd FATIGUE-ALERTNESS-SYSTEM-FOR-AVIATION
```

Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Cockpit Evaluator

Start the local server:
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 📋 Calibration Procedure

1. Position yourself directly in front of the camera with your head upright and shoulders visible.
2. Click **BEGIN EVALUATION**.
3. Maintain a natural, alert resting posture and look straight ahead during the **10-second calibration period**.
4. The system automatically measures your individual:
   - Left & Right Resting Eye Aperture (EAR)
   - Baseline Mouth Aspect Ratio (MAR)
   - Neutral Head Pitch, Yaw, and Roll
   - Clavicle-to-Chin Spinal Ratio & Shoulder Alignment
5. Once complete, real-time flight monitoring and fatigue tracking begins automatically.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

