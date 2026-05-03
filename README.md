# AR Filter Webcam Prototype

Real-time Python webcam app with a browser-based control panel, MediaPipe face and hand landmark detection, expression and gesture classification, OpenCV-rendered effects, and optional virtual-camera output.

## Features

- Webcam capture with mirrored selfie preview
- MediaPipe `FaceLandmarker` and `HandLandmarker`
- Face expression heuristics for smile, mouth-open, and raised-brow detection
- Hand gesture heuristics for peace sign, thumbs up, and open palm
- Real-time procedural AR-style overlays:
  - hearts on smile
  - flame burst on mouth open
  - brow flash on raised eyebrows
  - sparkles on peace sign
  - burst ring on thumbs up
  - aura ring on open palm
- FastAPI web UI with live MJPEG preview and effect toggles
- Optional `pyvirtualcam` output so processed frames can appear as a virtual webcam in other apps

## Project Layout

```text
app.py
requirements.txt
scripts/download_models.py
models/
src/
  config.py
  engine.py
  state.py
  detectors/
    face_detector.py
    hand_detector.py
  effects/
    renderer.py
  web/static/
    index.html
    styles.css
    app.js
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

3. Download the MediaPipe task models:

```powershell
python scripts\download_models.py
```

4. Start the app:

```powershell
python app.py
```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Virtual Camera Notes

- The web UI can request virtual-camera output at runtime.
- On Windows, `pyvirtualcam` typically needs an installed virtual-camera backend such as OBS Virtual Camera or UnityCapture.
- If virtual camera startup fails, the error is surfaced in the UI without stopping the main webcam preview.

## Controls

- Toggle effects individually from the browser
- Start or stop virtual camera output from the browser
- Use the live preview panel to monitor expression and gesture triggers

