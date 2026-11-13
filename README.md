# MirageIME WEBCHEAT

Real-time multi-person AI exam proctoring and behavioral anomaly detection system built for **Linux (CachyOS / Arch Linux)**.

## Features
- **HUD Overlay (`MirageIME WEBCHEAT`)**: Live room telemetry showing total tracked students, active high-risk alerts, and real-time FPS.
- **Multi-Person Detection & Persistent IDs**: Tracks every individual in the room simultaneously using YOLOv8 + IoU/Centroid tracking (`STUDENT #1`, `STUDENT #2`, etc.).
- **3D Head Pose & Iris Gaze Tracking**: Uses MediaPipe FaceMesh with refined iris landmarks and 3D PnP solver to compute Pitch, Yaw, Roll, and eye-shift vectors.
- **Hand Movement & Concealment Zone Analysis**: Tracks up to 24 hands simultaneously, computing wrist velocity vectors, under-desk/lap concealment, and outward reaching toward neighbors.
- **Tactical Directional Arrows**: Draws color-coded vector arrows pointing toward possible cheating targets (neighboring desks/screens, concealed lap zones, contraband devices, and rapid hand trajectories).
- **Per-Student Cheating Percentage (`0%–100%`)**: Exponentially smoothed risk probability displayed above each student with live progress bars and behavioral trigger flags.

## Quick Start on CachyOS / Arch Linux

```bash
unzip MirageIME_WEBCHEAT.zip
cd MirageIME_WEBCHEAT
chmod +x run_cachyos.sh
./run_cachyos.sh
```

### Optional CLI Flags
```bash
# Use a secondary USB webcam (/dev/video2)
./run_cachyos.sh --source 2

# Use a higher-accuracy YOLOv8 model (yolov8s.pt or yolov8m.pt)
./run_cachyos.sh --model yolov8s.pt

# Analyze a pre-recorded classroom video or RTSP IP camera stream
./run_cachyos.sh --source classroom_exam.mp4
```

### Keyboard Controls
- **`Q` / `ESC`**: Quit application
- **`S`**: Save timestamped evidence screenshot (`mirage_alert_<timestamp>.png`)
