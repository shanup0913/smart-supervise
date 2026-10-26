# Smart Supervise
## AI-Powered CCTV Target Search & Tracking System

A real-time CCTV intelligence system that lets you search for any person by photo across live cameras or recorded video footage.

---

## Features
- **Search by Photo**: Upload a reference photo → system finds that person in CCTV feeds instantly
- **Live IP Camera / Webcam**: Connect via RTSP stream or local webcam
- **Recorded Video Search**: Scan `.mp4` / `.avi` files to find when & where a person appeared
- **Animal / Intruder Detection**: Detect stray dogs, cats, or unauthorized persons entering restricted zones
- **Movement Timeline**: Full camera-by-camera trail of where the target was spotted
- **Last Known Location**: Always know the most recent confirmed location with timestamp
- **Export Reports**: Download the movement log as CSV or JSON

---

## Installation

```bash
pip install -r requirements.txt
```

> ⚠️ **Important**: OpenCV must be version `4.x`, NOT `5.x`.
> ```bash
> pip install "opencv-python>=4.7.0,<5.0.0"
> ```

---

## Run the App

```bash
cd SmartSupervise
python app.py
```

---

## How to Use

1. **Choose Mode**: Select "Search Person by Photo" or "Animal / Intruder Alert".
2. **Upload Photo** (Person mode): Click "Upload Target Photo" and choose any picture of the person you want to find.
3. **Set Video Source**: Enter `0` for your webcam, an RTSP URL for IP camera, or browse a video file.
4. **Click START SCANNING**: The AI will analyze every frame in real time.
5. **Watch the results**:
   - 🟢 Green box = Target Found with match percentage
   - 🔵 Blue box = Other person detected (not target)
   - 🔴 Red box = Intruder/Animal alert
6. **Movement Trail** (right panel): See every location the target has been spotted.
7. **Export**: Click "Export Timeline" to save a CSV + JSON report.

---

## Project Structure

```
SmartSupervise/
├── app.py                    Main desktop GUI application
├── run_live_cctv.py          Lightweight terminal-mode scanner
├── requirements.txt
├── core/
│   ├── reid_matcher.py       AI photo matching engine (MobileNetV3 + LAB color)
│   ├── cctv_scanner.py       YOLOv8 real-time person/animal detector
│   └── location_tracker.py  Movement timeline & last-known location logger
├── models/
│   └── yolov8n.pt            Pre-downloaded YOLOv8 nano model
├── alerts/                   Auto-saved snapshot images of each sighting
└── reports/                  Exported CSV and JSON timeline reports
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8n (Ultralytics) |
| Person Re-ID | MobileNetV3 (PyTorch) + LAB Color Histograms |
| Video Processing | OpenCV 4.x |
| Desktop GUI | CustomTkinter (dark mode) |
| Deep Learning | PyTorch 2.x (CPU) |
