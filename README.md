# Smart Supervise
## AI-Powered Multi-Camera CCTV Target Search & Journey Path Engine

A high-performance real-time CCTV intelligence system that searches for and tracks any target person across up to 4 concurrent cameras or video files with time-synchronized playback, automated 360° pose extraction, and journey path mapping.

---

## Key Features

- **Multi-Camera Parallel Engine**: Process up to 4 CCTV feeds simultaneously using 4-core multi-threading with time-synchronized clocks.
- **Auto 360° Video Pose Extractor**: Upload a 2–10 second clip of the target; the AI automatically extracts front, side profile, and back walking angles to maximize detection accuracy from any viewpoint.
- **Multi-Photo Target Gallery**: Add multiple reference photos (front, side, alternate angles) with individual match scores.
- **Ultra-Low False Positives**:
  - *Strict Human Body Geometry Filter*: Instantly rejects floating hands, horizontal arms, table edges, and background artifacts.
  - *Dual-Zone 3D LAB Histograms*: Independently analyzes upper-body shirt and lower-body pants colors; penalizes mismatches heavily.
  - *Calibrated Deep Feature Cosine Similarity*: MobileNetV3 deep embeddings map true matches to high confidence and reject random dark clothing.
- **Target Match Sensitivity Control**: One-click UI toggle between `Strict (70%)` [default], `Balanced (60%)`, and `Flexible (50%)`.
- **High-Speed Video Search (1x to 40x)**: Jump-step sampling searches 40-minute videos in under 2 minutes with live measured speed and ETA telemetry.
- **Interactive Floorplan / Journey Map**: Visual CCTV camera layout with animated pulse alerts, real-time target path trails, and instant camera focus.
- **Animal & Intrusion Alert Mode**: Monitors restricted zones for unauthorized animals (dogs, cats) and persons.
- **Responsive Telemetry Header**: Dedicated non-clipping speed and status telemetry strip designed for laptop and desktop displays.
- **Export Reports**: Generate timestamped CSV and JSON journey logs with captured alert snapshots.

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
python app.py
```

---

## How to Use

1. **Choose Mode**: Select "Search Person by Photo" or "Animal / Intruder Alert".
2. **Setup Target Person**:
   - Click **Upload Target Photo** for a single photo.
   - Click **+ Add Angle** to add side or back photos.
   - Or click **🎥 Auto 360° from Video Clip** to automatically extract all angles from a short video clip!
3. **Set Match Sensitivity**: Choose `Strict (70%)` (recommended), `Balanced (60%)`, or `Flexible (50%)`.
4. **Configure Cameras**: Select your video files or enter RTSP/webcam inputs for Cameras 1 to 4.
5. **Select Playback Speed**: Choose `1x`, `5x`, `10x`, `20x`, or `40x` for high-speed synchronized search.
6. **Click START SCANNING**: Watch real-time feeds or switch to **Map View** to see the animated journey path.
7. **Export**: Click "Export Timeline" to save full CSV + JSON journey records.

---

## Project Structure

```
SmartSupervise/
├── app.py                     # Main desktop application orchestrator
├── requirements.txt
├── core/
│   ├── cctv_scanner.py        # YOLOv8 detection & body geometry filter
│   ├── reid_matcher.py        # MobileNetV3 + 3D LAB color re-identification
│   ├── video_engine.py        # Parallel 4-camera video streaming & speed engine
│   ├── location_tracker.py   # Sighting timeline & journey tracking
│   └── map_visualizer.py      # Vector floorplan & camera path rendering
├── ui/
│   ├── theme.py               # Dark mode color palette & fonts
│   ├── left_panel.py          # Target setup, 360° extractor, camera controls
│   ├── center_grid.py         # 4-camera video matrix & floorplan switch
│   └── right_panel.py         # Journey timeline, last seen card, log export
├── alerts/                    # Auto-saved snapshots of confirmed sightings
└── reports/                   # Exported CSV and JSON timeline reports
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Feature Extraction | MobileNetV3 (PyTorch) + 3D LAB Histograms |
| Video Decoding | OpenCV 4.x (Optimized Thread Pool) |
| Desktop GUI | CustomTkinter (Modular Dark Theme) |
| Concurrency | Python `concurrent.futures` Multi-threading |
