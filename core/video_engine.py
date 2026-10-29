"""
VideoEngine: Manages multi-camera video streams, time synchronization,
individual camera playback rates, motion gating, and background AI scanning.
"""

import os
import cv2
import time
import threading
import numpy as np


class VideoEngine:
    """
    Decoupled Multi-Camera Video Playback & Synchronized AI Scanning Engine.
    
    Responsibilities:
    1. Stream Acquisition: Opens and decodes up to 4 concurrent RTSP/Webcam/File sources.
    2. Global Time Synchronization: Advances video feeds along a unified timeline.
    3. Per-Camera Speed Control: Supports independent speed multipliers (1x to 60x).
    4. Smart Motion Gating (0.18ms): Skips static frames during high-speed scans.
    5. Thread Separation: Runs all decoding & neural inference in a worker thread,
       keeping the GUI completely lag-free.
    """
    def __init__(self, scanner, tracker):
        self.scanner = scanner
        self.tracker = tracker

        self.is_running = False
        self.worker_thread = None
        self._lock = threading.Lock()

        self.global_clock_sec = 0.0
        self.max_video_duration = 0.0
        self.current_fps = 0.0
        self.last_spotlight_idx = 0

        # Speed calibration table: (step_frames, sleep_time)
        self.speed_map = {
            "1x":  (1, 0.030),   # 1 frame jump (~30 FPS playback)
            "4x":  (4, 0.015),   # 4 frame jump (~4x speed)
            "10x": (10, 0.005),  # 10 frame jump (~10x speed)
            "30x": (30, 0.002),  # 1 second jump per step (~30x TRUE fast search!)
            "60x": (60, 0.001)   # 2 second jump per step (~60x TRUE fast search!)
        }

        # 4 Camera State Slots
        default_locations = [
            "Camera 1 (Main Gate)",
            "Camera 2 (Corridor)",
            "Camera 3 (Room 204)",
            "Camera 4 (Cafeteria)"
        ]
        self.slots = []
        for i in range(4):
            self.slots.append({
                "id": i,
                "enabled": (i == 0),
                "source": "0" if i == 0 else "",
                "location": default_locations[i],
                "speed": "30x",
                "cap": None,
                "fps": 30.0,
                "current_frame": None,
                "spotted": False,
                "total_frames": 0,
                "duration_sec": 0.0,
                "current_frame_pos": 0,
                "is_file": False,
                "ended": False,
                "last_gray_thumb": None
            })

    def configure_slot(self, idx, enabled=None, source=None, location=None, speed=None):
        """Updates slot configuration parameters."""
        if 0 <= idx < len(self.slots):
            slot = self.slots[idx]
            if enabled is not None:
                slot["enabled"] = bool(enabled)
            if source is not None:
                slot["source"] = str(source).strip()
            if location is not None:
                slot["location"] = str(location).strip()
            if speed is not None:
                slot["speed"] = str(speed).strip()

    def start(self, global_speed="30x", on_finished_callback=None):
        """Opens video streams and starts the background scanning thread."""
        active_count = 0
        self.max_video_duration = 0.0

        for i, slot in enumerate(self.slots):
            if slot["enabled"]:
                raw_src = slot["source"]
                if raw_src == "" or raw_src == "0":
                    src = 0
                    slot["is_file"] = False
                else:
                    try:
                        src = int(raw_src)
                        slot["is_file"] = False
                    except ValueError:
                        src = raw_src
                        slot["is_file"] = os.path.isfile(raw_src)

                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    slot["cap"] = cap
                    slot["spotted"] = False
                    slot["ended"] = False
                    slot["last_gray_thumb"] = None
                    slot["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    slot["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if slot["is_file"] else 0
                    slot["duration_sec"] = (slot["total_frames"] / slot["fps"]) if slot["is_file"] else 0.0
                    if slot["duration_sec"] > self.max_video_duration:
                        self.max_video_duration = slot["duration_sec"]
                    active_count += 1
                else:
                    self.stop()
                    raise ConnectionError(f"Could not connect to Camera {i+1} source:\n{raw_src}")
            else:
                slot["cap"] = None

        if active_count == 0:
            raise ValueError("Please enable at least one Camera slot!")

        self.is_running = True
        self.global_clock_sec = 0.0
        self.worker_thread = threading.Thread(
            target=self._synchronized_scan_loop,
            args=(global_speed, on_finished_callback),
            daemon=True
        )
        self.worker_thread.start()
        return active_count

    def stop(self):
        """Halts the worker thread and safely releases all camera resources."""
        self.is_running = False
        with self._lock:
            for slot in self.slots:
                if slot["cap"] is not None:
                    try:
                        slot["cap"].release()
                    except Exception:
                        pass
                    slot["cap"] = None
                slot["spotted"] = False
                slot["current_frame"] = None

    def get_slot_state(self, idx):
        """Returns thread-safe snapshot of slot state for GUI rendering."""
        if 0 <= idx < len(self.slots):
            with self._lock:
                slot = self.slots[idx]
                return {
                    "enabled": slot["enabled"],
                    "frame": slot["current_frame"],
                    "spotted": slot["spotted"],
                    "location": slot["location"],
                    "ended": slot["ended"]
                }
        return None

    def get_clock_display(self):
        """Returns formatted global clock string (HH:MM:SS) and total seconds."""
        cur_sec = int(self.global_clock_sec)
        gh = cur_sec // 3600
        gm = (cur_sec % 3600) // 60
        gs = cur_sec % 60
        return f"{gh:02d}:{gm:02d}:{gs:02d}", cur_sec

    def _synchronized_scan_loop(self, global_speed, on_finished_callback):
        """Background worker thread advancing cameras along synchronized timeline."""
        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            step_frames, sleep_time = self.speed_map.get(global_speed, (10, 0.005))
            any_active = False
            all_ended = True

            cur_sec = int(self.global_clock_sec)
            gh = cur_sec // 3600
            gm = (cur_sec % 3600) // 60
            gs = cur_sec % 60
            global_time_str = f"{gh:02d}:{gm:02d}:{gs:02d}"

            for i, slot in enumerate(self.slots):
                if not self.is_running:
                    break
                cap = slot["cap"]
                if cap is None or not cap.isOpened() or not slot["enabled"]:
                    continue

                any_active = True

                # Camera-specific playback speed
                cam_spd = slot["speed"]
                cam_step_frames, _ = self.speed_map.get(cam_spd, (10, 0.005))
                is_fast = cam_spd in ["10x", "30x", "60x"]

                if slot["is_file"]:
                    slot["current_frame_pos"] += cam_step_frames
                    if slot["total_frames"] > 0 and slot["current_frame_pos"] >= slot["total_frames"]:
                        slot["ended"] = True
                        continue
                    else:
                        all_ended = False
                        cap.set(cv2.CAP_PROP_POS_FRAMES, slot["current_frame_pos"])

                    cam_sec = int(slot["current_frame_pos"] / max(1.0, slot["fps"]))
                    ch = cam_sec // 3600
                    cm_ = (cam_sec % 3600) // 60
                    cs = cam_sec % 60
                    cam_time_str = f"{ch:02d}:{cm_:02d}:{cs:02d}"
                else:
                    all_ended = False
                    cam_time_str = global_time_str

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                loc_name = slot["location"].strip() or f"Camera {i+1}"

                # ── Smart Motion Gating (0.18ms Check) ──
                should_run_ai = True
                if is_fast and slot["is_file"]:
                    small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_NEAREST)
                    last_thumb = slot.get("last_gray_thumb")
                    if last_thumb is not None:
                        diff = cv2.absdiff(small_gray, last_thumb)
                        motion_pixels = int(np.count_nonzero(diff > 22))
                        if motion_pixels < 35 and not slot["spotted"] and (cur_sec % 5 != 0):
                            should_run_ai = False
                    slot["last_gray_thumb"] = small_gray

                if not should_run_ai:
                    with self._lock:
                        slot["spotted"] = False
                        slot["current_frame"] = frame
                    continue

                # Run AI Scanner on current frame
                annotated_frame, spotted = self.scanner.process_frame(
                    frame, camera_name=loc_name, video_time_str=cam_time_str, fast_mode=is_fast
                )

                with self._lock:
                    slot["spotted"] = spotted
                    slot["current_frame"] = annotated_frame

                if spotted and self.scanner.last_sighting["found"]:
                    self.last_spotlight_idx = i
                    self.tracker.log_sighting(
                        camera_name=self.scanner.last_sighting["camera_name"],
                        confidence=self.scanner.last_sighting["confidence"],
                        snapshot_path=self.scanner.last_sighting.get("snapshot_path", ""),
                        video_time=cam_time_str
                    )

            # Advance global clock synchronously
            step_seconds = step_frames / 30.0
            self.global_clock_sec += step_seconds

            # Check if all video files finished
            if any_active and all_ended and self.max_video_duration > 0:
                if on_finished_callback:
                    on_finished_callback()
                break

            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            time.sleep(sleep_time)
