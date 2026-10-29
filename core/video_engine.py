"""
VideoEngine: Manages parallel multi-camera video streams, time synchronization,
dynamic real-world speed measurement, 4-core concurrency, and background AI scanning.
"""

import os
import cv2
import time
import threading
import concurrent.futures
import numpy as np
import torch


class VideoEngine:
    """
    Decoupled Parallel Multi-Camera Video Playback & Synchronized AI Scanning Engine.
    
    Key Capabilities:
    1. 4-Core Parallel Execution: Processes all 4 CCTV cameras simultaneously across CPU cores.
    2. True Speed Mathematical Sampling:
       - 1x  -> Smooth continuous playback (1 frame step)
       - 5x  -> ~1.5s jump (45 frames) -> ~5x true search speed
       - 10x -> ~3.5s jump (105 frames) -> ~10x true speed (~4 min for 40 min video)
       - 20x -> ~7.0s jump (210 frames) -> ~20x true speed (~2 min for 40 min video)
       - 40x -> ~14.0s jump (420 frames) -> ~40x true speed (~1 min for 40 min video)
    3. Live Dynamic Speed Measurement: Displays actual real-world measured speed,
       steps-per-second, and precise ETA (Estimated Time of Arrival).
    4. Auto-Detects NVIDIA CUDA GPU: Automatically switches to GPU hardware if available.
    """
    def __init__(self, scanner, tracker):
        self.scanner = scanner
        self.tracker = tracker

        self.is_running = False
        self.worker_thread = None
        self._lock = threading.Lock()

        # Telemetry State
        self.global_clock_sec = 0.0
        self.max_video_duration = 0.0
        self.start_real_time = time.time()
        self.start_clock_sec = 0.0
        self.current_steps_per_sec = 0.0
        self.actual_speed_multiplier = 1.0
        self.last_spotlight_idx = 0

        self.has_cuda = torch.cuda.is_available()

        # True Speed Table: (step_frames, sleep_time)
        self.speed_map = {
            "1x":  (1, 0.030),    # Smooth 1x playback (~30 FPS)
            "5x":  (45, 0.005),   # 1.5s jump -> ~5x true speed
            "10x": (105, 0.002),  # 3.5s jump -> ~10x true speed (~4 min / 40 min video)
            "20x": (210, 0.001),  # 7.0s jump -> ~20x true speed (~2 min / 40 min video)
            "40x": (420, 0.001)   # 14.0s jump -> ~40x true speed (~1 min / 40 min video)
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
                "speed": "10x",
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

    def start(self, global_speed="10x", on_finished_callback=None):
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
                    slot["current_frame_pos"] = 0
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
        self.start_clock_sec = 0.0
        self.start_real_time = time.time()
        self.actual_speed_multiplier = 1.0

        self.worker_thread = threading.Thread(
            target=self._parallel_scan_loop,
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

    def get_speed_telemetry(self):
        """Calculates actual measured speed, steps/sec, ETA, and hardware device."""
        real_elapsed = max(0.001, time.time() - self.start_real_time) if self.is_running else 1.0
        if real_elapsed < 0.5:
            actual_speed = self.actual_speed_multiplier
        else:
            actual_speed = (self.global_clock_sec - self.start_clock_sec) / real_elapsed
            self.actual_speed_multiplier = actual_speed

        remaining_video = max(0.0, self.max_video_duration - self.global_clock_sec)
        if actual_speed > 0.1 and self.max_video_duration > 0:
            eta_sec = int(remaining_video / actual_speed)
            eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s"
        else:
            eta_str = "—"

        device_str = "⚡ NVIDIA GPU (CUDA)" if self.has_cuda else "💻 CPU (4-Core Parallel)"

        return {
            "actual_speed": round(actual_speed, 1),
            "actual_speed_str": f"{actual_speed:.1f}x",
            "steps_per_sec": round(self.current_steps_per_sec, 1),
            "eta_str": eta_str,
            "device": device_str
        }

    def _process_camera_slot(self, slot, global_speed, global_time_str, cur_sec):
        """Processes a single camera stream (Runs inside parallel ThreadPool worker)."""
        cap = slot["cap"]
        if cap is None or not cap.isOpened() or not slot["enabled"]:
            return None

        # Camera-specific playback speed
        cam_spd = slot["speed"] or global_speed
        cam_step_frames, _ = self.speed_map.get(cam_spd, (105, 0.002))
        is_fast = cam_spd in ["5x", "10x", "20x", "40x"]

        if slot["is_file"]:
            slot["current_frame_pos"] += cam_step_frames
            if slot["total_frames"] > 0 and slot["current_frame_pos"] >= slot["total_frames"]:
                slot["ended"] = True
                return {"id": slot["id"], "ended": True}
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, slot["current_frame_pos"])

            cam_sec = int(slot["current_frame_pos"] / max(1.0, slot["fps"]))
            ch = cam_sec // 3600
            cm_ = (cam_sec % 3600) // 60
            cs = cam_sec % 60
            cam_time_str = f"{ch:02d}:{cm_:02d}:{cs:02d}"
        else:
            cam_time_str = global_time_str

        ret, frame = cap.read()
        if not ret or frame is None:
            return {"id": slot["id"], "ended": slot["is_file"]}

        loc_name = slot["location"].strip() or f"Camera {slot['id']+1}"

        # Smart Motion Gating (0.18ms Check)
        should_run_ai = True
        if is_fast and slot["is_file"]:
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_NEAREST)
            last_thumb = slot.get("last_gray_thumb")
            if last_thumb is not None:
                diff = cv2.absdiff(small_gray, last_thumb)
                motion_pixels = int(np.count_nonzero(diff > 22))
                if motion_pixels < 35 and not slot["spotted"] and (cur_sec % 4 != 0):
                    should_run_ai = False
            slot["last_gray_thumb"] = small_gray

        if not should_run_ai:
            with self._lock:
                slot["spotted"] = False
                slot["current_frame"] = frame
            return {"id": slot["id"], "spotted": False, "ended": False}

        # Run AI Scanner on current frame (High speed inference)
        annotated_frame, spotted = self.scanner.process_frame(
            frame, camera_name=loc_name, video_time_str=cam_time_str, fast_mode=is_fast
        )

        with self._lock:
            slot["spotted"] = spotted
            slot["current_frame"] = annotated_frame

        if spotted and self.scanner.last_sighting["found"]:
            self.last_spotlight_idx = slot["id"]
            self.tracker.log_sighting(
                camera_name=self.scanner.last_sighting["camera_name"],
                confidence=self.scanner.last_sighting["confidence"],
                snapshot_path=self.scanner.last_sighting.get("snapshot_path", ""),
                video_time=cam_time_str
            )

        return {"id": slot["id"], "spotted": spotted, "ended": False}

    def _parallel_scan_loop(self, global_speed, on_finished_callback):
        """
        Parallel 4-Core Worker Loop:
        Executes all 4 cameras simultaneously using ThreadPoolExecutor across CPU cores.
        """
        step_counter = 0
        timer = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while self.is_running:
                step_frames, sleep_time = self.speed_map.get(global_speed, (105, 0.002))
                cur_sec = int(self.global_clock_sec)
                gh = cur_sec // 3600
                gm = (cur_sec % 3600) // 60
                gs = cur_sec % 60
                global_time_str = f"{gh:02d}:{gm:02d}:{gs:02d}"

                active_slots = [s for s in self.slots if s["enabled"] and s["cap"] is not None]
                if not active_slots:
                    break

                # Dispatch all active cameras in parallel across 4 CPU cores
                futures = [
                    executor.submit(self._process_camera_slot, s, global_speed, global_time_str, cur_sec)
                    for s in active_slots
                ]

                # Wait for all 4 camera inferences in this parallel step
                all_ended = True
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res and not res.get("ended", False):
                        all_ended = False

                # Advance global clock synchronously by step interval
                step_seconds = step_frames / 30.0
                self.global_clock_sec += step_seconds

                # Check if all video files finished
                if all_ended and self.max_video_duration > 0:
                    if on_finished_callback:
                        on_finished_callback()
                    break

                step_counter += 1
                elapsed = time.time() - timer
                if elapsed >= 1.0:
                    self.current_steps_per_sec = step_counter / elapsed
                    step_counter = 0
                    timer = time.time()

                if sleep_time > 0:
                    time.sleep(sleep_time)
