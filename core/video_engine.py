"""
VideoEngine: Manages parallel multi-camera video streams, time synchronization,
dynamic real-world speed measurement, 4-core concurrency, and background AI scanning.
"""

import os
import re
import sys
import cv2
import time
import socket
import threading
import urllib.request
import urllib.error
import concurrent.futures
import numpy as np
import torch


def _cap_height_480(frame):
    """Downscales frames taller than 480px once so UI and AI reuse lightweight frames."""
    if frame is None:
        return None
    fh, fw = frame.shape[:2]
    if fh > 480:
        scale = 480.0 / fh
        return cv2.resize(frame, (max(1, int(fw * scale)), 480), interpolation=cv2.INTER_LINEAR)
    return frame


class _LiveStreamReader:
    """
    Zero-buffer background frame grabber for live HTTP MJPEG, RTSP, and local webcams.
    Always retains only the newest decoded frame so FIFO buffers never cause lag.
    """

    def __init__(self, source):
        self.source = source
        self.is_http = isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        )
        self.is_mjpeg = False
        self.boundary_bytes = None
        self.fps = 30.0

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._resp = None
        self._cap = None

        self.latest_frame = None
        self.frame_seq = 0
        self.last_frame_time = 0.0

    def open(self, timeout=4.5):
        """Connects to the live source and waits for the first valid frame."""
        if self.is_http:
            opened_mjpeg = self._try_open_http_mjpeg(timeout=timeout)
            if opened_mjpeg:
                self.is_mjpeg = True
                self._running = True
                self._thread = threading.Thread(target=self._mjpeg_loop, daemon=True)
                self._thread.start()
                if self._wait_first_frame(timeout=timeout):
                    return True
                self.stop()
                return False

        # Fallback: non-MJPEG HTTP (e.g. HLS/MP4), RTSP, or local Webcam (0, 1)
        cap = self._open_cv_capture(self.source)
        if cap is None or not cap.isOpened():
            return False

        self._cap = cap
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self.fps = fps if 1.0 <= fps <= 120.0 else 30.0
        except Exception:
            self.fps = 30.0

        # Grab an initial frame before returning so invalid/blocked sources fail fast
        ret, frame = cap.read()
        if not ret or frame is None:
            self.stop()
            return False

        with self._lock:
            self.latest_frame = _cap_height_480(frame)
            self.frame_seq = 1
            self.last_frame_time = time.time()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _try_open_http_mjpeg(self, timeout=4.0):
        try:
            req = urllib.request.Request(
                self.source,
                headers={"User-Agent": "SmartSupervise/1.0", "Connection": "close"},
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            ctype = resp.headers.get("Content-Type") or ""
            ctype_lower = ctype.lower()
            if "multipart/" in ctype_lower or "image/jpeg" in ctype_lower:
                m = re.search(r'boundary=([^;]+)', ctype, re.IGNORECASE)
                if m:
                    bval = m.group(1).strip().strip('"').strip("'")
                    self.boundary_bytes = bval.encode("latin-1", errors="ignore")
                else:
                    self.boundary_bytes = None
                try:
                    raw_sock = getattr(getattr(resp, "fp", None), "raw", None)
                    sock = getattr(raw_sock, "_sock", None)
                    if sock is not None:
                        sock.settimeout(timeout)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                self._resp = resp
                return True
            resp.close()
        except Exception:
            pass
        return False

    @staticmethod
    def _open_cv_capture(source):
        if isinstance(source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _wait_first_frame(self, timeout=4.5):
        deadline = time.time() + timeout
        while time.time() < deadline and self._running:
            with self._lock:
                if self.frame_seq > 0 and self.latest_frame is not None:
                    return True
            time.sleep(0.015)
        return False

    @staticmethod
    def _extract_latest_jpeg(buf, boundary_bytes):
        """
        Extracts the newest complete JPEG frame from buf and trims consumed bytes.
        Skips decoding stale queued JPEGs when multiple frames arrive together.
        """
        latest_jpg = None
        consumed = 0

        if boundary_bytes:
            core = boundary_bytes.lstrip(b"-")
            sep = b"--" + core if core else boundary_bytes
            pos = buf.find(sep)
            if pos == -1 and core and core != sep:
                sep = core
                pos = buf.find(sep)

            while pos != -1:
                next_sep = buf.find(sep, pos + len(sep))
                if next_sep != -1:
                    soi = buf.find(b"\xff\xd8", pos + len(sep), next_sep)
                    if soi != -1:
                        eoi = buf.rfind(b"\xff\xd9", soi + 2, next_sep)
                        if eoi != -1:
                            latest_jpg = bytes(buf[soi : eoi + 2])
                    consumed = next_sep
                    pos = next_sep
                    continue

                # Tail section: server often flushes the full JPEG before next boundary
                hdr_end = buf.find(b"\r\n\r\n", pos + len(sep))
                sep_len = 4
                if hdr_end == -1:
                    hdr_end = buf.find(b"\n\n", pos + len(sep))
                    sep_len = 2
                if hdr_end == -1:
                    break

                body_start = hdr_end + sep_len
                hdr_text = buf[pos + len(sep) : hdr_end].decode("latin-1", errors="ignore")
                cl_match = re.search(r"content-length\s*:\s*(\d+)", hdr_text, re.IGNORECASE)
                if cl_match:
                    cl_val = int(cl_match.group(1))
                    if cl_val > 0 and len(buf) >= body_start + cl_val:
                        part_end = body_start + cl_val
                        soi = buf.find(b"\xff\xd8", body_start, part_end)
                        eoi = buf.rfind(b"\xff\xd9", soi + 2, part_end) if soi != -1 else -1
                        if soi != -1 and eoi != -1:
                            latest_jpg = bytes(buf[soi : eoi + 2])
                        consumed = part_end
                else:
                    soi = buf.find(b"\xff\xd8", body_start)
                    if soi != -1:
                        sos = buf.find(b"\xff\xda", soi + 2)
                        search_from = sos + 2 if sos != -1 else soi + 2
                        eoi = buf.find(b"\xff\xd9", search_from)
                        if eoi != -1:
                            latest_jpg = bytes(buf[soi : eoi + 2])
                            consumed = eoi + 2
                break
        else:
            pos = 0
            while True:
                soi = buf.find(b"\xff\xd8", pos)
                if soi == -1:
                    break
                sos = buf.find(b"\xff\xda", soi + 2)
                search_from = sos + 2 if sos != -1 else soi + 2
                eoi = buf.find(b"\xff\xd9", search_from)
                if eoi == -1:
                    break
                latest_jpg = bytes(buf[soi : eoi + 2])
                pos = eoi + 2
                consumed = pos

        if consumed > 0:
            del buf[:consumed]
        elif len(buf) > 4 * 1024 * 1024:
            del buf[:-262144]

        return latest_jpg

    def _mjpeg_loop(self):
        buf = bytearray()
        while self._running:
            resp = self._resp
            if resp is None:
                if not self._running:
                    break
                buf.clear()
                if not self._try_open_http_mjpeg(timeout=3.0):
                    time.sleep(0.25)
                    continue
                resp = self._resp

            try:
                read_fn = getattr(resp, "read1", None) or resp.read
                chunk = read_fn(65536)
                if not chunk:
                    raise EOFError("MJPEG stream ended")
                buf.extend(chunk)
                latest_jpg = self._extract_latest_jpeg(buf, self.boundary_bytes)
                if latest_jpg is not None:
                    arr = np.frombuffer(latest_jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        scaled = _cap_height_480(frame)
                        with self._lock:
                            self.latest_frame = scaled
                            self.frame_seq += 1
                            self.last_frame_time = time.time()
            except Exception:
                try:
                    resp.close()
                except Exception:
                    pass
                self._resp = None
                if self._running:
                    time.sleep(0.2)

    def _capture_loop(self):
        while self._running:
            cap = self._cap
            if cap is None or not cap.isOpened():
                if not self._running:
                    break
                time.sleep(0.25)
                self._cap = self._open_cv_capture(self.source)
                continue

            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.is_http or (isinstance(self.source, str) and self.source.startswith("rtsp://")):
                        try:
                            cap.release()
                        except Exception:
                            pass
                        self._cap = None
                        time.sleep(0.2)
                    else:
                        time.sleep(0.01)
                    continue

                scaled = _cap_height_480(frame)
                with self._lock:
                    self.latest_frame = scaled
                    self.frame_seq += 1
                    self.last_frame_time = time.time()
            except Exception:
                if not self._running:
                    break
                time.sleep(0.05)

    def get_latest(self):
        """Returns (frame_copy_or_None, seq, last_frame_time)."""
        with self._lock:
            if self.latest_frame is None:
                return None, self.frame_seq, self.last_frame_time
            return self.latest_frame.copy(), self.frame_seq, self.last_frame_time

    def stop(self):
        self._running = False
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
        self._thread = None


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
    @staticmethod
    def normalize_stream_source(raw_src: str) -> str:
        """Auto-formats phone IP streams, RTSP, and local webcams."""
        src = str(raw_src).strip()
        if not src or src == "0":
            return "0"
        import re
        ip_match = re.match(r'^(?:https?://)?(\d{1,3}(?:\.\d{1,3}){3})(?::(\d+))?(?:/(.*))?$', src)
        if ip_match:
            ip = ip_match.group(1)
            port = ip_match.group(2) or "8080"
            path = ip_match.group(3)
            if not path or path.strip() == "":
                path = "video"
            return f"http://{ip}:{port}/{path}"
        return src

    def __init__(self, scanner, tracker):
        self.scanner = scanner
        self.tracker = tracker

        self.is_running = False
        self.is_paused = False
        self.current_speed = "10x"
        self.worker_thread = None
        self.live_ai_thread = None
        self.has_live_slots = False
        self.has_file_slots = False
        self._lock = threading.RLock()

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
                "reader": None,
                "fps": 30.0,
                "current_frame": None,
                "frame_seq": 0,
                "last_display_seq": -1,
                "last_ai_seq": -1,
                "last_ai_time": 0.0,
                "cached_detections": [],
                "cached_spotted": False,
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
        if self.is_running:
            self.stop()

        active_count = 0
        file_count = 0
        live_count = 0
        self.max_video_duration = 0.0

        # Set network capture options for wireless phone cameras & RTSP (3s timeout)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|rtsp_transport;tcp"

        for i, slot in enumerate(self.slots):
            slot["spotted"] = False
            slot["ended"] = False
            slot["last_gray_thumb"] = None
            slot["current_frame_pos"] = 0
            slot["current_frame"] = None
            slot["frame_seq"] = 0
            slot["last_display_seq"] = -1
            slot["last_ai_seq"] = -1
            slot["last_ai_time"] = 0.0
            slot["cached_detections"] = []
            slot["cached_spotted"] = False
            slot["reader"] = None
            slot["cap"] = None

            if slot["enabled"]:
                raw_src = slot["source"]
                norm_src = self.normalize_stream_source(raw_src)
                slot["source"] = norm_src

                if norm_src == "" or norm_src == "0":
                    src = 0
                    slot["is_file"] = False
                else:
                    try:
                        src = int(norm_src)
                        slot["is_file"] = False
                    except ValueError:
                        src = norm_src
                        slot["is_file"] = os.path.isfile(norm_src)

                # For live streams (phone / webcam / IP stream), lock playback speed to 1x
                if not slot["is_file"] and slot["speed"] in ["5x", "10x", "20x", "40x"]:
                    slot["speed"] = "1x"

                if slot["is_file"]:
                    cap = cv2.VideoCapture(src)
                    if cap.isOpened():
                        slot["cap"] = cap
                        slot["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        slot["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        slot["duration_sec"] = slot["total_frames"] / max(1.0, slot["fps"])
                        if slot["duration_sec"] > self.max_video_duration:
                            self.max_video_duration = slot["duration_sec"]
                        file_count += 1
                        active_count += 1
                        continue
                else:
                    is_live_stream = isinstance(src, int) or str(src).startswith("http://") or str(src).startswith("https://") or str(src).startswith("rtsp://")
                    if is_live_stream:
                        reader = _LiveStreamReader(src)
                        if reader.open(timeout=4.5):
                            slot["reader"] = reader
                            slot["fps"] = reader.fps
                            slot["total_frames"] = 0
                            slot["duration_sec"] = 0.0
                            live_count += 1
                            active_count += 1
                            continue
                        reader.stop()

                self.stop()
                if isinstance(src, int):
                    hint = (
                        f"Could not open Webcam (index {src}) on Camera {i+1}.\n\n"
                        "Quick Fixes:\n"
                        "• Close any app using your camera (Zoom, Teams, Meet, or Windows Camera app).\n"
                        "• Check Windows Settings → Privacy & Security → Camera (Allow apps access).\n"
                        "• If you have an external or IR camera, try index '1'."
                    )
                elif str(src).startswith("http://") or str(src).startswith("https://") or str(src).startswith("rtsp://"):
                    hint = (
                        f"Could not connect to Phone / IP Camera {i+1}:\n{raw_src}\n\n"
                        "Quick Fixes:\n"
                        "1. Make sure you tapped 'Start server' inside the phone app.\n"
                        "2. Ensure laptop & phone are on the SAME Wi-Fi (or turn on Phone Hotspot).\n"
                        f"3. Test in Chrome: Open '{raw_src}' in your browser to verify it connects."
                    )
                else:
                    hint = f"Could not read video file for Camera {i+1}:\n{raw_src}\nPlease verify the file exists."
                raise ConnectionError(hint)

        if active_count == 0:
            raise ValueError("Please enable at least one Camera slot!")

        self.has_file_slots = file_count > 0
        self.has_live_slots = live_count > 0
        self.is_running = True
        self.is_paused = False
        self.current_speed = str(global_speed).strip()
        self.global_clock_sec = 0.0
        self.start_clock_sec = 0.0
        self.start_real_time = time.time()
        self.actual_speed_multiplier = 1.0
        self.current_steps_per_sec = 0.0

        if self.has_live_slots:
            self.live_ai_thread = threading.Thread(
                target=self._live_ai_loop,
                daemon=True
            )
            self.live_ai_thread.start()

        if self.has_file_slots:
            self.worker_thread = threading.Thread(
                target=self._parallel_scan_loop,
                args=(on_finished_callback,),
                daemon=True
            )
            self.worker_thread.start()
        return active_count

    def _sync_live_clock_locked(self):
        """Advances global_clock_sec from wall clock when running pure live streams."""
        if self.is_running and not self.is_paused and self.has_live_slots and not self.has_file_slots:
            self.global_clock_sec = self.start_clock_sec + max(0.0, time.time() - self.start_real_time)

    def toggle_pause(self):
        """Toggles pause/resume state. Returns True if now paused, False if resumed."""
        with self._lock:
            if not self.is_paused:
                self._sync_live_clock_locked()
            self.is_paused = not self.is_paused
            if not self.is_paused:
                self.start_real_time = time.time()
                self.start_clock_sec = self.global_clock_sec
            return self.is_paused

    def pause(self):
        with self._lock:
            if not self.is_paused:
                self._sync_live_clock_locked()
            self.is_paused = True

    def resume(self):
        with self._lock:
            self.is_paused = False
            self.start_real_time = time.time()
            self.start_clock_sec = self.global_clock_sec

    def set_global_speed(self, new_speed):
        """Dynamically adjusts search speed mid-playback without restarting feeds."""
        with self._lock:
            self.current_speed = str(new_speed).strip()
            self.start_real_time = time.time()
            self.start_clock_sec = self.global_clock_sec
            for slot in self.slots:
                if slot["is_file"]:
                    slot["speed"] = self.current_speed

    def stop(self):
        """Halts the worker threads safely and releases camera resources without race conditions."""
        self.is_running = False
        self.is_paused = False
        self.has_live_slots = False
        self.has_file_slots = False

        if self.worker_thread is not None and self.worker_thread.is_alive():
            try:
                self.worker_thread.join(timeout=1.5)
            except Exception:
                pass
            self.worker_thread = None

        if self.live_ai_thread is not None and self.live_ai_thread.is_alive():
            try:
                self.live_ai_thread.join(timeout=1.5)
            except Exception:
                pass
            self.live_ai_thread = None

        with self._lock:
            for slot in self.slots:
                reader = slot.get("reader")
                if reader is not None:
                    try:
                        reader.stop()
                    except Exception:
                        pass
                    slot["reader"] = None
                if slot["cap"] is not None:
                    try:
                        slot["cap"].release()
                    except Exception:
                        pass
                    slot["cap"] = None
                slot["spotted"] = False
                slot["cached_spotted"] = False
                slot["cached_detections"] = []
                slot["current_frame"] = None

    def get_slot_state(self, idx):
        """Returns thread-safe snapshot of slot state for GUI rendering with per-camera progress."""
        if not (0 <= idx < len(self.slots)):
            return None

        slot = self.slots[idx]
        reader = slot.get("reader")

        # Decoupled live rendering: overlay latest AI boxes onto every fresh raw camera frame (~30 FPS)
        if slot["enabled"] and not slot["is_file"] and reader is not None and self.is_running and not self.is_paused:
            raw_frame, raw_seq, _ = reader.get_latest()
            need_draw = False
            with self._lock:
                self._sync_live_clock_locked()
                if raw_frame is not None and raw_seq != slot["last_display_seq"]:
                    need_draw = True
                    cached_dets = list(slot["cached_detections"])
                    cached_spotted = slot["cached_spotted"]
                    loc_name = slot["location"].strip() or f"Camera {slot['id'] + 1}"
                    cur_sec = int(self.global_clock_sec)
                    cam_time_str = f"{cur_sec // 3600:02d}:{(cur_sec % 3600) // 60:02d}:{cur_sec % 60:02d}"
            if need_draw:
                annotated = self.scanner.draw_overlays(
                    raw_frame,
                    cached_dets,
                    cached_spotted,
                    camera_name=loc_name,
                    video_time_str=cam_time_str,
                )
                with self._lock:
                    slot["current_frame"] = annotated
                    slot["last_display_seq"] = raw_seq
                    slot["frame_seq"] = raw_seq
                    slot["spotted"] = cached_spotted

        with self._lock:
            is_file = slot["is_file"]
            tot_frames = slot["total_frames"]
            cur_pos = slot["current_frame_pos"]
            fps = max(1.0, slot["fps"])
            ended = slot["ended"]

            if not slot["enabled"]:
                pct = 0.0
                progress_str = "Disabled"
            elif not is_file:
                pct = 100.0
                progress_str = "Live Stream"
            elif ended or (tot_frames > 0 and cur_pos >= tot_frames):
                pct = 100.0
                tot_sec = int(slot["duration_sec"]) if slot["duration_sec"] > 0 else int(tot_frames / fps)
                tot_m, tot_s = tot_sec // 60, tot_sec % 60
                progress_str = f"100% ({tot_m:02d}:{tot_s:02d})"
            elif tot_frames > 0:
                pct = min(99.9, max(0.0, (cur_pos / tot_frames) * 100.0))
                cur_sec = int(cur_pos / fps)
                tot_sec = int(slot["duration_sec"]) if slot["duration_sec"] > 0 else int(tot_frames / fps)
                cur_m, cur_s = cur_sec // 60, cur_sec % 60
                tot_m, tot_s = tot_sec // 60, tot_sec % 60
                progress_str = f"{pct:.1f}% ({cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d})"
            else:
                pct = 0.0
                progress_str = "0.0%"

            # Return isolated copy of frame so GUI thread can resize safely
            frame_copy = slot["current_frame"].copy() if slot["current_frame"] is not None else None

            return {
                "enabled": slot["enabled"],
                "frame": frame_copy,
                "frame_seq": slot.get("frame_seq", 0),
                "spotted": slot["spotted"],
                "location": slot["location"],
                "ended": ended,
                "is_file": is_file,
                "progress_pct": pct,
                "progress_str": progress_str,
                "speed": slot["speed"]
            }

    def get_clock_display(self):
        """Returns formatted global clock string (HH:MM:SS) and total seconds."""
        with self._lock:
            self._sync_live_clock_locked()
            cur_sec = int(self.global_clock_sec)
        gh = cur_sec // 3600
        gm = (cur_sec % 3600) // 60
        gs = cur_sec % 60
        return f"{gh:02d}:{gm:02d}:{gs:02d}", cur_sec

    def get_progress(self):
        """Returns (percentage: float, cur_time_str: str, max_time_str: str, summary_str: str)"""
        with self._lock:
            self._sync_live_clock_locked()
            cur_sec = int(self.global_clock_sec)
            max_sec = int(self.max_video_duration)
            global_clock_val = self.global_clock_sec
        if max_sec > 0:
            pct = min(100.0, max(0.0, (global_clock_val / self.max_video_duration) * 100.0))
        else:
            pct = 0.0

        cur_str = f"{cur_sec // 3600:02d}:{(cur_sec % 3600) // 60:02d}:{cur_sec % 60:02d}"
        max_str = f"{max_sec // 3600:02d}:{(max_sec % 3600) // 60:02d}:{max_sec % 60:02d}" if max_sec > 0 else "Live"
        summary = f"{pct:.1f}% ({cur_str} / {max_str})" if max_sec > 0 else cur_str
        return pct, cur_str, max_str, summary

    def get_speed_telemetry(self):
        """Calculates actual measured speed, steps/sec, ETA, and hardware device."""
        with self._lock:
            self._sync_live_clock_locked()
            clock_val = self.global_clock_sec
            start_clock = self.start_clock_sec
            start_real = self.start_real_time
            pure_live = self.has_live_slots and not self.has_file_slots

        real_elapsed = max(0.001, time.time() - start_real) if self.is_running else 1.0
        if pure_live:
            actual_speed = 1.0
            self.actual_speed_multiplier = 1.0
        elif real_elapsed < 0.5:
            actual_speed = self.actual_speed_multiplier
        else:
            actual_speed = (clock_val - start_clock) / real_elapsed
            self.actual_speed_multiplier = actual_speed

        remaining_video = max(0.0, self.max_video_duration - clock_val)
        if pure_live:
            eta_str = "Live Stream"
        elif actual_speed > 0.1 and self.max_video_duration > 0:
            eta_sec = int(remaining_video / actual_speed)
            eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s"
        else:
            eta_str = "—"

        device_str = "⚡ NVIDIA GPU (CUDA)" if self.has_cuda else "💻 CPU (4-Core Parallel)"
        pct, _, _, progress_summary = self.get_progress()

        if self.is_paused:
            actual_speed_str = "⏸ PAUSED"
            eta_str = "Paused"
        else:
            actual_speed_str = f"{actual_speed:.1f}x"

        return {
            "actual_speed": round(actual_speed, 1),
            "actual_speed_str": actual_speed_str,
            "steps_per_sec": round(self.current_steps_per_sec, 1) if not self.is_paused else 0.0,
            "eta_str": eta_str,
            "device": device_str,
            "progress_pct": round(pct, 1),
            "progress_str": progress_summary
        }

    def _process_camera_slot(self, slot, global_speed, global_time_str, cur_sec):
        """Processes a single camera stream (Runs inside parallel ThreadPool worker)."""
        try:
            cap = slot["cap"]
            if cap is None or not cap.isOpened() or not slot["enabled"] or not self.is_running:
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
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, slot["current_frame_pos"])
                    except Exception:
                        pass

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
                try:
                    small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_NEAREST)
                    last_thumb = slot.get("last_gray_thumb")
                    if last_thumb is not None:
                        diff = cv2.absdiff(small_gray, last_thumb)
                        motion_pixels = int(np.count_nonzero(diff > 22))
                        if motion_pixels < 35 and not slot["spotted"] and (cur_sec % 4 != 0):
                            should_run_ai = False
                    slot["last_gray_thumb"] = small_gray
                except Exception:
                    pass

            if not should_run_ai:
                # Pre-downscale preview frame in background worker to save GUI thread CPU
                preview = _cap_height_480(frame)
                if preview is frame:
                    preview = frame.copy()
                with self._lock:
                    slot["spotted"] = False
                    slot["current_frame"] = preview
                    slot["frame_seq"] = slot.get("frame_seq", 0) + 1
                return {"id": slot["id"], "spotted": False, "ended": False}

            # Run AI Scanner on current frame (High speed inference)
            annotated_frame, spotted, match_info = self.scanner.process_frame(
                frame, camera_name=loc_name, video_time_str=cam_time_str, fast_mode=is_fast
            )

            raw_out = annotated_frame if annotated_frame is not None else frame
            out_frame = _cap_height_480(raw_out)
            if out_frame is raw_out:
                out_frame = raw_out.copy()

            with self._lock:
                slot["spotted"] = spotted
                slot["current_frame"] = out_frame
                slot["frame_seq"] = slot.get("frame_seq", 0) + 1

            if spotted and match_info:
                self.last_spotlight_idx = slot["id"]
                self.tracker.log_sighting(
                    camera_name=loc_name,
                    confidence=match_info.get("confidence", 0.8),
                    snapshot_path=match_info.get("snapshot_path", ""),
                    video_time=cam_time_str,
                    camera_id=slot["id"]
                )

            return {"id": slot["id"], "spotted": spotted, "ended": False}
        except Exception:
            return {"id": slot["id"], "spotted": False, "ended": False}

    def _live_ai_loop(self):
        """
        Decoupled Live Stream AI Worker:
        Runs 320px YOLOv8 + Motion Gating asynchronously on latest frames from _LiveStreamReader
        without ever blocking network frame acquisition or 30 FPS display rendering.
        """
        live_sweeps = 0
        rate_timer = time.time()

        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.04)
                    continue

                with self._lock:
                    self._sync_live_clock_locked()
                    cur_sec = int(self.global_clock_sec)
                    live_slots = [
                        s for s in self.slots
                        if s["enabled"] and not s["is_file"] and s.get("reader") is not None
                    ]

                if not live_slots:
                    time.sleep(0.05)
                    continue

                cam_time_str = f"{cur_sec // 3600:02d}:{(cur_sec % 3600) // 60:02d}:{cur_sec % 60:02d}"
                did_work = False

                for slot in live_slots:
                    if not self.is_running or self.is_paused:
                        break
                    reader = slot.get("reader")
                    if reader is None:
                        continue

                    frame, raw_seq, _ = reader.get_latest()
                    if frame is None or raw_seq == slot["last_ai_seq"]:
                        continue

                    did_work = True
                    now = time.time()
                    should_run_ai = True
                    try:
                        small_gray = cv2.resize(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            (160, 90),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        last_thumb = slot.get("last_gray_thumb")
                        if last_thumb is not None:
                            diff = cv2.absdiff(small_gray, last_thumb)
                            motion_pixels = int(np.count_nonzero(diff > 22))
                            if motion_pixels < 25 and (now - slot["last_ai_time"]) < 0.8:
                                should_run_ai = False
                        slot["last_gray_thumb"] = small_gray
                    except Exception:
                        pass

                    slot["last_ai_seq"] = raw_seq
                    if not should_run_ai:
                        continue

                    loc_name = slot["location"].strip() or f"Camera {slot['id'] + 1}"
                    detections, spotted, match_info = self.scanner.analyze_frame(
                        frame,
                        camera_name=loc_name,
                        video_time_str=cam_time_str,
                        fast_mode=True,
                    )

                    with self._lock:
                        slot["cached_detections"] = detections
                        slot["cached_spotted"] = spotted
                        slot["spotted"] = spotted
                        slot["last_ai_time"] = time.time()
                        # Trigger immediate redraw if display already drew this raw_seq before AI finished
                        if slot["last_display_seq"] == raw_seq:
                            slot["last_display_seq"] = -1

                    live_sweeps += 1

                    if spotted and match_info:
                        self.last_spotlight_idx = slot["id"]
                        self.tracker.log_sighting(
                            camera_name=loc_name,
                            confidence=match_info.get("confidence", 0.8),
                            snapshot_path=match_info.get("snapshot_path", ""),
                            video_time=cam_time_str,
                            camera_id=slot["id"]
                        )

                elapsed = time.time() - rate_timer
                if elapsed >= 1.0:
                    if not self.has_file_slots:
                        self.current_steps_per_sec = live_sweeps / elapsed
                    live_sweeps = 0
                    rate_timer = time.time()

                if not did_work:
                    time.sleep(0.008)
            except Exception:
                if not self.is_running:
                    break
                time.sleep(0.02)

    def _parallel_scan_loop(self, on_finished_callback):
        """
        Parallel 4-Core Worker Loop:
        Executes recorded video file slots simultaneously using ThreadPoolExecutor across CPU cores.
        Supports real-time mid-video speed adjustment and instant freeze/pause.
        """
        step_counter = 0
        timer = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while self.is_running:
                try:
                    if self.is_paused:
                        time.sleep(0.04)
                        continue

                    with self._lock:
                        cur_speed = self.current_speed

                    step_frames, sleep_time = self.speed_map.get(cur_speed, (105, 0.002))
                    cur_sec = int(self.global_clock_sec)
                    gh = cur_sec // 3600
                    gm = (cur_sec % 3600) // 60
                    gs = cur_sec % 60
                    global_time_str = f"{gh:02d}:{gm:02d}:{gs:02d}"

                    active_slots = [
                        s for s in self.slots
                        if s["enabled"] and s["is_file"] and s["cap"] is not None
                    ]
                    if not active_slots or not self.is_running:
                        break

                    # Dispatch all active file cameras in parallel across 4 CPU cores
                    futures = [
                        executor.submit(self._process_camera_slot, s, cur_speed, global_time_str, cur_sec)
                        for s in active_slots
                    ]

                    # Wait for all camera inferences in this parallel step
                    all_ended = True
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            res = f.result()
                            if res and not res.get("ended", False):
                                all_ended = False
                        except Exception:
                            pass

                    # Advance global clock synchronously by step interval
                    step_seconds = step_frames / 30.0
                    self.global_clock_sec += step_seconds

                    # Check if all video files finished
                    if all_ended and self.max_video_duration > 0 and self.is_running:
                        self.is_running = False
                        if on_finished_callback:
                            try:
                                on_finished_callback()
                            except Exception:
                                pass
                        break

                    step_counter += 1
                    elapsed = time.time() - timer
                    if elapsed >= 1.0:
                        self.current_steps_per_sec = step_counter / elapsed
                        step_counter = 0
                        timer = time.time()

                    if sleep_time > 0 and self.is_running:
                        time.sleep(sleep_time)
                except Exception:
                    if not self.is_running:
                        break
                    time.sleep(0.01)
