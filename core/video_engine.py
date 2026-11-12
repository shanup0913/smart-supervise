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

        # Grab an initial frame (with brief warmup retries for slow webcams/RTSP sensors)
        ret, frame = False, None
        deadline = time.time() + 1.2
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.05)
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

    @staticmethod
    def is_live_source(raw_src: str) -> bool:
        """Returns True if source text refers to a webcam index or HTTP/RTSP live camera."""
        norm = VideoEngine.normalize_stream_source(raw_src)
        if not norm or norm == "0" or norm.isdigit():
            return True
        lower = norm.lower()
        return lower.startswith("http://") or lower.startswith("https://") or lower.startswith("rtsp://")

    def __init__(self, scanner, tracker):
        self.scanner = scanner
        self.tracker = tracker

        self.is_running = False
        self.is_paused = False
        self.current_speed = "10x"
        self.worker_thread = None
        self.realtime_thread = None
        self.live_ai_thread = None
        self.has_live_slots = False
        self.has_file_slots = False
        self.has_realtime_file_slots = False
        self.has_fast_file_slots = False
        self._on_finished_callback = None
        self._lock = threading.RLock()
        self._custom_sources = {}

        # Telemetry State
        self.global_clock_sec = 0.0
        self.max_video_duration = 0.0
        self.start_real_time = time.time()
        self.start_clock_sec = 0.0
        self.current_steps_per_sec = 0.0
        self.actual_speed_multiplier = 1.0
        self.last_spotlight_idx = 0

        self.has_cuda = torch.cuda.is_available()

        # Target speed multipliers & min step frame budgets
        self.speed_multipliers = {
            "1x": 1.0,
            "5x": 5.0,
            "10x": 10.0,
            "20x": 20.0,
            "40x": 40.0,
        }
        self.speed_map = {
            "1x":  (1, 0.030),
            "5x":  (45, 0.005),
            "10x": (105, 0.002),
            "20x": (210, 0.001),
            "40x": (420, 0.001)
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
                "op_lock": threading.RLock(),
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
                "cap_next_pos": 0,
                "_rt_base_time": 0.0,
                "_rt_base_frame": 0,
                "_raw_seq": 0,
                "_latest_raw_frame": None,
                "is_file": False,
                "ended": False,
                "last_gray_thumb": None
            })

    @staticmethod
    def _close_slot_media(slot):
        """Safely closes a single slot's capture/reader under its per-slot lock."""
        with slot["op_lock"]:
            reader = slot.get("reader")
            if reader is not None:
                try:
                    reader.stop()
                except Exception:
                    pass
                slot["reader"] = None
            cap = slot.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                slot["cap"] = None

    @staticmethod
    def _reset_slot_runtime_fields(slot):
        slot["spotted"] = False
        slot["ended"] = False
        slot["last_gray_thumb"] = None
        slot["current_frame_pos"] = 0
        slot["cap_next_pos"] = 0
        slot["_rt_base_time"] = time.time()
        slot["_rt_base_frame"] = 0
        slot["_raw_seq"] = 0
        slot["_latest_raw_frame"] = None
        slot["current_frame"] = None
        slot["frame_seq"] = slot.get("frame_seq", 0) + 1
        slot["last_display_seq"] = -1
        slot["last_ai_seq"] = -1
        slot["last_ai_time"] = 0.0
        slot["cached_detections"] = []
        slot["cached_spotted"] = False

    def _effective_slot_speed(self, slot):
        return (slot["speed"] or self.current_speed).strip()

    def _refresh_active_flags_locked(self):
        active_files = [
            s for s in self.slots
            if s["enabled"] and s["is_file"] and s.get("cap") is not None and not s.get("ended", False)
        ]
        self.has_file_slots = len(active_files) > 0
        self.has_realtime_file_slots = any(self._effective_slot_speed(s) == "1x" for s in active_files)
        self.has_fast_file_slots = any(self._effective_slot_speed(s) != "1x" for s in active_files)
        self.has_live_slots = any(
            s["enabled"] and not s["is_file"] and s.get("reader") is not None
            for s in self.slots
        )
        self.max_video_duration = max(
            (s["duration_sec"] for s in self.slots if s["enabled"] and s["is_file"]),
            default=0.0,
        )

    def _fast_owns_clock_locked(self):
        return self.has_fast_file_slots and self.current_speed != "1x"

    def _reanchor_realtime_slots_locked(self, now=None):
        now = time.time() if now is None else now
        for s in self.slots:
            s["_rt_base_time"] = now
            s["_rt_base_frame"] = s["current_frame_pos"]

    def _ensure_loops_running(self):
        if not self.is_running:
            return
        if (self.has_live_slots or self.has_realtime_file_slots) and (
            self.live_ai_thread is None or not self.live_ai_thread.is_alive()
        ):
            self.live_ai_thread = threading.Thread(
                target=self._live_ai_loop,
                daemon=True,
            )
            self.live_ai_thread.start()
        if self.has_realtime_file_slots and (
            self.realtime_thread is None or not self.realtime_thread.is_alive()
        ):
            self.realtime_thread = threading.Thread(
                target=self._realtime_file_loop,
                daemon=True,
            )
            self.realtime_thread.start()
        if self.has_fast_file_slots and (
            self.worker_thread is None or not self.worker_thread.is_alive()
        ):
            self.worker_thread = threading.Thread(
                target=self._parallel_scan_loop,
                args=(self._on_finished_callback,),
                daemon=True,
            )
            self.worker_thread.start()

    def register_custom_source(self, source_key: str, factory):
        """Registers a virtual VideoCapture-compatible factory (e.g. browser frame stream)."""
        with self._lock:
            self._custom_sources[str(source_key).strip()] = factory

    def unregister_custom_source(self, source_key: str):
        """Removes a registered virtual VideoCapture factory."""
        with self._lock:
            self._custom_sources.pop(str(source_key).strip(), None)

    def _open_source_for_slot(self, idx, raw_src):
        """Parses and opens a file, webcam, or live URL for a slot, raising ConnectionError on failure."""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|rtsp_transport;tcp"
        norm_src = self.normalize_stream_source(raw_src)

        with self._lock:
            custom_factory = self._custom_sources.get(norm_src)

        if custom_factory is not None:
            cap = custom_factory()
            if cap is not None and cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / max(1.0, fps)
                return norm_src, True, cap, None, fps, total_frames, duration_sec
            raise ConnectionError(
                f"Instant browser video stream for Camera {idx+1} is not ready. Please re-select the video."
            )

        if norm_src == "" or norm_src == "0":
            src = 0
            is_file = False
        else:
            try:
                src = int(norm_src)
                is_file = False
            except ValueError:
                src = norm_src
                is_file = os.path.isfile(norm_src)

        if is_file:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / max(1.0, fps)
                return norm_src, True, cap, None, fps, total_frames, duration_sec
        else:
            src_lower = str(src).lower()
            is_live_stream = isinstance(src, int) or src_lower.startswith("http://") or src_lower.startswith("https://") or src_lower.startswith("rtsp://")
            if is_live_stream:
                reader = _LiveStreamReader(src)
                if reader.open(timeout=4.5):
                    return norm_src, False, None, reader, reader.fps, 0, 0.0
                reader.stop()

        if isinstance(src, int):
            hint = (
                f"Could not open Webcam (index {src}) on Camera {idx+1}.\n\n"
                "Quick Fixes:\n"
                "• Close any app using your camera (Zoom, Teams, Meet, or Windows Camera app).\n"
                "• Check Windows Settings → Privacy & Security → Camera (Allow apps access).\n"
                "• If you have an external or IR camera, try index '1'."
            )
        elif str(src).startswith("http://") or str(src).startswith("https://") or str(src).startswith("rtsp://"):
            hint = (
                f"Could not connect to Phone / IP Camera {idx+1}:\n{raw_src}\n\n"
                "Quick Fixes:\n"
                "1. Make sure you tapped 'Start server' inside the phone app.\n"
                "2. Ensure laptop & phone are on the SAME Wi-Fi (or turn on Phone Hotspot).\n"
                f"3. Test in Chrome: Open '{raw_src}' in your browser to verify it connects."
            )
        else:
            hint = f"Could not read video file for Camera {idx+1}:\n{raw_src}\nPlease verify the file exists."
        raise ConnectionError(hint)

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
        self.max_video_duration = 0.0
        self._on_finished_callback = on_finished_callback

        for i, slot in enumerate(self.slots):
            self._reset_slot_runtime_fields(slot)
            slot["frame_seq"] = 0
            slot["reader"] = None
            slot["cap"] = None

            if slot["enabled"]:
                try:
                    norm_src, is_file, cap, reader, fps, total_frames, duration_sec = self._open_source_for_slot(i, slot["source"])
                except ConnectionError:
                    self.stop()
                    raise

                slot["source"] = norm_src
                slot["is_file"] = is_file
                slot["cap"] = cap
                slot["reader"] = reader
                slot["fps"] = fps
                slot["total_frames"] = total_frames
                slot["duration_sec"] = duration_sec

                # For live streams (phone / webcam / IP stream), lock playback speed to 1x
                if not is_file and slot["speed"] in ["5x", "10x", "20x", "40x"]:
                    slot["speed"] = "1x"

                active_count += 1

        if active_count == 0:
            raise ValueError("Please enable at least one Camera slot!")

        with self._lock:
            now = time.time()
            self.current_speed = str(global_speed).strip()
            self.is_running = True
            self.is_paused = False
            self.global_clock_sec = 0.0
            self.start_clock_sec = 0.0
            self.start_real_time = now
            self.actual_speed_multiplier = 1.0
            self.current_steps_per_sec = 0.0
            self._reanchor_realtime_slots_locked(now)
            self._refresh_active_flags_locked()

        self._ensure_loops_running()
        return active_count

    def _sync_live_clock_locked(self):
        """Advances global_clock_sec from wall clock when running at 1x (live or 1x file playback)."""
        if (
            self.is_running
            and not self.is_paused
            and not self._fast_owns_clock_locked()
            and (self.has_live_slots or self.has_realtime_file_slots)
        ):
            self.global_clock_sec = self.start_clock_sec + max(0.0, time.time() - self.start_real_time)

    def toggle_pause(self):
        """Toggles pause/resume state. Returns True if now paused, False if resumed."""
        with self._lock:
            if not self.is_paused:
                self._sync_live_clock_locked()
            self.is_paused = not self.is_paused
            if not self.is_paused:
                now = time.time()
                self.start_real_time = now
                self.start_clock_sec = self.global_clock_sec
                self._reanchor_realtime_slots_locked(now)
            return self.is_paused

    def pause(self):
        with self._lock:
            if not self.is_paused:
                self._sync_live_clock_locked()
            self.is_paused = True

    def resume(self):
        with self._lock:
            now = time.time()
            self.is_paused = False
            self.start_real_time = now
            self.start_clock_sec = self.global_clock_sec
            self._reanchor_realtime_slots_locked(now)

    def set_global_speed(self, new_speed):
        """Dynamically adjusts search speed mid-playback without restarting feeds."""
        with self._lock:
            self._sync_live_clock_locked()
            now = time.time()
            self.current_speed = str(new_speed).strip()
            self.start_real_time = now
            self.start_clock_sec = self.global_clock_sec
            for slot in self.slots:
                if slot["is_file"] or not self.is_running:
                    if not (self.is_running and not slot["is_file"]):
                        slot["speed"] = self.current_speed
            self._reanchor_realtime_slots_locked(now)
            self._refresh_active_flags_locked()
            running = self.is_running
        if running:
            self._ensure_loops_running()

    def set_slot_speed(self, idx, new_speed):
        """Dynamically adjusts one camera's speed; live streams stay locked at 1x."""
        if not (0 <= idx < len(self.slots)):
            return False, "1x"
        spd = str(new_speed).strip()
        with self._lock:
            slot = self.slots[idx]
            is_live = (not slot["is_file"]) if self.is_running else self.is_live_source(slot["source"])
            if is_live and spd != "1x":
                slot["speed"] = "1x"
                return False, "1x"
            self._sync_live_clock_locked()
            now = time.time()
            slot["speed"] = spd
            slot["_rt_base_time"] = now
            slot["_rt_base_frame"] = slot["current_frame_pos"]
            self.start_real_time = now
            self.start_clock_sec = self.global_clock_sec
            self._refresh_active_flags_locked()
            running = self.is_running
        if running:
            self._ensure_loops_running()
        return True, spd

    def set_slot_location(self, idx, location):
        """Updates a camera slot's display/tracker location name live."""
        if 0 <= idx < len(self.slots):
            with self._lock:
                self.slots[idx]["location"] = str(location).strip()

    def remove_slot(self, idx, clear_source=True):
        """
        Safely disables/removes a single camera slot (even mid-scan) without stopping other cameras.
        Returns count of remaining enabled cameras; stops engine automatically if 0 remain.
        """
        if not (0 <= idx < len(self.slots)):
            return 0
        slot = self.slots[idx]
        with self._lock:
            slot["enabled"] = False
            if clear_source:
                slot["source"] = ""
                slot["is_file"] = False
            slot["total_frames"] = 0
            slot["duration_sec"] = 0.0
            self._reset_slot_runtime_fields(slot)

        self._close_slot_media(slot)

        with self._lock:
            self._refresh_active_flags_locked()
            remaining = sum(
                1 for s in self.slots
                if s["enabled"] and (not self.is_running or s.get("cap") is not None or s.get("reader") is not None)
            )
            running_now = self.is_running

        if running_now and remaining == 0:
            self.stop()
        return remaining

    def replace_slot(self, idx, source, location=None, speed=None):
        """Hot-swaps or enables a single camera slot with a new video file or stream mid-run."""
        if not (0 <= idx < len(self.slots)):
            return False
        slot = self.slots[idx]
        raw_src = str(source).strip()

        if not self.is_running:
            self.configure_slot(idx, enabled=True, source=raw_src, location=location, speed=speed)
            with self._lock:
                self._reset_slot_runtime_fields(slot)
            return True

        # Disable slot briefly during close -> reopen so worker loops skip it cleanly
        with self._lock:
            slot["enabled"] = False
            self._reset_slot_runtime_fields(slot)
            self._refresh_active_flags_locked()

        self._close_slot_media(slot)

        try:
            norm_src, is_file, cap, reader, fps, total_frames, duration_sec = self._open_source_for_slot(idx, raw_src)
        except ConnectionError:
            with self._lock:
                self._refresh_active_flags_locked()
            raise

        with self._lock:
            slot["source"] = norm_src
            if location is not None:
                slot["location"] = str(location).strip()
            chosen_speed = str(speed).strip() if speed is not None else slot["speed"]
            slot["speed"] = "1x" if not is_file else (chosen_speed or self.current_speed)
            slot["is_file"] = is_file
            slot["cap"] = cap
            slot["reader"] = reader
            slot["fps"] = fps
            slot["total_frames"] = total_frames
            slot["duration_sec"] = duration_sec
            self._reset_slot_runtime_fields(slot)
            slot["enabled"] = True
            self._refresh_active_flags_locked()

        self._ensure_loops_running()
        return True

    def stop(self):
        """Halts the worker threads safely and releases camera resources without race conditions."""
        self.is_running = False
        self.is_paused = False
        self.has_live_slots = False
        self.has_file_slots = False
        self.has_realtime_file_slots = False
        self.has_fast_file_slots = False

        for attr in ("worker_thread", "realtime_thread", "live_ai_thread"):
            th = getattr(self, attr, None)
            if th is not None and th.is_alive():
                try:
                    th.join(timeout=1.5)
                except Exception:
                    pass
                setattr(self, attr, None)

        for slot in self.slots:
            self._close_slot_media(slot)
            with self._lock:
                slot["spotted"] = False
                slot["cached_spotted"] = False
                slot["cached_detections"] = []
                slot["current_frame"] = None

    def get_slot_state(self, idx, include_frame=True):
        """Returns thread-safe snapshot of slot state for GUI rendering with per-camera progress."""
        if not (0 <= idx < len(self.slots)):
            return None

        slot = self.slots[idx]
        reader = slot.get("reader")

        # Decoupled live rendering: overlay latest AI boxes onto every fresh raw camera frame (~30 FPS)
        if include_frame and slot["enabled"] and not slot["is_file"] and reader is not None and self.is_running and not self.is_paused:
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
                preview_raw = _cap_height_480(raw_frame)
                annotated = self.scanner.draw_overlays(
                    preview_raw,
                    cached_dets,
                    cached_spotted,
                    camera_name=loc_name,
                    video_time_str=cam_time_str,
                )
                with self._lock:
                    if slot["enabled"]:
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

            # Only copy image buffer when caller actually needs the video frame
            frame_copy = (
                slot["current_frame"].copy()
                if (include_frame and slot["current_frame"] is not None)
                else None
            )

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
        elif real_elapsed < 0.25:
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

    @staticmethod
    def _format_hms(sec_float):
        cur_sec = max(0, int(sec_float))
        return f"{cur_sec // 3600:02d}:{(cur_sec % 3600) // 60:02d}:{cur_sec % 60:02d}"

    @staticmethod
    def _read_slot_frame_at(slot, target_pos):
        """
        Reads target_pos using zero-seek sequential cap.read()/cap.grab() when advancing
        by 1-4 frames, falling back to cap.set() only for larger fast-scan jumps.
        Must be called under slot['op_lock'].
        """
        cap = slot.get("cap")
        if cap is None or not cap.isOpened():
            return False, None

        next_pos = slot.get("cap_next_pos", -999999)
        delta = target_pos - next_pos
        if delta == 0:
            ret, frame = cap.read()
        elif 1 <= delta <= 4:
            ok = True
            for _ in range(delta):
                if not cap.grab():
                    ok = False
                    break
            ret, frame = cap.read() if ok else (False, None)
        else:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            except Exception:
                pass
            ret, frame = cap.read()

        if ret and frame is not None:
            slot["cap_next_pos"] = target_pos + 1
            slot["current_frame_pos"] = target_pos
        return ret, frame

    @staticmethod
    def _should_run_motion_ai(slot, frame, min_pixels=30, heartbeat_sec=1.5):
        """Lightweight motion gate with periodic heartbeat so still targets are still verified."""
        now = time.time()
        should_run = True
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
                if (
                    motion_pixels < min_pixels
                    and not slot.get("spotted", False)
                    and (now - slot.get("last_ai_time", 0.0)) < heartbeat_sec
                ):
                    should_run = False
            slot["last_gray_thumb"] = small_gray
        except Exception:
            pass
        return should_run

    def _record_slot_sighting(self, slot, loc_name, cam_time_str, match_info):
        self.last_spotlight_idx = slot["id"]
        self.tracker.log_sighting(
            camera_name=loc_name,
            confidence=match_info.get("confidence", 0.8),
            snapshot_path=match_info.get("snapshot_path", ""),
            video_time=cam_time_str,
            camera_id=slot["id"],
        )

    def _check_and_finish_if_all_ended(self):
        """Stops the engine and fires on_finished_callback only once all live and file feeds finish."""
        callback = None
        with self._lock:
            if not self.is_running:
                return
            self._refresh_active_flags_locked()
            if self.has_live_slots or self.has_file_slots:
                self.start_clock_sec = self.global_clock_sec
                self.start_real_time = time.time()
                return
            self.is_running = False
            if self.max_video_duration > 0:
                self.global_clock_sec = min(self.max_video_duration, self.global_clock_sec)
            callback = self._on_finished_callback
        if callback:
            try:
                callback()
            except Exception:
                pass

    def _realtime_file_loop(self):
        """
        True 1.0x Wall-Clock File Playback Loop:
        Reads 1x video files sequentially without cap.set() seek overhead and decouples
        30 FPS preview rendering from async batched AI so CPU never slows 1x down to 0.4x.
        """
        step_counter = 0
        rate_timer = time.time()

        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.04)
                    continue

                with self._lock:
                    cur_speed = self.current_speed
                    rt_slots = [
                        s for s in self.slots
                        if s["enabled"]
                        and s["is_file"]
                        and s.get("cap") is not None
                        and not s.get("ended", False)
                        and (s["speed"] or cur_speed) == "1x"
                    ]

                if not rt_slots:
                    self._check_and_finish_if_all_ended()
                    with self._lock:
                        if not (
                            self.is_running
                            and any(
                                s["enabled"] and s["is_file"] and s.get("cap") is not None
                                and not s.get("ended", False)
                                and (s["speed"] or self.current_speed) == "1x"
                                for s in self.slots
                            )
                        ):
                            break
                    continue

                now = time.time()
                min_wait = 0.020
                did_step = False

                for slot in rt_slots:
                    if not self.is_running or self.is_paused:
                        break
                    with slot["op_lock"]:
                        if not slot["enabled"] or not slot["is_file"] or slot.get("ended", False):
                            continue
                        fps = max(1.0, slot["fps"])
                        first_frame = (slot.get("_raw_seq", 0) == 0)
                        if first_frame:
                            slot["_rt_base_time"] = now
                            slot["_rt_base_frame"] = slot["current_frame_pos"]
                            target_pos = slot["current_frame_pos"]
                        else:
                            elapsed_rt = max(0.0, now - slot.get("_rt_base_time", now))
                            target_pos = int(slot.get("_rt_base_frame", 0) + elapsed_rt * fps)

                        tot = slot["total_frames"]
                        if tot > 0 and target_pos >= tot:
                            slot["ended"] = True
                            continue

                        if not first_frame and target_pos <= slot["current_frame_pos"]:
                            # Calculate sleep until next frame boundary on this video
                            next_frame_t = (
                                slot.get("_rt_base_time", now)
                                + ((slot["current_frame_pos"] + 1) - slot.get("_rt_base_frame", 0)) / fps
                            )
                            min_wait = min(min_wait, max(0.004, next_frame_t - now))
                            continue

                        ret, frame = self._read_slot_frame_at(slot, target_pos)
                        if not ret or frame is None:
                            slot["ended"] = True
                            continue

                    did_step = True
                    with self._lock:
                        cached_dets = list(slot["cached_detections"])
                        cached_spotted = slot["cached_spotted"]
                        loc_name = slot["location"].strip() or f"Camera {slot['id'] + 1}"
                        cam_time_str = self._format_hms(target_pos / max(1.0, slot["fps"]))
                        slot["_latest_raw_frame"] = frame
                        slot["_raw_seq"] = slot.get("_raw_seq", 0) + 1

                    preview = _cap_height_480(frame)
                    annotated = self.scanner.draw_overlays(
                        preview,
                        cached_dets,
                        cached_spotted,
                        camera_name=loc_name,
                        video_time_str=cam_time_str,
                    )

                    with self._lock:
                        if slot["enabled"] and not slot["ended"]:
                            slot["current_frame"] = annotated
                            slot["spotted"] = cached_spotted
                            slot["frame_seq"] = slot.get("frame_seq", 0) + 1

                if did_step:
                    step_counter += 1
                    with self._lock:
                        self._sync_live_clock_locked()
                else:
                    with self._lock:
                        self._sync_live_clock_locked()

                elapsed_rate = time.time() - rate_timer
                if elapsed_rate >= 1.0:
                    with self._lock:
                        if not self._fast_owns_clock_locked():
                            self.current_steps_per_sec = step_counter / elapsed_rate
                    step_counter = 0
                    rate_timer = time.time()

                # Check if any 1x slot just reached EOF
                with self._lock:
                    rem_rt = any(
                        s["enabled"] and s["is_file"] and not s.get("ended", False)
                        and (s["speed"] or self.current_speed) == "1x"
                        for s in self.slots
                    )
                if not rem_rt:
                    self._check_and_finish_if_all_ended()
                    break

                time.sleep(max(0.004, min(0.015, min_wait)))
            except Exception:
                if not self.is_running:
                    break
                time.sleep(0.01)

    def _live_ai_loop(self):
        """
        Async Batched AI Worker for Live Streams & 1x Recorded Files:
        Runs single-pass batched 320px YOLOv8 + Motion Gating without stalling 30 FPS display.
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
                    global_hms = self._format_hms(self.global_clock_sec)
                    cur_speed = self.current_speed
                    async_slots = [
                        s for s in self.slots
                        if s["enabled"] and (
                            (not s["is_file"] and s.get("reader") is not None)
                            or (
                                s["is_file"]
                                and s.get("cap") is not None
                                and not s.get("ended", False)
                                and (s["speed"] or cur_speed) == "1x"
                            )
                        )
                    ]

                if not async_slots:
                    break

                batch_entries = []  # (slot, frame, raw_seq, loc_name, cam_time_str)
                did_work = False

                for slot in async_slots:
                    if not self.is_running or self.is_paused:
                        break
                    if not slot["is_file"]:
                        reader = slot.get("reader")
                        if reader is None:
                            continue
                        frame, raw_seq, _ = reader.get_latest()
                        cam_time_str = global_hms
                    else:
                        with self._lock:
                            frame = slot.get("_latest_raw_frame")
                            raw_seq = slot.get("_raw_seq", 0)
                            cam_pos = slot["current_frame_pos"]
                            cam_fps = max(1.0, slot["fps"])
                        cam_time_str = self._format_hms(cam_pos / cam_fps)

                    if frame is None or raw_seq == 0 or raw_seq == slot["last_ai_seq"]:
                        continue

                    did_work = True
                    slot["last_ai_seq"] = raw_seq
                    if not self._should_run_motion_ai(slot, frame, min_pixels=25, heartbeat_sec=0.8):
                        continue

                    loc_name = slot["location"].strip() or f"Camera {slot['id'] + 1}"
                    batch_entries.append((slot, frame, raw_seq, loc_name, cam_time_str))

                if batch_entries and self.is_running and not self.is_paused:
                    ai_items = [(fr, loc, cts, True) for (_, fr, _, loc, cts) in batch_entries]
                    batch_out = self.scanner.analyze_batch(ai_items)
                    ai_now = time.time()

                    for (slot, raw_frame, raw_seq, loc_name, cam_time_str), (detections, spotted, match_info) in zip(
                        batch_entries, batch_out
                    ):
                        with self._lock:
                            if not slot["enabled"]:
                                continue
                            slot["cached_detections"] = detections
                            slot["cached_spotted"] = spotted
                            slot["spotted"] = spotted
                            slot["last_ai_time"] = ai_now
                            if not slot["is_file"]:
                                if slot["last_display_seq"] == raw_seq:
                                    slot["last_display_seq"] = -1
                            else:
                                # Refresh current frame overlay immediately if video is paused on a low FPS cadence
                                preview = _cap_height_480(raw_frame)
                                slot["current_frame"] = self.scanner.draw_overlays(
                                    preview, detections, spotted, camera_name=loc_name, video_time_str=cam_time_str
                                )
                                slot["frame_seq"] = slot.get("frame_seq", 0) + 1

                        if spotted and match_info:
                            self._record_slot_sighting(slot, loc_name, cam_time_str, match_info)

                    live_sweeps += 1
                    time.sleep(0.005)

                elapsed = time.time() - rate_timer
                if elapsed >= 1.0:
                    with self._lock:
                        if self.has_live_slots and not self.has_file_slots:
                            self.current_steps_per_sec = live_sweeps / elapsed
                    live_sweeps = 0
                    rate_timer = time.time()

                if not did_work or not batch_entries:
                    time.sleep(0.008)
            except Exception:
                if not self.is_running:
                    break
                time.sleep(0.02)

    def _grab_fast_slot_frame(self, slot, cur_speed):
        """Worker task: decodes next fast-scan frame for one slot and evaluates motion gate."""
        try:
            with slot["op_lock"]:
                cap = slot.get("cap")
                if cap is None or not cap.isOpened() or not slot["enabled"] or not self.is_running:
                    return None
                pos = slot["current_frame_pos"]
                if slot["total_frames"] > 0 and pos >= slot["total_frames"]:
                    slot["ended"] = True
                    return {"slot": slot, "ended": True}
                ret, frame = self._read_slot_frame_at(slot, pos)
                if not ret or frame is None:
                    slot["ended"] = True
                    return {"slot": slot, "ended": True}

            if not slot["enabled"] or not self.is_running:
                return None

            cam_time_str = self._format_hms(pos / max(1.0, slot["fps"]))
            loc_name = slot["location"].strip() or f"Camera {slot['id'] + 1}"
            should_ai = self._should_run_motion_ai(slot, frame, min_pixels=10, heartbeat_sec=0.5)
            return {
                "slot": slot,
                "frame": frame,
                "loc_name": loc_name,
                "cam_time_str": cam_time_str,
                "should_ai": should_ai,
                "ended": False,
            }
        except Exception:
            return {"slot": slot, "ended": False, "frame": None}

    def _parallel_scan_loop(self, on_finished_callback):
        """
        Batched + Wall-Clock Paced Fast-Scan Worker Loop (5x, 10x, 20x, 40x):
        Decodes frames in parallel, runs a single batched YOLO pass across all active
        fast cameras (~4x faster on CPU), and scales frame jumps dynamically so 40x
        runs near true 40x speed on both fast GPUs and multi-camera CPUs.
        """
        step_counter = 0
        rate_timer = time.time()
        min_step_sec = 0.065

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while self.is_running:
                try:
                    if self.is_paused:
                        time.sleep(0.04)
                        continue

                    step_start = time.time()
                    with self._lock:
                        cur_speed = self.current_speed
                        active_slots = [
                            s for s in self.slots
                            if s["enabled"]
                            and s["is_file"]
                            and s.get("cap") is not None
                            and not s.get("ended", False)
                            and (s["speed"] or cur_speed) != "1x"
                        ]

                    if not active_slots or not self.is_running:
                        self._check_and_finish_if_all_ended()
                        with self._lock:
                            if not (
                                self.is_running
                                and any(
                                    s["enabled"] and s["is_file"] and s.get("cap") is not None
                                    and not s.get("ended", False)
                                    and (s["speed"] or self.current_speed) != "1x"
                                    for s in self.slots
                                )
                            ):
                                break
                        continue

                    # Phase 1: Parallel capture read + motion gate
                    grab_futures = [executor.submit(self._grab_fast_slot_frame, s, cur_speed) for s in active_slots]
                    grabbed = []
                    for f in concurrent.futures.as_completed(grab_futures):
                        res = f.result()
                        if res is not None:
                            grabbed.append(res)

                    ai_targets = [g for g in grabbed if not g.get("ended") and g.get("frame") is not None and g.get("should_ai")]
                    all_ended = all(g.get("ended", False) for g in grabbed) if grabbed else True

                    # Phase 2: Single batched YOLO inference + parallel Re-ID across active cameras
                    ai_results = {}
                    if ai_targets and self.is_running:
                        ai_items = [(g["frame"], g["loc_name"], g["cam_time_str"], True) for g in ai_targets]
                        batch_out = self.scanner.analyze_batch(ai_items, executor=executor)
                        now_ai = time.time()
                        for g, out in zip(ai_targets, batch_out):
                            ai_results[g["slot"]["id"]] = out
                            g["slot"]["last_ai_time"] = now_ai

                    # Phase 3: Render lightweight 480p overlays in parallel
                    def _publish_fast_slot(g):
                        slot = g["slot"]
                        frame = g.get("frame")
                        if frame is None or not slot["enabled"] or not self.is_running:
                            return
                        preview = _cap_height_480(frame)
                        out = ai_results.get(slot["id"])
                        if out is not None:
                            detections, spotted, match_info = out
                            annotated = self.scanner.draw_overlays(
                                preview, detections, spotted, camera_name=g["loc_name"], video_time_str=g["cam_time_str"]
                            )
                            with self._lock:
                                slot["spotted"] = spotted
                                slot["cached_detections"] = detections
                                slot["cached_spotted"] = spotted
                                slot["current_frame"] = annotated
                                slot["frame_seq"] = slot.get("frame_seq", 0) + 1
                            if spotted and match_info:
                                self._record_slot_sighting(slot, g["loc_name"], g["cam_time_str"], match_info)
                        else:
                            # Fast path for static frames skipped by motion gate: keep cached detections drawn cleanly
                            with self._lock:
                                cached_dets = list(slot["cached_detections"])
                                cached_spotted = slot["cached_spotted"]
                            annotated = self.scanner.draw_overlays(
                                preview, cached_dets, cached_spotted, camera_name=g["loc_name"], video_time_str=g["cam_time_str"]
                            )
                            with self._lock:
                                slot["spotted"] = cached_spotted
                                slot["current_frame"] = annotated
                                slot["frame_seq"] = slot.get("frame_seq", 0) + 1

                    pub_targets = [g for g in grabbed if not g.get("ended") and g.get("frame") is not None]
                    if len(pub_targets) > 1:
                        list(executor.map(_publish_fast_slot, pub_targets))
                    elif pub_targets:
                        _publish_fast_slot(pub_targets[0])

                    if all_ended:
                        self._check_and_finish_if_all_ended()
                        break

                    # Phase 4: Wall-clock pacing & adaptive jump for NEXT step
                    work_elapsed = time.time() - step_start
                    if work_elapsed < min_step_sec and self.is_running and not self.is_paused:
                        time.sleep(min_step_sec - work_elapsed)

                    dt = max(0.015, time.time() - step_start)
                    if not self.is_paused and self.is_running:
                        with self._lock:
                            for g in pub_targets:
                                slot = g["slot"]
                                if not slot["enabled"] or slot.get("ended", False):
                                    continue
                                slot_spd = slot["speed"] or self.current_speed
                                mult = self.speed_multipliers.get(slot_spd, 10.0)
                                jump_frames = max(1, int(round(dt * mult * max(1.0, slot["fps"]))))
                                next_pos = slot["current_frame_pos"] + jump_frames
                                if slot["total_frames"] > 0 and next_pos >= slot["total_frames"]:
                                    slot["current_frame_pos"] = slot["total_frames"]
                                    slot["ended"] = True
                                else:
                                    slot["current_frame_pos"] = next_pos

                            if self._fast_owns_clock_locked():
                                global_mult = self.speed_multipliers.get(self.current_speed, 10.0)
                                self.global_clock_sec += dt * global_mult
                                if self.max_video_duration > 0:
                                    self.global_clock_sec = min(self.max_video_duration, self.global_clock_sec)

                    step_counter += 1
                    elapsed_rate = time.time() - rate_timer
                    if elapsed_rate >= 1.0:
                        with self._lock:
                            self.current_steps_per_sec = step_counter / elapsed_rate
                        step_counter = 0
                        rate_timer = time.time()

                    with self._lock:
                        still_fast = any(
                            s["enabled"] and s["is_file"] and not s.get("ended", False)
                            and (s["speed"] or self.current_speed) != "1x"
                            for s in self.slots
                        )
                    if not still_fast:
                        self._check_and_finish_if_all_ended()
                        break
                except Exception:
                    if not self.is_running:
                        break
                    time.sleep(0.01)
