"""
Smart Supervise — Responsive Cross-Platform Web Application (Mobile, Tablet & Desktop)
Shares core/ AI, VideoEngine, LocationTracker, and MapVisualizer directly with app.py.
Deploys seamlessly on Hugging Face Spaces (Docker port 7860), local Wi-Fi, or any cloud host.
"""

import os
import sys
import cv2
import json
import time
import uuid
import string
import socket
import base64
import asyncio
import threading
import numpy as np
from contextlib import asynccontextmanager

# Prevent OpenCV internal thread thrashing alongside VideoEngine workers
cv2.setNumThreads(1)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.responses import (
    JSONResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
    FileResponse,
)
import uvicorn

from core.cctv_scanner import CCTVScanner
from core.location_tracker import LocationTracker
from core.map_visualizer import MapVisualizer
from core.video_engine import VideoEngine


WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def _sanitize_filename(filename, default_ext=".bin"):
    base = os.path.basename(str(filename or "upload")).strip()
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    if not safe:
        safe = f"file{default_ext}"
    return f"{uuid.uuid4().hex[:8]}_{safe}"


def _make_placeholder_jpeg(title, subtitle="", accent=(59, 110, 246)):
    img = np.zeros((270, 480, 3), dtype=np.uint8)
    img[:] = (22, 23, 45)
    cv2.rectangle(img, (8, 8), (472, 262), accent, 1)
    cv2.putText(
        img, title, (28, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 238, 255), 2, cv2.LINE_AA
    )
    if subtitle:
        cv2.putText(
            img, subtitle[:46], (28, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 155, 190), 1, cv2.LINE_AA
        )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else b""


_STANDBY_JPEG = _make_placeholder_jpeg("SMART SUPERVISE", "Feed in standby — configure camera and press Start")
_DISABLED_JPEG = _make_placeholder_jpeg("CAMERA OFF", "Enable or upload video to monitor this feed", (90, 95, 130))


class _BrowserFrameCapture:
    """
    OpenCV VideoCapture-compatible capture backed by on-demand browser <video> decoding.
    Avoids uploading multi-GB video files upfront: the browser keeps the file local and
    streams only the lightweight ~480p JPEG frames needed at 1x-40x scan speeds.
    """

    def __init__(self, slot_id: int, filename: str, duration_sec: float, fps: float = 15.0, on_demand=None):
        self.slot_id = int(slot_id)
        self.filename = str(filename or f"camera_{slot_id + 1}.mp4")
        self.duration_sec = max(0.5, float(duration_sec))
        self.fps = max(5.0, min(30.0, float(fps)))
        self.total_frames = max(1, int(round(self.duration_sec * self.fps)))
        self._on_demand = on_demand

        self._cond = threading.Condition()
        self._opened = True
        self._released = False
        self._eof = False

        self.cursor_pos = 0
        self.demand_seq = 0
        self.demand_sec = 0.0
        self._delivered_seq = -1
        self._delivered_frame = None

    def isOpened(self):
        return bool(self._opened and not self._released)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.cursor_pos)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return float((self.cursor_pos / max(1.0, self.fps)) * 1000.0)
        return 0.0

    def set(self, prop_id, value):
        if prop_id != cv2.CAP_PROP_POS_FRAMES:
            return False
        notify_args = None
        with self._cond:
            if self._released or not self._opened:
                return False
            target = max(0, int(round(value)))
            self.cursor_pos = target
            if target >= self.total_frames:
                self._eof = True
                self._cond.notify_all()
                return True
            self._eof = False
            self.demand_sec = min(max(0.0, self.duration_sec - 0.02), target / max(1.0, self.fps))
            self.demand_seq += 1
            notify_args = (self.slot_id, self.demand_seq, self.demand_sec, self.fps)
            self._cond.notify_all()
        if notify_args and self._on_demand:
            self._on_demand(*notify_args)
        return True

    def grab(self):
        with self._cond:
            if self._released or not self._opened or self._eof:
                return False
            self.cursor_pos += 1
            if self.cursor_pos >= self.total_frames:
                self._eof = True
                self._cond.notify_all()
                return False
            return True

    def read(self):
        notify_args = None
        with self._cond:
            if self._released or not self._opened or self._eof or self.cursor_pos >= self.total_frames:
                return False, None
            if self.demand_seq == 0 or self._delivered_seq >= self.demand_seq:
                self.demand_sec = min(max(0.0, self.duration_sec - 0.02), self.cursor_pos / max(1.0, self.fps))
                self.demand_seq += 1
                notify_args = (self.slot_id, self.demand_seq, self.demand_sec, self.fps)
            target_seq = self.demand_seq

        if notify_args and self._on_demand:
            self._on_demand(*notify_args)

        deadline = time.time() + 10.0
        with self._cond:
            while (
                not self._released
                and self._opened
                and not self._eof
                and self._delivered_seq < target_seq
            ):
                rem = deadline - time.time()
                if rem <= 0:
                    return False, None
                self._cond.wait(timeout=min(0.20, rem))

            if self._released or not self._opened or self._eof or self._delivered_frame is None:
                return False, None
            frame = self._delivered_frame
            self.cursor_pos += 1
            return True, frame

    def deliver_jpeg(self, seq: int, raw_bytes: bytes) -> bool:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        with self._cond:
            if self._released:
                return False
            if int(seq) >= self.demand_seq or self.demand_seq == 0:
                self._delivered_frame = img
                self._delivered_seq = max(self._delivered_seq, int(seq))
                self._cond.notify_all()
            return True

    def signal_eof(self):
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def pending_demand(self):
        with self._cond:
            if (
                not self._released
                and self._opened
                and not self._eof
                and self.demand_seq > self._delivered_seq
                and self.demand_seq > 0
            ):
                return {"slot_id": self.slot_id, "seq": self.demand_seq, "sec": self.demand_sec, "fps": self.fps}
            return None

    def release(self):
        with self._cond:
            self._released = True
            self._opened = False
            self._cond.notify_all()


class WebAppState:
    """Thread-safe coordinator binding Starlette HTTP/MJPEG handlers to core engines."""

    def __init__(self):
        self.scanner = CCTVScanner()
        self.tracker = LocationTracker(lost_timeout_seconds=30)
        self.map_visualizer = MapVisualizer()
        self.video_engine = VideoEngine(self.scanner, self.tracker)

        self._ui_lock = threading.RLock()
        self._ws_lock = threading.Lock()
        self._ws_clients = set()
        self._loop = None
        self._browser_meta = {}       # slot_id -> {"filename", "duration_sec", "fps"}
        self._browser_captures = {}   # slot_id -> latest _BrowserFrameCapture

        self.tracking_mode = "person"
        self.target_name = "Target Person"
        self.class_name = "dog"
        self.sensitivity_key = "strict"  # strict (0.70) | balanced (0.60) | flexible (0.50)
        self.global_speed = "10x"
        self.status_text = "IDLE — Upload Target & Configure Cameras"
        self.status_level = "dim"  # dim | green | amber | red

    def set_status(self, text, level="dim"):
        with self._ui_lock:
            self.status_text = str(text)
            self.status_level = level

    def on_scan_finished(self):
        self.set_status("SEARCH FINISHED — ALL FEEDS COMPLETE", "green")

    def register_browser_stream(self, slot_id: int, filename: str, duration_sec: float, fps: float = 15.0):
        slot_id = int(slot_id)
        meta = {
            "filename": str(filename or f"camera_{slot_id + 1}.mp4"),
            "duration_sec": max(0.5, float(duration_sec)),
            "fps": max(5.0, min(30.0, float(fps))),
        }
        src_key = f"browser://{slot_id}"
        with self._ui_lock:
            old_cap = self._browser_captures.pop(slot_id, None)
            if old_cap is not None:
                old_cap.release()
            self._browser_meta[slot_id] = meta

        def _opener():
            m = self._browser_meta.get(slot_id, meta)
            cap = _BrowserFrameCapture(
                slot_id,
                m["filename"],
                m["duration_sec"],
                m["fps"],
                on_demand=self.notify_browser_demand,
            )
            with self._ui_lock:
                self._browser_captures[slot_id] = cap
            return cap

        self.video_engine.register_custom_source(src_key, _opener)
        return src_key

    def clear_browser_stream(self, slot_id: int):
        slot_id = int(slot_id)
        src_key = f"browser://{slot_id}"
        self.video_engine.unregister_custom_source(src_key)
        with self._ui_lock:
            self._browser_meta.pop(slot_id, None)
            cap = self._browser_captures.pop(slot_id, None)
            if cap is not None:
                cap.release()

    def notify_browser_demand(self, slot_id: int, seq: int, sec: float, fps: float):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        payload = json.dumps(
            {
                "type": "need_frame",
                "slot_id": int(slot_id),
                "seq": int(seq),
                "sec": float(sec),
                "fps": float(fps),
            }
        )
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_ws_text(payload), loop)
        except Exception:
            pass

    async def _broadcast_ws_text(self, text: str):
        with self._ws_lock:
            clients = list(self._ws_clients)
        for ws in clients:
            try:
                await ws.send_text(text)
            except Exception:
                with self._ws_lock:
                    self._ws_clients.discard(ws)

    def get_pending_browser_demands(self):
        out = []
        with self._ui_lock:
            caps = list(self._browser_captures.values())
        for cap in caps:
            d = cap.pending_demand()
            if d is not None:
                out.append(d)
        return out

    def _encode_gallery(self):
        out = []
        for idx, p in enumerate(self.scanner.matcher.target_profiles[:8]):
            b64 = getattr(p, "_web_thumb_b64", None)
            if b64 is None and getattr(p, "thumbnail", None) is not None:
                ok, buf = cv2.imencode(".jpg", p.thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
                p._web_thumb_b64 = b64
            out.append(
                {
                    "index": idx,
                    "angle_label": p.angle_label,
                    "has_face": bool(p.has_face),
                    "thumb_b64": b64 or "",
                }
            )
        return out

    def get_snapshot_state(self):
        ve = self.video_engine
        clock_str, clock_sec = ve.get_clock_display()
        telem = ve.get_speed_telemetry()
        pct, cur_hms, max_hms, progress_summary = ve.get_progress()

        slots_data = []
        active_count = 0
        remaining_count = 0
        for i in range(4):
            st = ve.get_slot_state(i, include_frame=False)
            raw_slot = ve.slots[i]
            if st["enabled"]:
                active_count += 1
                if not st["ended"]:
                    remaining_count += 1
            src = raw_slot["source"]
            is_browser = isinstance(src, str) and src.startswith("browser://")
            b_meta = self._browser_meta.get(i, {}) if is_browser else {}
            if is_browser:
                src_label = f"⚡ {b_meta.get('filename', 'Device Video')}"
            elif src and os.path.isfile(src):
                src_label = os.path.basename(src)
            else:
                src_label = src
            slots_data.append(
                {
                    "id": i,
                    "enabled": bool(st["enabled"]),
                    "source": src,
                    "source_label": src_label,
                    "is_browser": is_browser,
                    "location": st["location"],
                    "speed": st["speed"],
                    "is_file": bool(st["is_file"]),
                    "ended": bool(st["ended"]),
                    "spotted": bool(st["spotted"]),
                    "progress_pct": round(float(st["progress_pct"]), 1),
                    "progress_str": st["progress_str"],
                    "frame_seq": int(st["frame_seq"]),
                }
            )

        with self.tracker._lock:
            last_known = dict(self.tracker.last_known)
            journey_route = list(self.tracker.journey_route)
            timeline = [
                {k: v for k, v in e.items() if k != "time_epoch"}
                for e in self.tracker.timeline
            ]
            route_arrow = self.tracker.get_journey_arrow_str()

        sec_since = self.tracker.seconds_since_last_seen()
        is_lost = self.tracker.is_target_lost()

        with self._ui_lock:
            status_text = self.status_text
            status_level = self.status_level
            tracking_mode = self.tracking_mode
            target_name = self.target_name
            class_name = self.class_name
            sensitivity_key = self.sensitivity_key
            global_speed = self.global_speed

        return {
            "is_running": bool(ve.is_running),
            "is_paused": bool(ve.is_paused),
            "status_text": status_text,
            "status_level": status_level,
            "clock_str": clock_str,
            "clock_sec": clock_sec,
            "active_count": active_count,
            "remaining_count": remaining_count,
            "telemetry": telem,
            "progress": {
                "pct": round(float(pct), 1),
                "cur": cur_hms,
                "max": max_hms,
                "summary": progress_summary,
            },
            "config": {
                "tracking_mode": tracking_mode,
                "target_name": target_name,
                "class_name": class_name,
                "sensitivity": sensitivity_key,
                "global_speed": global_speed,
            },
            "gallery": self._encode_gallery(),
            "slots": slots_data,
            "tracker": {
                "last_known": last_known,
                "route_arrow": route_arrow,
                "journey_route": journey_route,
                "timeline": timeline[-40:],
                "total_sightings": len(timeline),
                "seconds_since_last": sec_since,
                "is_lost": is_lost,
            },
            "map": {
                "node_positions": [list(p) for p in self.map_visualizer.node_positions],
                "node_angles": list(self.map_visualizer.node_angles),
                "show_fov_cones": bool(self.map_visualizer.show_fov_cones),
                "fov_span_deg": float(self.map_visualizer.fov_span_deg),
                "has_blueprint": bool(
                    self.map_visualizer.custom_blueprint_path
                    and os.path.isfile(self.map_visualizer.custom_blueprint_path)
                ),
            },
            "browser_demands": self.get_pending_browser_demands(),
            "local_fs_enabled": not bool(os.environ.get("SPACE_ID")),
        }

    def render_slot_jpeg(self, slot_id, last_seq=-1):
        """Returns (jpeg_bytes, new_seq, changed) for slot_id without copying if unchanged."""
        if not (0 <= slot_id < len(self.video_engine.slots)):
            return _DISABLED_JPEG, -1, True

        st = self.video_engine.get_slot_state(slot_id, include_frame=True)
        if not st or not st["enabled"]:
            marker = -2
            return _DISABLED_JPEG, marker, (last_seq != marker)

        frame = st.get("frame")
        seq = int(st.get("frame_seq", 0))
        if frame is None:
            marker = -1
            return _STANDBY_JPEG, marker, (last_seq != marker)

        if seq == last_seq and last_seq >= 0:
            return None, seq, False

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None, seq, False
        return buf.tobytes(), seq, True


STATE: WebAppState | None = None


def get_state() -> WebAppState:
    global STATE
    if STATE is None:
        STATE = WebAppState()
    return STATE


@asynccontextmanager
async def lifespan(_app):
    st = get_state()
    st._loop = asyncio.get_running_loop()
    yield
    if STATE is not None:
        for sid in range(4):
            STATE.clear_browser_stream(sid)
        STATE.video_engine.stop()


async def index_page(_request: Request):
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


async def api_health(_request: Request):
    st = get_state()
    return JSONResponse({"ok": True, "cuda": bool(st.video_engine.has_cuda)})


async def api_state(_request: Request):
    st = get_state()
    data = await asyncio.to_thread(st.get_snapshot_state)
    return JSONResponse(data)


async def api_frame(request: Request):
    slot_id = int(request.path_params["slot_id"])
    st = get_state()
    jpg, _, _ = await asyncio.to_thread(st.render_slot_jpeg, slot_id, -999)
    return Response(content=jpg or _STANDBY_JPEG, media_type="image/jpeg")


async def api_stream(request: Request):
    slot_id = int(request.path_params["slot_id"])
    st = get_state()

    async def frame_generator():
        last_seq = -999
        while True:
            if await request.is_disconnected():
                break
            jpg, new_seq, changed = await asyncio.to_thread(st.render_slot_jpeg, slot_id, last_seq)
            if changed and jpg:
                last_seq = new_seq
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                )
                yield header + jpg + b"\r\n"
            await asyncio.sleep(0.045)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


async def _save_upload_file(upload_file, subdir):
    dest_dir = os.path.join(UPLOADS_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = _sanitize_filename(getattr(upload_file, "filename", "upload"))
    dest_path = os.path.join(dest_dir, safe_name)
    content = await upload_file.read()
    with open(dest_path, "wb") as f:
        f.write(content)
    return dest_path


async def api_target_upload(request: Request):
    st = get_state()
    form = await request.form()
    name = str(form.get("name") or st.target_name or "Target Person").strip()
    append = str(form.get("append") or "false").lower() == "true"
    files = form.getlist("files")
    if not files:
        single = form.get("file")
        if single is not None:
            files = [single]
    if not files:
        return JSONResponse({"ok": False, "error": "Please select at least one photo."}, status_code=400)

    raw_bytes_list = []
    for f in files:
        if hasattr(f, "read"):
            raw_bytes_list.append(await f.read())

    def _process():
        if not append:
            st.scanner.matcher.clear_target()
        st.scanner.tracking_mode = "person_photo"
        st.scanner.matcher.target_name = name
        st.tracker.set_target(name)
        added = 0
        for raw in raw_bytes_list:
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                st.scanner.add_person_photo(img)
                added += 1
        with st._ui_lock:
            st.tracking_mode = "person"
            st.target_name = name
        if added > 0:
            st.scanner.last_sighting["found"] = False
            st.scanner.last_sighting["target_name"] = name
            st.scanner.sightings_history.clear()
            cnt = len(st.scanner.matcher.target_profiles)
            st.set_status(f"Target ready: {name} ({cnt} angle profile(s))", "green")
        return added

    added = await asyncio.to_thread(_process)
    if added == 0:
        return JSONResponse({"ok": False, "error": "Could not decode uploaded image(s)."}, status_code=400)
    return JSONResponse({"ok": True, "added": added, "total_profiles": len(st.scanner.matcher.target_profiles)})


async def api_target_upload_360(request: Request):
    st = get_state()
    form = await request.form()
    up = form.get("file")
    if up is None or not hasattr(up, "read"):
        return JSONResponse({"ok": False, "error": "No 360° video clip provided."}, status_code=400)
    name = str(form.get("name") or st.target_name or "Target Person").strip()
    saved_path = await _save_upload_file(up, "targets")

    try:
        def _extract():
            st.tracker.set_target(name)
            count = st.scanner.extract_360_angles_from_video(saved_path, target_name=name)
            with st._ui_lock:
                st.tracking_mode = "person"
                st.target_name = name
            st.set_status(f"360° Target Ready: {name} ({count} angles extracted)", "green")
            return count

        count = await asyncio.to_thread(_extract)
        return JSONResponse({"ok": True, "extracted": count})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


async def api_target_clear(_request: Request):
    st = get_state()
    st.scanner.matcher.clear_target()
    st.set_status("Target gallery cleared.", "dim")
    return JSONResponse({"ok": True})


async def api_target_mode(request: Request):
    st = get_state()
    data = await request.json()
    mode = str(data.get("tracking_mode") or st.tracking_mode).strip()
    target_name = str(data.get("target_name") or st.target_name).strip() or "Target Person"
    class_name = str(data.get("class_name") or st.class_name).strip() or "dog"
    sens = str(data.get("sensitivity") or st.sensitivity_key).strip().lower()

    thresh_map = {"strict": 0.70, "balanced": 0.60, "flexible": 0.50}
    st.scanner.matcher.match_threshold = thresh_map.get(sens, 0.70)

    with st._ui_lock:
        st.tracking_mode = "class" if mode == "class" else "person"
        st.target_name = target_name
        st.class_name = class_name
        st.sensitivity_key = sens if sens in thresh_map else "strict"

    if st.tracking_mode == "class":
        st.scanner.set_class_target(class_name)
        st.tracker.set_target(f"{class_name.capitalize()} Intrusion Detection")
    else:
        st.scanner.tracking_mode = "person_photo"
        st.scanner.matcher.target_name = target_name

    st.set_status(
        f"Mode: {'Class (' + class_name + ')' if st.tracking_mode == 'class' else 'Person'} | Sensitivity: {st.sensitivity_key.title()}",
        "green",
    )
    return JSONResponse({"ok": True})


async def api_slot_upload_video(request: Request):
    slot_id = int(request.path_params["slot_id"])
    if not (0 <= slot_id < 4):
        return JSONResponse({"ok": False, "error": "Invalid camera slot."}, status_code=400)

    st = get_state()
    form = await request.form()
    up = form.get("file")
    if up is None or not hasattr(up, "read"):
        return JSONResponse({"ok": False, "error": "No video file provided."}, status_code=400)

    orig_name = os.path.basename(getattr(up, "filename", "video.mp4"))
    saved_path = await _save_upload_file(up, "videos")
    clean_loc = os.path.splitext(orig_name)[0].replace("_", " ").strip().title()
    cur_slot = st.video_engine.slots[slot_id]
    new_loc = cur_slot["location"]
    if not new_loc or new_loc.startswith("Camera"):
        new_loc = f"Camera {slot_id + 1} ({clean_loc})"

    def _apply():
        st.clear_browser_stream(slot_id)
        if st.video_engine.is_running:
            st.video_engine.replace_slot(slot_id, saved_path, location=new_loc, speed=cur_slot["speed"])
        else:
            st.video_engine.configure_slot(
                slot_id, enabled=True, source=saved_path, location=new_loc, speed=cur_slot["speed"]
            )
        st.set_status(f"Camera {slot_id + 1} loaded: {orig_name}", "green")

    try:
        await asyncio.to_thread(_apply)
        return JSONResponse({"ok": True, "source": saved_path, "location": new_loc})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


async def api_slot_configure(request: Request):
    slot_id = int(request.path_params["slot_id"])
    if not (0 <= slot_id < 4):
        return JSONResponse({"ok": False, "error": "Invalid camera slot."}, status_code=400)

    st = get_state()
    data = await request.json()
    ve = st.video_engine

    def _apply():
        cur = ve.slots[slot_id]
        enabled = data.get("enabled", cur["enabled"])
        source = data.get("source", cur["source"])
        location = data.get("location", cur["location"])
        speed = data.get("speed", cur["speed"])

        if str(source).strip() != f"browser://{slot_id}":
            st.clear_browser_stream(slot_id)

        if "location" in data and data["location"] is not None:
            ve.set_slot_location(slot_id, str(location))

        if ve.is_running:
            if enabled is False and cur["enabled"]:
                ve.remove_slot(slot_id, clear_source=False)
                return True, speed
            if enabled is True and (not cur["enabled"] or str(source).strip() != cur["source"]):
                ve.replace_slot(slot_id, str(source).strip() or "0", location=location, speed=speed)
                return True, ve.slots[slot_id]["speed"]
            if "speed" in data:
                return ve.set_slot_speed(slot_id, str(speed))
            return True, cur["speed"]

        ve.configure_slot(slot_id, enabled=enabled, source=source, location=location, speed=speed)
        return True, str(speed)

    try:
        changed, applied_spd = await asyncio.to_thread(_apply)
        return JSONResponse({"ok": True, "changed": changed, "applied_speed": applied_spd})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


async def api_slot_remove(request: Request):
    slot_id = int(request.path_params["slot_id"])
    if not (0 <= slot_id < 4):
        return JSONResponse({"ok": False, "error": "Invalid camera slot."}, status_code=400)
    st = get_state()
    remaining = await asyncio.to_thread(st.video_engine.remove_slot, slot_id, True)
    st.clear_browser_stream(slot_id)
    st.set_status(
        f"Removed Camera {slot_id + 1}" + (f" ({remaining} active)" if remaining else " — Idle"),
        "amber" if remaining else "dim",
    )
    return JSONResponse({"ok": True, "remaining": remaining})


async def api_control_start(request: Request):
    st = get_state()
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    ve = st.video_engine
    global_speed = str(body.get("global_speed") or st.global_speed or "10x").strip()
    with st._ui_lock:
        st.global_speed = global_speed
        mode = st.tracking_mode
        target_name = st.target_name
        class_name = st.class_name
        sens = st.sensitivity_key

    thresh_map = {"strict": 0.70, "balanced": 0.60, "flexible": 0.50}
    st.scanner.matcher.match_threshold = thresh_map.get(sens, 0.70)

    if mode == "person":
        if len(st.scanner.matcher.target_profiles) == 0:
            return JSONResponse(
                {"ok": False, "error": "Please upload at least one target photo first!"},
                status_code=400,
            )
        st.scanner.tracking_mode = "person_photo"
        st.scanner.matcher.target_name = target_name
        st.tracker.set_target(target_name)
    else:
        st.scanner.set_class_target(class_name)
        st.tracker.set_target(f"{class_name.capitalize()} Intrusion Detection")

    st.set_status("INITIALIZING CAMERA STREAMS...", "amber")
    try:
        active_count = await asyncio.to_thread(ve.start, global_speed, st.on_scan_finished)
        st.set_status(f"SYNCHRONIZED SEARCH: {active_count} CAMERAS @ {global_speed}", "green")
        return JSONResponse({"ok": True, "active_count": active_count})
    except Exception as exc:
        st.set_status("STREAM CONNECTION ERROR", "red")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


async def api_control_pause(_request: Request):
    st = get_state()
    paused = await asyncio.to_thread(st.video_engine.toggle_pause)
    st.set_status("PAUSED — SCANNING FROZEN" if paused else f"RESUMED @ {st.global_speed}", "amber" if paused else "green")
    return JSONResponse({"ok": True, "is_paused": bool(paused)})


async def api_control_stop(_request: Request):
    st = get_state()
    await asyncio.to_thread(st.video_engine.stop)
    st.set_status("IDLE", "dim")
    return JSONResponse({"ok": True})


async def api_control_speed(request: Request):
    st = get_state()
    data = await request.json()
    spd = str(data.get("global_speed") or "10x").strip()
    with st._ui_lock:
        st.global_speed = spd
    await asyncio.to_thread(st.video_engine.set_global_speed, spd)
    if st.video_engine.is_running and not st.video_engine.is_paused:
        st.set_status(f"Live Playback Speed set to {spd}", "green")
    return JSONResponse({"ok": True, "global_speed": spd})


async def api_tracker_clear(_request: Request):
    st = get_state()
    st.tracker.set_target(st.target_name if st.tracking_mode == "person" else f"{st.class_name.capitalize()} Intrusion")
    return JSONResponse({"ok": True})


async def api_tracker_export(request: Request):
    fmt = str(request.path_params["fmt"]).lower()
    st = get_state()
    if fmt == "csv":
        path = await asyncio.to_thread(st.tracker.export_to_csv)
        media = "text/csv"
    elif fmt == "json":
        path = await asyncio.to_thread(st.tracker.export_to_json)
        media = "application/json"
    else:
        return JSONResponse({"ok": False, "error": "Unsupported export format."}, status_code=400)
    if not path or not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "No sightings recorded yet to export."}, status_code=404)
    return FileResponse(path, media_type=media, filename=os.path.basename(path))


async def api_snapshot(request: Request):
    st = get_state()
    raw_path = request.query_params.get("path", "")
    if not raw_path:
        return Response(status_code=404)
    alerts_root = os.path.realpath(st.scanner.alerts_dir)
    cand = os.path.realpath(raw_path)
    if not cand.startswith(alerts_root + os.sep) or not os.path.isfile(cand):
        return Response(status_code=404)
    return FileResponse(cand, media_type="image/jpeg")


async def api_map_config(request: Request):
    st = get_state()
    data = await request.json()
    mv = st.map_visualizer
    if "cam_idx" in data:
        idx = int(data["cam_idx"])
        if "u" in data and "v" in data:
            mv.set_node_position(idx, float(data["u"]), float(data["v"]))
        if "angle" in data:
            mv.set_node_angle(idx, float(data["angle"]))
    if "toggle_fov" in data and data["toggle_fov"]:
        mv.toggle_fov_cones()
    return JSONResponse(
        {
            "ok": True,
            "node_positions": [list(p) for p in mv.node_positions],
            "node_angles": list(mv.node_angles),
            "show_fov_cones": bool(mv.show_fov_cones),
        }
    )


async def api_map_reset(_request: Request):
    st = get_state()
    st.map_visualizer.reset_to_default()
    return JSONResponse({"ok": True})


async def api_map_blueprint_upload(request: Request):
    st = get_state()
    form = await request.form()
    if form.get("clear") == "true":
        st.map_visualizer.clear_custom_blueprint()
        return JSONResponse({"ok": True, "has_blueprint": False})
    up = form.get("file")
    if up is None or not hasattr(up, "read"):
        return JSONResponse({"ok": False, "error": "No floorplan image uploaded."}, status_code=400)
    saved = await _save_upload_file(up, "blueprints")
    if not st.map_visualizer.set_custom_blueprint(saved):
        return JSONResponse({"ok": False, "error": "Invalid image file."}, status_code=400)
    return JSONResponse({"ok": True, "has_blueprint": True})


async def api_map_blueprint_get(_request: Request):
    st = get_state()
    bp = st.map_visualizer.custom_blueprint_path
    if not bp or not os.path.isfile(bp):
        return Response(status_code=404)
    return FileResponse(bp)


async def api_slot_browser_register(request: Request):
    slot_id = int(request.path_params["slot_id"])
    if not (0 <= slot_id < 4):
        return JSONResponse({"ok": False, "error": "Invalid camera slot."}, status_code=400)

    st = get_state()
    data = await request.json()
    filename = os.path.basename(str(data.get("filename") or f"device_video_{slot_id + 1}.mp4"))
    duration_sec = float(data.get("duration_sec") or 0.0)
    fps = float(data.get("fps") or 15.0)
    if duration_sec <= 0:
        return JSONResponse({"ok": False, "error": "Invalid video duration."}, status_code=400)

    src_key = st.register_browser_stream(slot_id, filename, duration_sec, fps=fps)
    clean_loc = os.path.splitext(filename)[0].replace("_", " ").strip().title()
    cur_slot = st.video_engine.slots[slot_id]
    new_loc = cur_slot["location"]
    if not new_loc or new_loc.startswith("Camera"):
        new_loc = f"Camera {slot_id + 1} ({clean_loc})"

    def _apply():
        if st.video_engine.is_running:
            st.video_engine.replace_slot(slot_id, src_key, location=new_loc, speed=cur_slot["speed"])
        else:
            st.video_engine.configure_slot(
                slot_id, enabled=True, source=src_key, location=new_loc, speed=cur_slot["speed"]
            )
        st.set_status(f"Camera {slot_id + 1} instant stream ready: {filename}", "green")

    try:
        await asyncio.to_thread(_apply)
        return JSONResponse({"ok": True, "source": src_key, "location": new_loc, "filename": filename})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


async def api_slot_push_frame(request: Request):
    slot_id = int(request.path_params["slot_id"])
    if not (0 <= slot_id < 4):
        return JSONResponse({"ok": False, "error": "Invalid camera slot."}, status_code=400)
    st = get_state()
    with st._ui_lock:
        cap = st._browser_captures.get(slot_id)
    if cap is None:
        return JSONResponse({"ok": False, "error": "No active browser stream for this slot."}, status_code=404)

    eof = request.query_params.get("eof") == "1"
    if eof:
        cap.signal_eof()
        return JSONResponse({"ok": True, "demand": None})

    try:
        seq = int(request.query_params.get("seq") or 0)
    except ValueError:
        seq = 0
    raw = await request.body()
    ok = await asyncio.to_thread(cap.deliver_jpeg, seq, raw) if raw else False
    return JSONResponse({"ok": bool(ok), "demand": cap.pending_demand()})


async def ws_browser_source(websocket: WebSocket):
    st = get_state()
    st._loop = asyncio.get_running_loop()
    await websocket.accept()
    with st._ws_lock:
        st._ws_clients.add(websocket)
    try:
        for d in st.get_pending_browser_demands():
            await websocket.send_text(json.dumps({"type": "need_frame", **d}))
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            raw_bytes = msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 5:
                slot_id = raw_bytes[0]
                seq = int.from_bytes(raw_bytes[1:5], "big", signed=False)
                jpeg_bytes = raw_bytes[5:]
                with st._ui_lock:
                    cap = st._browser_captures.get(slot_id)
                if cap is not None:
                    await asyncio.to_thread(cap.deliver_jpeg, seq, jpeg_bytes)
                continue
            txt = msg.get("text")
            if txt:
                try:
                    ev = json.loads(txt)
                    if ev.get("type") == "eof":
                        sid = int(ev.get("slot_id", -1))
                        with st._ui_lock:
                            cap = st._browser_captures.get(sid)
                        if cap is not None:
                            cap.signal_eof()
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        with st._ws_lock:
            st._ws_clients.discard(websocket)


_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".ts", ".flv", ".wmv"}


async def api_fs_list(request: Request):
    if os.environ.get("SPACE_ID"):
        return JSONResponse(
            {
                "ok": True,
                "enabled": False,
                "reason": "Cloud container mode — use Instant Video to stream from your device without uploading.",
            }
        )

    raw_path = str(request.query_params.get("path") or "").strip()

    def _scan():
        if not raw_path:
            shortcuts = []
            home = os.path.expanduser("~")
            for label, p in [
                ("SmartSupervise Folder", BASE_DIR),
                ("Desktop", os.path.join(home, "Desktop")),
                ("Videos", os.path.join(home, "Videos")),
                ("Downloads", os.path.join(home, "Downloads")),
                ("Home", home),
            ]:
                if os.path.isdir(p):
                    shortcuts.append({"name": label, "path": p, "is_dir": True, "size_mb": None})
            if os.name == "nt":
                for letter in string.ascii_uppercase:
                    drv = f"{letter}:\\"
                    if os.path.exists(drv):
                        shortcuts.append({"name": f"Drive ({letter}:)", "path": drv, "is_dir": True, "size_mb": None})
            elif os.path.isdir("/"):
                shortcuts.append({"name": "Root /", "path": "/", "is_dir": True, "size_mb": None})
            return {"ok": True, "enabled": True, "current": "", "parent": "", "items": shortcuts}

        target = os.path.abspath(raw_path)
        if not os.path.isdir(target):
            return {"ok": False, "error": f"Folder not found: {raw_path}"}

        parent = os.path.dirname(target.rstrip("\\/")) if len(target) > 3 else ""
        if os.name == "nt" and len(target) <= 3 and target.endswith("\\"):
            parent = ""

        dirs, files = [], []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith(".") or name in ("$Recycle.Bin", "System Volume Information"):
                        continue
                    try:
                        if entry.is_dir():
                            dirs.append({"name": name, "path": entry.path, "is_dir": True, "size_mb": None})
                        elif entry.is_file() and os.path.splitext(name)[1].lower() in _VIDEO_EXTS:
                            sz = round(entry.stat().st_size / (1024 * 1024), 1)
                            files.append({"name": name, "path": entry.path, "is_dir": False, "size_mb": sz})
                    except OSError:
                        continue
        except PermissionError:
            return {"ok": False, "error": f"Permission denied opening: {target}"}

        dirs.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())
        return {
            "ok": True,
            "enabled": True,
            "current": target,
            "parent": parent,
            "items": (dirs + files)[:250],
        }

    res = await asyncio.to_thread(_scan)
    return JSONResponse(res, status_code=200 if res.get("ok") else 400)


routes = [
    Route("/", index_page, methods=["GET"]),
    Route("/api/health", api_health, methods=["GET"]),
    Route("/api/state", api_state, methods=["GET"]),
    Route("/api/frame/{slot_id:int}", api_frame, methods=["GET"]),
    Route("/api/stream/{slot_id:int}", api_stream, methods=["GET"]),
    Route("/api/target/upload", api_target_upload, methods=["POST"]),
    Route("/api/target/upload_360_video", api_target_upload_360, methods=["POST"]),
    Route("/api/target/clear", api_target_clear, methods=["POST"]),
    Route("/api/target/mode", api_target_mode, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/upload_video", api_slot_upload_video, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/browser_register", api_slot_browser_register, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/push_frame", api_slot_push_frame, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/configure", api_slot_configure, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/remove", api_slot_remove, methods=["POST"]),
    Route("/api/fs/list", api_fs_list, methods=["GET"]),
    Route("/api/control/start", api_control_start, methods=["POST"]),
    Route("/api/control/pause", api_control_pause, methods=["POST"]),
    Route("/api/control/stop", api_control_stop, methods=["POST"]),
    Route("/api/control/speed", api_control_speed, methods=["POST"]),
    Route("/api/tracker/clear", api_tracker_clear, methods=["POST"]),
    Route("/api/tracker/export/{fmt:str}", api_tracker_export, methods=["GET"]),
    Route("/api/snapshot", api_snapshot, methods=["GET"]),
    Route("/api/map/config", api_map_config, methods=["POST"]),
    Route("/api/map/reset", api_map_reset, methods=["POST"]),
    Route("/api/map/blueprint", api_map_blueprint_upload, methods=["POST"]),
    Route("/api/map/blueprint", api_map_blueprint_get, methods=["GET"]),
    WebSocketRoute("/ws/browser_source", ws_browser_source),
]

app = Starlette(routes=routes, lifespan=lifespan)


def _detect_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    default_port = 7860 if os.environ.get("SPACE_ID") else 8000
    port = int(os.environ.get("PORT", default_port))
    lan_ip = _detect_lan_ip()
    print("=" * 64)
    print("  SMART SUPERVISE — Responsive Web Console (Desktop + Mobile)")
    print(f"  Local URL:  http://127.0.0.1:{port}")
    print(f"  Phone URL:  http://{lan_ip}:{port}  (on same Wi-Fi)")
    print("=" * 64)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
