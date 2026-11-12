"""
Thread-safe application state singleton connecting core AI/video/tracking engines
with Starlette routes, WebSockets, and MJPEG stream generators.
"""

import os
import cv2
import json
import base64
import asyncio
import threading
import numpy as np
from contextlib import asynccontextmanager

from core.cctv_scanner import CCTVScanner
from core.location_tracker import LocationTracker
from core.map_visualizer import MapVisualizer
from core.video_engine import VideoEngine

from webapp.browser_stream import BrowserFrameCapture
from webapp.security import auth_enabled, host_fs_allowed, install_windows_proactor_filter


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


STANDBY_JPEG = _make_placeholder_jpeg("SMART SUPERVISE", "Feed in standby — configure camera and press Start")
DISABLED_JPEG = _make_placeholder_jpeg("CAMERA OFF", "Enable or upload video to monitor this feed", (90, 95, 130))


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
        self._browser_captures = {}   # slot_id -> latest BrowserFrameCapture

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
            cap = BrowserFrameCapture(
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
        tl_summary = self.tracker.get_timeline_summary()
        formatted_journey = self.tracker.get_formatted_journey()

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
            "last_spotlight_idx": int(getattr(ve, "last_spotlight_idx", 0)),
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
                "time_since_last_seen": tl_summary.get("time_since_last_seen"),
                "formatted_journey": formatted_journey,
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
            "local_fs_enabled": host_fs_allowed(),
            "auth_required": auth_enabled(),
        }

    def render_slot_jpeg(self, slot_id, last_seq=-1):
        """Return (jpeg_bytes, new_seq, changed) for slot_id without copying if unchanged."""
        if not (0 <= slot_id < len(self.video_engine.slots)):
            return DISABLED_JPEG, -1, True

        st = self.video_engine.get_slot_state(slot_id, include_frame=True)
        if not st or not st["enabled"]:
            marker = -2
            return DISABLED_JPEG, marker, (last_seq != marker)

        frame = st.get("frame")
        seq = int(st.get("frame_seq", 0))
        if frame is None:
            marker = -1
            return STANDBY_JPEG, marker, (last_seq != marker)

        if seq == last_seq and last_seq >= 0:
            return None, seq, False

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None, seq, False
        return buf.tobytes(), seq, True


STATE: WebAppState | None = None


def get_state() -> WebAppState:
    """Return the lazily instantiated global WebAppState singleton."""
    global STATE
    if STATE is None:
        STATE = WebAppState()
    return STATE


@asynccontextmanager
async def lifespan(_app):
    """Initialize event loop integration and clean up camera threads on server exit."""
    loop = asyncio.get_running_loop()
    install_windows_proactor_filter(loop)
    st = get_state()
    st._loop = loop
    yield
    if STATE is not None:
        for sid in range(4):
            STATE.clear_browser_stream(sid)
        STATE.video_engine.stop()
