"""
Smart Supervise — Responsive Cross-Platform Web Application (Mobile, Tablet & Desktop)
Shares core/ AI, VideoEngine, LocationTracker, and MapVisualizer directly with app.py.
Deploys seamlessly on Hugging Face Spaces (Docker port 7860), local Wi-Fi, or any cloud host.
"""

import os
import sys
import cv2
import time
import uuid
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
from starlette.routing import Route
from starlette.requests import Request
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


class WebAppState:
    """Thread-safe coordinator binding Starlette HTTP/MJPEG handlers to core engines."""

    def __init__(self):
        self.scanner = CCTVScanner()
        self.tracker = LocationTracker(lost_timeout_seconds=30)
        self.map_visualizer = MapVisualizer()
        self.video_engine = VideoEngine(self.scanner, self.tracker)

        self._ui_lock = threading.RLock()
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
            slots_data.append(
                {
                    "id": i,
                    "enabled": bool(st["enabled"]),
                    "source": src,
                    "source_label": os.path.basename(src) if (src and os.path.isfile(src)) else src,
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
    get_state()
    yield
    if STATE is not None:
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
    Route("/api/slot/{slot_id:int}/configure", api_slot_configure, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/remove", api_slot_remove, methods=["POST"]),
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
