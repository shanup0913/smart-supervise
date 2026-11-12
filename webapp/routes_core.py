"""
HTTP route handlers for UI entry, authentication, target setup, playback control,
timeline exports, radar map configuration, and restricted host media folder browsing.
"""

import os
import cv2
import hmac
import asyncio
import numpy as np

from starlette.requests import Request
from starlette.routing import Route
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response

from webapp.engine_state import get_state
from webapp.security import (
    BASE_DIR,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    UPLOADS_DIR,
    WEB_DIR,
    _COOKIE_NAME,
    _IMAGE_EXTS,
    _SESSION_TTL_SEC,
    _VIDEO_EXTS,
    allowed_media_roots,
    auth_enabled,
    can_browse_host_fs,
    check_login_rate_limit,
    clear_login_failures,
    client_ip,
    get_access_code,
    is_authenticated,
    is_cloud_container,
    is_under_root,
    record_login_failure,
    save_upload_file,
    sign_session_token,
)

_BLOCKED_DIR_NAMES = {
    "tools",
    "models",
    "core",
    "ui",
    "web",
    "webapp",
    "reports",
    "alerts",
    "__pycache__",
    "$Recycle.Bin",
    "System Volume Information",
    "Windows",
    "Program Files",
    "Program Files (x86)",
}


async def index_page(_request: Request):
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


async def api_health(request: Request):
    if auth_enabled() and not is_authenticated(request):
        return JSONResponse({"ok": True})
    st = get_state()
    return JSONResponse({"ok": True, "cuda": bool(st.video_engine.has_cuda)})


async def api_auth_status(request: Request):
    return JSONResponse(
        {
            "ok": True,
            "auth_required": auth_enabled(),
            "authenticated": is_authenticated(request),
        }
    )


async def api_auth_login(request: Request):
    if not auth_enabled():
        return JSONResponse({"ok": True, "auth_required": False, "authenticated": True})
    ip = client_ip(request)
    if check_login_rate_limit(ip):
        return JSONResponse(
            {"ok": False, "error": "Too many failed passcode attempts. Wait 60 seconds."},
            status_code=429,
        )
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass
    supplied = str(data.get("passcode") or "").strip()
    expected = get_access_code()
    if not supplied or not expected or not hmac.compare_digest(supplied, expected):
        record_login_failure(ip)
        return JSONResponse({"ok": False, "error": "Invalid access passcode."}, status_code=401)

    clear_login_failures(ip)
    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )
    cookie_samesite = "none" if (os.environ.get("SPACE_ID") and is_https) else "lax"
    resp = JSONResponse({"ok": True, "authenticated": True})
    resp.set_cookie(
        key=_COOKIE_NAME,
        value=sign_session_token(),
        max_age=_SESSION_TTL_SEC,
        httponly=True,
        samesite=cookie_samesite,
        secure=is_https,
        path="/",
    )
    return resp


async def api_auth_logout(request: Request):
    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )
    cookie_samesite = "none" if (os.environ.get("SPACE_ID") and is_https) else "lax"
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=_COOKIE_NAME, path="/", samesite=cookie_samesite, secure=is_https)
    return resp


async def api_state(request: Request):
    st = get_state()
    data = await asyncio.to_thread(st.get_snapshot_state)
    data["local_fs_enabled"] = can_browse_host_fs(request)
    return JSONResponse(data)


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
    for f in files[:12]:
        if hasattr(f, "read"):
            ext = os.path.splitext(str(getattr(f, "filename", "") or ""))[1].lower()
            if ext and ext not in _IMAGE_EXTS:
                return JSONResponse({"ok": False, "error": f"Unsupported image format: {ext}"}, status_code=400)
            chunks = []
            total = 0
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    return JSONResponse({"ok": False, "error": "Image exceeds 20 MB limit."}, status_code=413)
                chunks.append(chunk)
            raw_bytes_list.append(b"".join(chunks))

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
    try:
        saved_path = await save_upload_file(up, "targets", allowed_exts=_VIDEO_EXTS, max_bytes=MAX_VIDEO_BYTES)
    except ValueError as exc:
        msg = str(exc)
        return JSONResponse({"ok": False, "error": msg}, status_code=413 if "limit" in msg.lower() else 400)

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
    st.set_status(
        "PAUSED — SCANNING FROZEN" if paused else f"RESUMED @ {st.global_speed}",
        "amber" if paused else "green",
    )
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


async def api_tracker_export_both(_request: Request):
    st = get_state()
    csv_path, json_path = await asyncio.to_thread(st.tracker.export_to_csv_and_json)
    if not csv_path or not json_path:
        return JSONResponse({"ok": False, "error": "No sightings recorded yet to export."}, status_code=404)
    return JSONResponse(
        {
            "ok": True,
            "csv_name": os.path.basename(csv_path),
            "json_name": os.path.basename(json_path),
        }
    )


async def api_snapshot(request: Request):
    st = get_state()
    raw_path = request.query_params.get("path", "")
    if not raw_path:
        return Response(status_code=404)
    alerts_root = os.path.realpath(st.scanner.alerts_dir)
    cand = os.path.realpath(raw_path)
    if not is_under_root(cand, alerts_root) or not os.path.isfile(cand):
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
    try:
        saved = await save_upload_file(up, "blueprints", allowed_exts=_IMAGE_EXTS, max_bytes=MAX_IMAGE_BYTES)
    except ValueError as exc:
        msg = str(exc)
        return JSONResponse({"ok": False, "error": msg}, status_code=413 if "limit" in msg.lower() else 400)
    if not st.map_visualizer.set_custom_blueprint(saved):
        return JSONResponse({"ok": False, "error": "Invalid image file."}, status_code=400)
    return JSONResponse({"ok": True, "has_blueprint": True})


async def api_map_blueprint_get(_request: Request):
    st = get_state()
    bp = st.map_visualizer.custom_blueprint_path
    if not bp or not os.path.isfile(bp):
        return Response(status_code=404)
    return FileResponse(bp)


async def api_fs_list(request: Request):
    if is_cloud_container():
        return JSONResponse(
            {
                "ok": True,
                "enabled": False,
                "reason": "Cloud container mode — use Instant Video to stream from your device without uploading.",
            }
        )
    if not can_browse_host_fs(request):
        return JSONResponse(
            {
                "ok": False,
                "enabled": False,
                "error": "Host folder browsing is disabled for this connection.",
            },
            status_code=403,
        )

    raw_path = str(request.query_params.get("path") or "").strip()
    roots = allowed_media_roots()

    def _scan():
        if not raw_path:
            shortcuts = []
            home = os.path.expanduser("~")
            label_map = {
                os.path.realpath(UPLOADS_DIR): "Uploaded Videos",
                os.path.realpath(BASE_DIR): "SmartSupervise Folder",
                os.path.realpath(os.path.join(home, "Videos")): "Videos",
                os.path.realpath(os.path.join(home, "Desktop")): "Desktop",
                os.path.realpath(os.path.join(home, "Downloads")): "Downloads",
                os.path.realpath(os.path.join(home, "Movies")): "Movies",
            }
            for rp in roots:
                shortcuts.append(
                    {"name": label_map.get(rp, os.path.basename(rp) or rp), "path": rp, "is_dir": True, "size_mb": None}
                )
            return {"ok": True, "enabled": True, "current": "", "parent": "", "items": shortcuts}

        target = os.path.realpath(raw_path)
        if not os.path.isdir(target) or not any(is_under_root(target, r) for r in roots):
            return {"ok": False, "error": "Folder is outside allowed media directories.", "status": 403}

        is_root = target in roots
        parent = "" if is_root else os.path.dirname(target.rstrip("\\/"))
        if parent and not any(is_under_root(parent, r) for r in roots):
            parent = ""

        dirs, files = [], []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith(".") or name in _BLOCKED_DIR_NAMES:
                        continue
                    try:
                        ep = os.path.realpath(entry.path)
                        if not any(is_under_root(ep, r) for r in roots):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append({"name": name, "path": ep, "is_dir": True, "size_mb": None})
                        elif entry.is_file(follow_symlinks=False) and os.path.splitext(name)[1].lower() in _VIDEO_EXTS:
                            sz = round(entry.stat().st_size / (1024 * 1024), 1)
                            files.append({"name": name, "path": ep, "is_dir": False, "size_mb": sz})
                    except OSError:
                        continue
        except PermissionError:
            return {"ok": False, "error": f"Permission denied opening: {target}", "status": 403}

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
    code = 200 if res.get("ok") else res.pop("status", 400)
    return JSONResponse(res, status_code=code)


core_routes = [
    Route("/", index_page, methods=["GET"]),
    Route("/api/health", api_health, methods=["GET"]),
    Route("/api/auth/status", api_auth_status, methods=["GET"]),
    Route("/api/auth/login", api_auth_login, methods=["POST"]),
    Route("/api/auth/logout", api_auth_logout, methods=["POST"]),
    Route("/api/state", api_state, methods=["GET"]),
    Route("/api/target/upload", api_target_upload, methods=["POST"]),
    Route("/api/target/upload_360_video", api_target_upload_360, methods=["POST"]),
    Route("/api/target/clear", api_target_clear, methods=["POST"]),
    Route("/api/target/mode", api_target_mode, methods=["POST"]),
    Route("/api/fs/list", api_fs_list, methods=["GET"]),
    Route("/api/control/start", api_control_start, methods=["POST"]),
    Route("/api/control/pause", api_control_pause, methods=["POST"]),
    Route("/api/control/stop", api_control_stop, methods=["POST"]),
    Route("/api/control/speed", api_control_speed, methods=["POST"]),
    Route("/api/tracker/clear", api_tracker_clear, methods=["POST"]),
    Route("/api/tracker/export/both", api_tracker_export_both, methods=["POST"]),
    Route("/api/tracker/export/{fmt:str}", api_tracker_export, methods=["GET"]),
    Route("/api/snapshot", api_snapshot, methods=["GET"]),
    Route("/api/map/config", api_map_config, methods=["POST"]),
    Route("/api/map/reset", api_map_reset, methods=["POST"]),
    Route("/api/map/blueprint", api_map_blueprint_upload, methods=["POST"]),
    Route("/api/map/blueprint", api_map_blueprint_get, methods=["GET"]),
]
