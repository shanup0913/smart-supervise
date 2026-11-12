"""
Streaming routes: camera slot management, MJPEG/snapshot image feeds,
and the zero-upload WebSocket browser video feeder.
"""

import os
import json
import asyncio

from starlette.requests import Request
from starlette.routing import Route, WebSocketRoute
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from webapp.engine_state import STANDBY_JPEG, get_state
from webapp.security import (
    MAX_VIDEO_BYTES,
    _VIDEO_EXTS,
    auth_enabled,
    is_authenticated,
    save_upload_file,
    validate_slot_source,
)


async def api_frame(request: Request):
    slot_id = int(request.path_params["slot_id"])
    st = get_state()
    jpg, _, _ = await asyncio.to_thread(st.render_slot_jpeg, slot_id, -999)
    return Response(content=jpg or STANDBY_JPEG, media_type="image/jpeg")


async def api_stream(request: Request):
    slot_id = int(request.path_params["slot_id"])
    st = get_state()

    async def frame_generator():
        last_seq = -999
        try:
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
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError, OSError):
            return

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


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
    try:
        saved_path = await save_upload_file(up, "videos", allowed_exts=_VIDEO_EXTS, max_bytes=MAX_VIDEO_BYTES)
    except ValueError as exc:
        msg = str(exc)
        return JSONResponse({"ok": False, "error": msg}, status_code=413 if "limit" in msg.lower() else 400)
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
        await asyncio.to_thread(st.run_storage_cleanup)
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

    if "source" in data:
        ok_src, norm_src_or_err = validate_slot_source(data.get("source"), conn=request)
        if not ok_src:
            return JSONResponse({"ok": False, "error": norm_src_or_err}, status_code=400)
        validated_source = norm_src_or_err
    else:
        validated_source = None

    def _apply():
        cur = ve.slots[slot_id]
        enabled = data.get("enabled", cur["enabled"])
        source = validated_source if validated_source is not None else cur["source"]
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
    if auth_enabled() and not is_authenticated(websocket):
        await websocket.close(code=4401)
        return
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
    except (WebSocketDisconnect, ConnectionResetError, OSError):
        pass
    finally:
        with st._ws_lock:
            st._ws_clients.discard(websocket)


stream_routes = [
    Route("/api/frame/{slot_id:int}", api_frame, methods=["GET"]),
    Route("/api/stream/{slot_id:int}", api_stream, methods=["GET"]),
    Route("/api/slot/{slot_id:int}/upload_video", api_slot_upload_video, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/browser_register", api_slot_browser_register, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/push_frame", api_slot_push_frame, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/configure", api_slot_configure, methods=["POST"]),
    Route("/api/slot/{slot_id:int}/remove", api_slot_remove, methods=["POST"]),
    WebSocketRoute("/ws/browser_source", ws_browser_source),
]
