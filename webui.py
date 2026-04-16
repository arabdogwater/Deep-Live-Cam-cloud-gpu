#!/usr/bin/env python3
"""
webui.py — FastAPI web UI for Deep-Live-Cam
Replaces the tkinter + noVNC pipeline with a proper browser-native interface.
Usage: python webui.py --execution-provider cuda [--port 8080]
"""

import os
import sys
import types
import shutil
import threading
import queue
import time
import asyncio
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Stub tkinter-dependent modules BEFORE any project code imports them ────────
# modules.core imports modules.ui (customtkinter/tkinter). We replace it with a
# lightweight stub so the full processing pipeline works headlessly.
_ui_stub = types.ModuleType("modules.ui")
_ui_stub.update_status = lambda msg, scope="": None
_ui_stub.check_and_ignore_nsfw = lambda path, fn: False
_ui_stub.POPUP = None
_ui_stub.POPUP_LIVE = None

class _NoopTip:
    def __init__(self, *a, **kw): pass

_tt_stub = types.ModuleType("modules.ui_tooltip")
_tt_stub.ToolTip = _NoopTip

sys.modules.setdefault("modules.ui", _ui_stub)
sys.modules.setdefault("modules.ui_tooltip", _tt_stub)

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Project modules ────────────────────────────────────────────────────────────
import modules.globals as G
import modules.metadata as META
import modules.core as _core
from modules.utilities import (
    is_image, is_video, has_image_extension,
    detect_fps, create_temp, extract_frames,
    get_temp_frame_paths, create_video,
    restore_audio, move_temp, clean_temp,
)
from modules.core import (
    pre_check, limit_resources, release_resources,
    decode_execution_providers, suggest_execution_providers,
)
from modules.processors.frame.core import get_frame_processors_modules, process_video_in_memory
from modules.face_analyser import get_one_face, get_many_faces

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent
_WS      = Path(os.environ.get("WORKSPACE", "/workspace"))
UPLOADS  = _WS / ".dlc_uploads"
OUTPUTS  = _WS / ".dlc_outputs"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── Initial globals ────────────────────────────────────────────────────────────
G.headless         = True
G.frame_processors = ["face_swapper"]
G.keep_fps         = True
G.keep_audio       = True
G.keep_frames      = False
G.many_faces       = False
G.map_faces        = False
G.poisson_blend    = False
G.color_correction = False
G.live_mirror      = False
G.show_fps         = False
G.opacity          = 1.0
G.sharpness        = 0.0
G.mouth_mask_size  = 0.0
G.mouth_mask       = False
G.execution_threads = 2

# ── App state ──────────────────────────────────────────────────────────────────
_st = {
    "status":   "Ready",
    "progress": 0.0,
    "busy":     False,
    "live":     False,
}

def _emit(msg: str, pct: float = -1.0):
    _st["status"] = msg
    if pct >= 0.0:
        _st["progress"] = pct
    print(f"[DLC] {msg}")

# Patch core's update_status so processing progress flows into our state
_core.update_status = lambda msg, scope="DLC.CORE": _emit(msg)

# ── Live stream state ──────────────────────────────────────────────────────────
_live_stop   = threading.Event()
_live_q_out: queue.Queue = queue.Queue(maxsize=2)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Deep Live Cam", docs_url=None, redoc_url=None)

_STATIC = _BASE / "static"
_STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    p = _STATIC / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text("utf-8"))
    return HTMLResponse("<h1>static/index.html not found</h1>", status_code=500)

# ── Upload endpoints ───────────────────────────────────────────────────────────
@app.post("/api/upload/source")
async def upload_source(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        raise HTTPException(400, "Source must be an image file")
    dest = UPLOADS / f"source{ext}"
    dest.write_bytes(await file.read())
    G.source_path = str(dest)
    return {"ok": True}

@app.post("/api/upload/target")
async def upload_target(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".mkv", ".avi", ".mov", ".webm"}
    if ext not in allowed:
        raise HTTPException(400, "Unsupported format")
    dest = UPLOADS / f"target{ext}"
    dest.write_bytes(await file.read())
    G.target_path  = str(dest)
    G.output_path  = str(OUTPUTS / f"output{ext}")
    kind = "image" if is_image(str(dest)) else "video"
    return {"ok": True, "type": kind}

# ── Preview endpoints ──────────────────────────────────────────────────────────
@app.get("/api/preview/source")
def preview_source():
    if not G.source_path or not Path(G.source_path).exists():
        raise HTTPException(404)
    return FileResponse(G.source_path)

@app.get("/api/preview/target")
def preview_target():
    if not G.target_path or not Path(G.target_path).exists():
        raise HTTPException(404)
    if is_video(G.target_path):
        cap = cv2.VideoCapture(G.target_path)
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(500, "Cannot read video frame")
        _, j = cv2.imencode(".jpg", f)
        return StreamingResponse(iter([j.tobytes()]), media_type="image/jpeg")
    return FileResponse(G.target_path)

@app.get("/api/preview/output")
def preview_output():
    if not G.output_path or not Path(G.output_path).exists():
        raise HTTPException(404)
    if is_video(G.output_path):
        cap = cv2.VideoCapture(G.output_path)
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(500)
        _, j = cv2.imencode(".jpg", f)
        return StreamingResponse(iter([j.tobytes()]), media_type="image/jpeg")
    return FileResponse(G.output_path)

# ── Settings ───────────────────────────────────────────────────────────────────
_BOOL_SETTINGS  = {
    "keep_fps", "keep_audio", "keep_frames", "many_faces",
    "map_faces", "poisson_blend", "color_correction",
    "live_mirror", "show_fps",
}
_FLOAT_SETTINGS = {"opacity", "sharpness", "mouth_mask_size"}
_ENHANCERS      = {"face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"}

@app.get("/api/settings")
def get_settings():
    return {
        **{k: getattr(G, k) for k in _BOOL_SETTINGS},
        **{k: getattr(G, k) for k in _FLOAT_SETTINGS},
        "face_enhancer":         G.fp_ui.get("face_enhancer", False),
        "face_enhancer_gpen256": G.fp_ui.get("face_enhancer_gpen256", False),
        "face_enhancer_gpen512": G.fp_ui.get("face_enhancer_gpen512", False),
        "execution_providers":   G.execution_providers,
    }

@app.post("/api/settings")
async def post_settings(req: Request):
    body = await req.json()
    for k, v in body.items():
        if k in _BOOL_SETTINGS:
            setattr(G, k, bool(v))
        elif k in _FLOAT_SETTINGS:
            setattr(G, k, float(v))
        elif k in _ENHANCERS:
            G.fp_ui[k] = bool(v)
    G.mouth_mask = G.mouth_mask_size > 0
    return {"ok": True}

# ── Camera list ────────────────────────────────────────────────────────────────
@app.get("/api/cameras")
def list_cameras():
    cams = []
    for i in range(12):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cams.append({"index": i, "name": f"Camera {i}"})
        cap.release()
    return {"cameras": cams}

# ── Static processing ──────────────────────────────────────────────────────────
def _bg_process():
    _st["busy"]     = True
    _st["progress"] = 0.0
    try:
        limit_resources()
        if not G.source_path or not Path(G.source_path).exists():
            return _emit("Error: no source face uploaded")
        if not G.target_path or not Path(G.target_path).exists():
            return _emit("Error: no target uploaded")
        if not G.frame_processors:
            G.frame_processors = ["face_swapper"]
        _emit("Initializing...", 0.02)
        _core.start()  # full pipeline; update_status is patched to feed _emit
    except Exception as e:
        _emit(f"Error: {e}")
    finally:
        _st["busy"] = False

@app.post("/api/process/start")
def api_start():
    if _st["busy"]:
        raise HTTPException(409, "Already processing")
    threading.Thread(target=_bg_process, daemon=True).start()
    return {"ok": True}

@app.post("/api/process/stop")
def api_stop():
    _st["busy"] = False
    return {"ok": True}

# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def api_status():
    return {
        "status":     _st["status"],
        "progress":   _st["progress"],
        "busy":       _st["busy"],
        "live":       _st["live"],
        "has_output": bool(G.output_path and Path(G.output_path).exists()),
        "has_source": bool(G.source_path and Path(G.source_path).exists()),
        "has_target": bool(G.target_path and Path(G.target_path).exists()),
    }

# ── Download processed output ──────────────────────────────────────────────────
@app.get("/api/download")
def download():
    if not G.output_path or not Path(G.output_path).exists():
        raise HTTPException(404, "No output available yet")
    return FileResponse(
        G.output_path,
        filename=Path(G.output_path).name,
        media_type="application/octet-stream",
    )

# ── Random face ────────────────────────────────────────────────────────────────
@app.post("/api/random-face")
def random_face():
    from urllib.request import urlopen, Request as _Req
    from urllib.error import URLError
    try:
        req = _Req(
            "https://thispersondoesnotexist.com",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urlopen(req, timeout=10) as resp:
            data = resp.read()
        dest = UPLOADS / "source.jpg"
        dest.write_bytes(data)
        G.source_path = str(dest)
        return {"ok": True}
    except (URLError, Exception) as e:
        raise HTTPException(500, f"Failed to fetch random face: {e}")

# ── Live MJPEG streaming ───────────────────────────────────────────────────────
def _cap_thread(idx: int, cap_q: queue.Queue, stop: threading.Event):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        stop.set()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            stop.set()
            break
        try:
            cap_q.put_nowait(frame)
        except queue.Full:
            try:
                cap_q.get_nowait()
            except queue.Empty:
                pass
            try:
                cap_q.put_nowait(frame)
            except queue.Full:
                pass
    cap.release()

def _proc_thread(cap_q: queue.Queue, out_q: queue.Queue, stop: threading.Event):
    try:
        _proc_thread_inner(cap_q, out_q, stop)
    except Exception as e:
        print(f"[DLC] proc_thread crashed: {e}")
    finally:
        # Always clean up live state so the frontend isn't left stuck
        _st["live"] = False
        stop.set()

def _proc_thread_inner(cap_q: queue.Queue, out_q: queue.Queue, stop: threading.Event):
    fps_procs  = get_frame_processors_modules(G.frame_processors)
    src_img    = None
    last_src   = None
    cached_face = None
    det_count  = 0
    prev_time  = time.time()
    frame_count = 0
    fps_display = 0.0

    while not stop.is_set():
        try:
            frame = cap_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if G.live_mirror:
            frame = cv2.flip(frame, 1)

        if G.source_path and G.source_path != last_src:
            last_src = G.source_path
            img = cv2.imread(G.source_path)
            src_img = get_one_face(img) if img is not None else None

        det_count += 1
        if det_count % 3 == 0:
            cached_face = (
                get_many_faces(frame) if G.many_faces
                else get_one_face(frame)
            )

        for fp in fps_procs:
            if fp.NAME == "DLC.FACE-SWAPPER":
                if src_img is not None:
                    if G.many_faces and isinstance(cached_face, list):
                        for cf in cached_face:
                            frame = fp.swap_face(src_img, cf, frame)
                    elif cached_face is not None and not isinstance(cached_face, list):
                        frame = fp.swap_face(src_img, cached_face, frame)
                    frame = fp.apply_post_processing(frame, [])
            elif fp.NAME == "DLC.FACE-ENHANCER" and G.fp_ui.get("face_enhancer"):
                frame = fp.process_frame(None, frame)
            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256" and G.fp_ui.get("face_enhancer_gpen256"):
                frame = fp.process_frame(None, frame)
            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512" and G.fp_ui.get("face_enhancer_gpen512"):
                frame = fp.process_frame(None, frame)

        if G.show_fps:
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now
            cv2.putText(
                frame, f"FPS: {fps_display:.1f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 60), 2,
            )

        try:
            out_q.put_nowait(frame)
        except queue.Full:
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
            try:
                out_q.put_nowait(frame)
            except queue.Full:
                pass

@app.post("/api/live/start")
async def live_start(req: Request):
    global _live_stop, _live_q_out, _raw_stop
    if _st["live"]:
        return {"ok": True, "already": True}
    body = await req.json()
    cam_idx = int(body.get("camera_index", 10))

    if not G.source_path or not Path(G.source_path).exists():
        raise HTTPException(400, "Upload a source face first")
    if not G.frame_processors:
        G.frame_processors = ["face_swapper"]

    # Stop raw preview thread so it releases the camera device before live grabs it
    _raw_stop.set()

    # Pre-warm models in a thread to avoid blocking the event loop
    def _prewarm():
        from modules.processors.frame.face_swapper import get_face_swapper
        from modules.face_analyser import get_face_analyser
        get_face_analyser()
        get_face_swapper()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _prewarm)

    _live_stop  = threading.Event()
    cap_q       = queue.Queue(maxsize=2)
    _live_q_out = queue.Queue(maxsize=2)

    threading.Thread(target=_cap_thread,  args=(cam_idx, cap_q,       _live_stop), daemon=True).start()
    threading.Thread(target=_proc_thread, args=(cap_q,   _live_q_out, _live_stop), daemon=True).start()
    _st["live"] = True
    return {"ok": True}

@app.post("/api/live/stop")
def live_stop():
    global _live_stop
    _live_stop.set()
    _st["live"] = False
    return {"ok": True}

async def _mjpeg_generator(q: queue.Queue, stop: threading.Event):
    loop = asyncio.get_event_loop()
    while not stop.is_set():
        try:
            frame = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
        except Exception:
            if not _st["live"]:
                break
            continue
        ok, j = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + j.tobytes()
                + b"\r\n"
            )

@app.get("/api/live/stream")
async def live_stream():
    if not _st["live"]:
        raise HTTPException(404, "Live mode is not active")
    return StreamingResponse(
        _mjpeg_generator(_live_q_out, _live_stop),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# ── Raw camera preview (no face swap) ─────────────────────────────────────────
_raw_stop: threading.Event = threading.Event()
_raw_q:    queue.Queue     = queue.Queue(maxsize=2)
_raw_idx:  int             = -1

def _raw_cam_thread(idx: int, q: queue.Queue, stop: threading.Event):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        stop.set()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            stop.set()
            break
        try:
            q.put_nowait(frame)
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            try: q.put_nowait(frame)
            except queue.Full: pass
    cap.release()

@app.get("/api/cam/stream")
async def cam_stream(index: int = 10):
    global _raw_stop, _raw_q, _raw_idx
    # restart raw thread if camera changed or stopped
    if _raw_idx != index or _raw_stop.is_set():
        _raw_stop.set()
        _raw_stop = threading.Event()
        _raw_q    = queue.Queue(maxsize=2)
        _raw_idx  = index
        threading.Thread(target=_raw_cam_thread, args=(index, _raw_q, _raw_stop), daemon=True).start()

    # Capture local references so the generator is not affected by later
    # global reassignments (e.g. when a second client connects).
    _local_stop = _raw_stop
    _local_q    = _raw_q

    async def _gen():
        loop = asyncio.get_event_loop()
        while not _local_stop.is_set():
            try:
                frame = await loop.run_in_executor(None, lambda: _local_q.get(timeout=0.5))
            except queue.Empty:
                continue   # camera just slow — keep waiting
            except Exception:
                break      # real error — close stream
            ok, j = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + j.tobytes() + b"\r\n"
                )

    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ── RTSP / virtual cam stream status ──────────────────────────────────────────
@app.get("/api/rtsp-status")
def rtsp_status():
    """Check if the webcam RTSP stream from stream-webcam.bat is active."""
    import urllib.request, json as _json
    # MediaMTX exposes a REST API on port 9997
    try:
        with urllib.request.urlopen("http://localhost:9997/v3/paths/list", timeout=1) as r:
            data = _json.loads(r.read())
        paths = data.get("items", [])
        for p in paths:
            if p.get("name") == "webcam" and p.get("ready"):
                readers = p.get("readersCount", 0)
                return {"connected": True, "readers": readers}
        return {"connected": False}
    except Exception:
        pass
    # Fallback: try to open the virtual cam and grab a frame
    try:
        cap = cv2.VideoCapture(10)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            return {"connected": ok}
        cap.release()
    except Exception:
        pass
    return {"connected": False}

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Live Cam — Web UI")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("OPEN_BUTTON_PORT", 8080)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--execution-provider", dest="ep", nargs="+",
        default=["cuda"], choices=suggest_execution_providers(),
    )
    parser.add_argument("--max-memory", type=int, default=16)
    args = parser.parse_args()

    G.execution_providers = decode_execution_providers(args.ep)
    G.max_memory          = args.max_memory

    if not pre_check():
        sys.exit(1)

    print(f"\n[DLC] Web UI → http://0.0.0.0:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
