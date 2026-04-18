#!/usr/bin/env python3
"""
gpu_server.py — Deep-Live-Cam GPU Processing Server
Runs on the cloud GPU (vast.ai). Handles all AI-heavy face-swap processing.
The local UI connects here via WebSocket for live frames and HTTP for uploads.

Usage (on GPU):  python gpu_server.py --execution-provider cuda
"""

import os
import sys
import types
import threading
import queue
import time
import json
import asyncio
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Optional: libjpeg-turbo for fast server-side JPEG encoding ────────────────
try:
    from turbojpeg import TurboJPEG as _TurboJPEG
    _tj = _TurboJPEG()
    _HAS_TJ = True
except Exception:
    _HAS_TJ = False

# ── Live log fan-out to browser console ───────────────────────────────────────
_LOG_LOCK = threading.Lock()
_LOG_BUFFER = []
# Maps client id → (event_loop, asyncio.Queue) so the producer only needs to
# call `call_soon_threadsafe` once per client instead of scheduling a full
# coroutine for every log line.
_LOG_QUEUES: dict = {}


def _enqueue_log(q: asyncio.Queue, payload: str) -> None:
    """Scheduled on the client's event loop; silently drops if queue is full."""
    if not q.full():
        q.put_nowait(payload)


def _srv_log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    line = f"[GPU {ts}] {msg}"
    print(line, flush=True)

    payload = json.dumps({"type": "log", "level": level, "ts": ts, "message": msg})
    with _LOG_LOCK:
        _LOG_BUFFER.append(payload)
        if len(_LOG_BUFFER) > 200:
            del _LOG_BUFFER[:-200]
        clients = list(_LOG_QUEUES.values())

    for loop, q in clients:
        try:
            loop.call_soon_threadsafe(_enqueue_log, q, payload)
        except Exception:
            pass

# ── Optional: PyAV for H.264 decoding (WebCodecs input) ───────────────────────
try:
    import av as _pyav
    _HAS_AV = True
except ImportError:
    _HAS_AV = False
    _srv_log("PyAV not found — H.264 input disabled (pip install av)", "warn")

class _H264Decoder:
    """Decode raw H.264 annexb chunks produced by browser WebCodecs VideoEncoder."""
    def __init__(self):
        self._codec = _pyav.CodecContext.create('h264', 'r')
        self._codec.thread_type = 'AUTO'
        self._got_keyframe = False
        self._err_count = 0
        self._ok_count  = 0

    def decode(self, data: bytes, is_keyframe: bool = False) -> Optional[np.ndarray]:
        if is_keyframe:
            self._got_keyframe = True
            # Reset decoder on keyframes to clear any corrupt state
            self._codec = _pyav.CodecContext.create('h264', 'r')
            self._codec.thread_type = 'AUTO'
        if not self._got_keyframe:
            return None
        try:
            pkt = _pyav.Packet(data)
            for f in self._codec.decode(pkt):
                self._ok_count += 1
                return f.to_ndarray(format='bgr24')
        except Exception as e:
            self._err_count += 1
            if self._err_count <= 5 or self._err_count % 100 == 0:
                _srv_log(
                    f"H264 decode error #{self._err_count} (ok={self._ok_count}): {e}",
                    "warn",
                )
        return None

# ── Stub tkinter-dependent modules BEFORE any project code imports them ────────
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
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, HTTPException, Request,
)
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Project modules ────────────────────────────────────────────────────────────
import modules.globals as G
import modules.metadata as META
import modules.core as _core
from modules.utilities import is_image, is_video
from modules.core import (
    pre_check, limit_resources, release_resources,
    decode_execution_providers, suggest_execution_providers,
)
from modules.processors.frame.core import get_frame_processors_modules
from modules.face_analyser import get_one_face, get_many_faces

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE   = Path(__file__).parent
_WS     = Path(os.environ.get("WORKSPACE", "/workspace"))
UPLOADS = _WS / ".dlc_uploads"
OUTPUTS = _WS / ".dlc_outputs"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── Initial globals ────────────────────────────────────────────────────────────
G.headless          = True
G.frame_processors  = ["face_swapper"]
G.keep_fps          = True
G.keep_audio        = True
G.keep_frames       = False
G.many_faces        = False
G.map_faces         = False
G.poisson_blend     = False
G.color_correction  = False
G.live_mirror       = False
G.show_fps          = False
G.opacity           = 1.0
G.sharpness         = 0.0
G.mouth_mask_size   = 0.0
G.mouth_mask        = False
G.execution_threads = 2

# ── App state ──────────────────────────────────────────────────────────────────
_st = {
    "status":   "Ready",
    "progress": 0.0,
    "busy":     False,
}

def _emit(msg: str, pct: float = -1.0):
    _st["status"] = msg
    if pct >= 0.0:
        _st["progress"] = pct
    _srv_log(msg)

_core.update_status = lambda msg, scope="DLC.CORE": _emit(msg)

# ── Model warmup ──────────────────────────────────────────────────────────────
_models_ready = False


def _first_face_or_none(frame):
    if frame is None:
        return None
    face = get_one_face(frame)
    if face is not None:
        return face
    faces = get_many_faces(frame) or []
    return faces[0] if isinstance(faces, list) and faces else None


def _prewarm_models():
    global _models_ready
    if _models_ready:
        return
    from modules.processors.frame.face_swapper import get_face_swapper
    from modules.face_analyser import get_face_analyser
    get_face_analyser()
    get_face_swapper()
    _models_ready = True
    _srv_log("Models prewarmed and ready")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Deep Live Cam GPU", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

@app.on_event("startup")
async def _startup():
    threading.Thread(target=_prewarm_models, daemon=True).start()

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "models_ready": _models_ready}

# ── Upload endpoints ───────────────────────────────────────────────────────────
@app.post("/api/upload/source")
async def upload_source(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        raise HTTPException(400, "Source must be an image file")
    dest = UPLOADS / f"source{ext}"
    dest.write_bytes(await file.read())
    G.source_path = str(dest)
    _srv_log(f"Source uploaded: {dest.name}")
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
    _srv_log(f"Target uploaded: {dest.name} ({kind})")
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
    "live_mirror", "show_fps", "mouth_mask",
}
_FLOAT_SETTINGS = {"opacity", "sharpness", "mouth_mask_size"}
_ENHANCERS      = {"face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"}

def _apply_settings(body: dict):
    for k, v in body.items():
        if k in _BOOL_SETTINGS:
            setattr(G, k, bool(v))
        elif k in _FLOAT_SETTINGS:
            setattr(G, k, float(v))
        elif k in _ENHANCERS:
            G.fp_ui[k] = bool(v)
    # Auto-enable mouth_mask from slider only if the toggle wasn't explicitly sent
    if "mouth_mask" not in body:
        G.mouth_mask = G.mouth_mask_size > 0
    # Rebuild frame_processors so enhancers actually get loaded
    procs = ["face_swapper"]
    for enh in _ENHANCERS:
        if G.fp_ui.get(enh, False):
            procs.append(enh)
    G.frame_processors = procs

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
    _apply_settings(body)
    _srv_log("Live settings updated")
    return {"ok": True}

# ── Static processing (offline) ───────────────────────────────────────────────
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
        _core.start()
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
        "has_output": bool(G.output_path and Path(G.output_path).exists()),
        "has_source": bool(G.source_path and Path(G.source_path).exists()),
        "has_target": bool(G.target_path and Path(G.target_path).exists()),
    }

# ── Download ───────────────────────────────────────────────────────────────────
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
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(req, timeout=10) as resp:
            data = resp.read()
        dest = UPLOADS / "source.jpg"
        dest.write_bytes(data)
        G.source_path = str(dest)
        return {"ok": True}
    except (URLError, Exception) as e:
        raise HTTPException(500, f"Failed to fetch random face: {e}")

# ── WebSocket: live face-swap processing ───────────────────────────────────────
@app.websocket("/ws")
async def ws_live(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    client_id = id(ws)

    # Per-client log queue: producer puts payloads in; a dedicated coroutine
    # drains it so _srv_log never blocks on a slow or stalled browser connection.
    log_q: asyncio.Queue = asyncio.Queue(maxsize=200)

    with _LOG_LOCK:
        _LOG_QUEUES[client_id] = (loop, log_q)
        backlog = list(_LOG_BUFFER)

    for payload in backlog[-80:]:
        try:
            await ws.send_text(payload)
        except Exception:
            break

    _srv_log("Live websocket connected")

    # Wait for models if still warming up
    if not _models_ready:
        await ws.send_text(json.dumps({"type": "status", "message": "Loading AI models..."}))
        while not _models_ready:
            await asyncio.sleep(0.5)

    await ws.send_text(json.dumps({"type": "ready"}))

    # Shared state between receive loop and processing thread
    latest_frame_lock = threading.Lock()
    latest_frame = [None]           # (mode, raw_bytes, is_keyframe)

    # Single-slot output buffer: the processing thread always writes the latest
    # processed frame here; the async sender drains it.  Using deque(maxlen=1)
    # with a lock and an asyncio.Event avoids the race condition that plagued
    # the old asyncio.Queue approach (where get_nowait ran in the wrong thread).
    out_slot: deque = deque(maxlen=1)
    out_slot_lock = threading.Lock()
    out_notify = asyncio.Event()

    stop = threading.Event()

    # ── Processing thread ──────────────────────────────────────────────────────
    def _process_loop():
        fps_procs  = get_frame_processors_modules(G.frame_processors)
        last_fp_key = tuple(G.frame_processors)  # track changes to avoid needless reloads
        h264_dec   = _H264Decoder() if _HAS_AV else None
        src_img    = None
        last_src   = None
        cached_face = None
        det_count  = 0
        _proc_count = 0
        _decode_fail = 0
        _log_time  = time.time()
        _last_src_state = None
        _last_face_state = None

        while not stop.is_set():
            with latest_frame_lock:
                item = latest_frame[0]
                latest_frame[0] = None

            if item is None:
                time.sleep(0.003)
                continue

            mode, raw, is_kf = item

            if mode == 'jpeg':
                nparr = np.frombuffer(raw, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    _decode_fail += 1
                    continue
            elif mode == 'h264':
                if h264_dec is None:
                    continue
                frame = h264_dec.decode(raw, is_keyframe=is_kf)
                if frame is None:
                    _decode_fail += 1
                    continue
            else:
                continue

            if G.live_mirror:
                frame = cv2.flip(frame, 1)

            # Re-read source face when changed
            if G.source_path and G.source_path != last_src:
                last_src = G.source_path
                img = cv2.imread(G.source_path)
                src_img = _first_face_or_none(img)
                src_state = src_img is not None
                if src_state != _last_src_state:
                    _srv_log(
                        "Source face detected and ready for swapping" if src_state
                        else "No face found in uploaded source image",
                        "info" if src_state else "warn",
                    )
                    _last_src_state = src_state

            det_count += 1
            # Detect on first frame, then every 3rd frame thereafter.
            # Preserve cached_face when detection returns nothing.
            if cached_face is None or det_count % 3 == 0:
                result = (
                    get_many_faces(frame) if G.many_faces
                    else _first_face_or_none(frame)
                )
                if result is not None:
                    cached_face = result

                face_state = bool(result) if G.many_faces else (cached_face is not None)
                if face_state != _last_face_state:
                    _srv_log(
                        "Live target face detected in camera feed" if face_state
                        else "No target face detected in camera frame",
                        "info" if face_state else "warn",
                    )
                    _last_face_state = face_state

            # Reload processor modules only when the list actually changed
            current_fp_key = tuple(G.frame_processors)
            if current_fp_key != last_fp_key:
                fps_procs = get_frame_processors_modules(G.frame_processors)
                last_fp_key = current_fp_key

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

            _proc_count += 1
            now = time.time()
            if now - _log_time >= 5.0:
                dt = now - _log_time
                fps = _proc_count / dt
                _srv_log(
                    f"Live: {fps:.1f} fps out | decode_fail={_decode_fail} | "
                    f"src={'yes' if src_img is not None else 'NO'} | "
                    f"face={'yes' if cached_face is not None else 'NO'}"
                )
                _proc_count = 0
                _decode_fail = 0
                _log_time = now

            # Encode output frame — libjpeg-turbo when available, else OpenCV
            if _HAS_TJ:
                try:
                    data = _tj.encode(frame, quality=85)
                except Exception:
                    continue
            else:
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                data = jpg.tobytes()

            # Deposit the latest frame in the single-slot output buffer and
            # wake the async sender.  deque(maxlen=1) automatically evicts any
            # frame that the sender hasn't yet consumed, so we never accumulate
            # a backlog of stale frames.
            with out_slot_lock:
                out_slot.append(data)
            loop.call_soon_threadsafe(out_notify.set)

    proc_t = threading.Thread(target=_process_loop, daemon=True)
    proc_t.start()

    # ── Log sender: drains the per-client log queue ────────────────────────────
    async def _log_sender():
        while not stop.is_set():
            try:
                payload = await asyncio.wait_for(log_q.get(), timeout=0.1)
                await ws.send_text(payload)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    log_sender_task = asyncio.create_task(_log_sender())

    # ── Frame sender: reads from the single-slot output buffer ─────────────────
    async def _sender():
        while not stop.is_set():
            try:
                await asyncio.wait_for(out_notify.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            out_notify.clear()
            with out_slot_lock:
                if out_slot:
                    data = out_slot.popleft()
                else:
                    continue
            try:
                await ws.send_bytes(data)
            except Exception:
                break

    sender_task = asyncio.create_task(_sender())

    # ── Receive loop ───────────────────────────────────────────────────────────
    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                # Binary frame: 1-byte type header
                # 0x00 = JPEG (fallback), 0x01 = H264 keyframe, 0x02 = H264 delta
                data = bytes(msg["bytes"])
                if data:
                    ftype   = data[0]
                    payload = data[1:]
                    if ftype == 0x00:
                        with latest_frame_lock:
                            latest_frame[0] = ('jpeg', payload, False)
                    elif ftype in (0x01, 0x02):
                        with latest_frame_lock:
                            latest_frame[0] = ('h264', payload, ftype == 0x01)

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                msg_type = data.get("type")

                if msg_type == "settings":
                    _apply_settings(data.get("data", {}))

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": data.get("ts")}))

    except WebSocketDisconnect:
        _srv_log("Live websocket disconnected", "warn")
    except Exception as e:
        _srv_log(f"WebSocket error: {e}", "warn")
    finally:
        with _LOG_LOCK:
            _LOG_QUEUES.pop(client_id, None)
        stop.set()
        log_sender_task.cancel()
        sender_task.cancel()
        try:
            await log_sender_task
        except asyncio.CancelledError:
            pass
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Live Cam — GPU Server")
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

    # Pre-bind so uvicorn adopts an already-open socket (no bind race).
    # If the requested port is occupied, fall back to port 0 and let the OS
    # assign any free port — never crash over a port conflict.
    import socket as _socket
    _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        _sock.bind((args.host, args.port))
    except OSError:
        _sock.bind((args.host, 0))          # let OS pick a free port
        args.port = _sock.getsockname()[1]
    _sock.set_inheritable(True)

    _srv_log(f"Server ready → http://0.0.0.0:{args.port}/")
    uvicorn.run(app, fd=_sock.fileno(), log_level="warning")
