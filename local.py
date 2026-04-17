#!/usr/bin/env python3
"""
local.py — Deep-Live-Cam Local Runner

Single entry-point.  Just run:
    python local.py

It will:
  1. Auto-install any missing Python dependencies
  2. Serve the web UI on http://localhost:8080
  3. Accept face-swapped JPEG frames from the browser via /vcam-feed WebSocket
     and write them to a Windows virtual camera (OBS VirtualCam)
  4. The real webcam feed is NEVER written to the virtual camera
  5. Virtual cam outputs solid black when disconnected / no frame within 1.5 s

Virtual camera settings (resolution, FPS, device) are configurable live from
the UI — no restart needed.

Requires OBS Studio for the virtual camera driver:
    https://obsproject.com
"""

# ── Step 1: Auto-install all dependencies ──────────────────────────────────────
import sys
import subprocess

_REQUIRED = [
    ("fastapi",      "fastapi>=0.111.0"),
    ("uvicorn",      "uvicorn[standard]>=0.29.0"),
    ("cv2",          "opencv-python>=4.8.0"),
    ("numpy",        "numpy>=1.23.5,<2"),
    ("pyvirtualcam", "pyvirtualcam"),
]


def _ensure_deps():
    missing = []
    for import_name, pip_spec in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_spec)

    if not missing:
        return

    print(f"\n  [setup] Installing {len(missing)} missing package(s)...")
    for pkg in missing:
        print(f"          pip install {pkg}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
        )
        print("  [setup] Done.\n")
    except subprocess.CalledProcessError:
        print("  [setup] pip install failed. Install manually:")
        for pkg in missing:
            print(f"    pip install {pkg}")
        print()

    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        print("  [vcam] pyvirtualcam needs the OBS VirtualCam driver.")
        print("         Install OBS Studio: https://obsproject.com\n")


_ensure_deps()

# ── Step 2: Normal imports ──────────────────────────────────────────────────────
import argparse
import asyncio
import json
import signal
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np

try:
    import pyvirtualcam
    _HAS_VCAM = True
except ImportError:
    _HAS_VCAM = False

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ───────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_STATIC = _HERE / "static"

# ── Virtual cam config ───────────────────────────────────────────────────────────
_vcam_cfg_lock = threading.Lock()
_vcam_cfg: dict = {
    "width":         1280,
    "height":        720,
    "fps":           30,
    "enabled":       _HAS_VCAM,
    "device":        "",        # "" = auto
    "_active_device": None,     # filled by vcam thread, read-only from outside
}

# ── Shared frame state ───────────────────────────────────────────────────────────
_vcam_state_lock = threading.Lock()
_vcam_frame_rgb  = None     # numpy RGB uint8
_vcam_frame_ts   = 0.0      # time.monotonic() of last frame
_vcam_connected  = False    # True while browser vcam-feed WS is open

# ── Signal to restart the vcam thread when settings change ──────────────────────
_vcam_restart_event = threading.Event()


def _push_vcam_frame(bgr: np.ndarray) -> None:
    global _vcam_frame_rgb, _vcam_frame_ts
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with _vcam_state_lock:
        _vcam_frame_rgb = rgb
        _vcam_frame_ts  = time.monotonic()


def _set_vcam_connected(v: bool) -> None:
    global _vcam_connected
    with _vcam_state_lock:
        _vcam_connected = v


# ── Virtual camera supervisor thread ────────────────────────────────────────────
def _vcam_supervisor() -> None:
    while True:
        with _vcam_cfg_lock:
            cfg = dict(_vcam_cfg)

        if not cfg["enabled"] or not _HAS_VCAM:
            _vcam_restart_event.wait(timeout=2.0)
            _vcam_restart_event.clear()
            continue

        _vcam_restart_event.clear()
        _run_vcam_once(cfg["width"], cfg["height"], cfg["fps"], cfg.get("device") or None)


def _run_vcam_once(width: int, height: int, fps: int, device=None) -> None:
    black = np.zeros((height, width, 3), dtype=np.uint8)
    stale = 1.5  # seconds before showing black

    kw: dict = dict(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB)
    if device:
        kw["device"] = device.strip()

    try:
        with pyvirtualcam.Camera(**kw) as cam:
            active_name = cam.device.strip()
            print(f"  [vcam] Active: {active_name}  ({width}x{height} @ {fps}fps)")
            with _vcam_cfg_lock:
                _vcam_cfg["_active_device"] = active_name

            while not _vcam_restart_event.is_set():
                with _vcam_state_lock:
                    frame = _vcam_frame_rgb
                    age   = time.monotonic() - _vcam_frame_ts
                    conn  = _vcam_connected

                if frame is None or not conn or age > stale:
                    out = black
                else:
                    if frame.shape[1] != width or frame.shape[0] != height:
                        out = cv2.resize(frame, (width, height))
                    else:
                        out = frame

                cam.send(out)
                cam.sleep_until_next_frame()

        with _vcam_cfg_lock:
            _vcam_cfg["_active_device"] = None
        print("  [vcam] Restarting (settings changed)")

    except Exception as exc:
        with _vcam_cfg_lock:
            _vcam_cfg["_active_device"] = None
        msg = str(exc).lower()
        print(f"  [vcam] Error: {exc}")
        if any(k in msg for k in ("obs", "no virtual", "not found", "failed to open")):
            print("  [vcam] Install OBS Studio: https://obsproject.com")
        time.sleep(3.0)


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket: receive face-swapped frames from browser → virtual cam ────────────
@app.websocket("/vcam-feed")
async def vcam_feed(ws: WebSocket) -> None:
    """
    Browser opens this WS and sends every JPEG frame it receives from the GPU.
    We decode and push to the virtual camera.
    Real webcam data is never sent here — only GPU-processed face-swap output.
    """
    await ws.accept()
    _set_vcam_connected(True)
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                await loop.run_in_executor(None, _decode_and_push, bytes(data))
    except Exception:
        pass
    finally:
        _set_vcam_connected(False)
        try:
            await ws.close()
        except Exception:
            pass


def _decode_and_push(data: bytes) -> None:
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        _push_vcam_frame(frame)


# ── Virtual cam settings API ─────────────────────────────────────────────────────
@app.get("/api/vcam")
def get_vcam():
    with _vcam_cfg_lock:
        return JSONResponse({
            "width":         _vcam_cfg["width"],
            "height":        _vcam_cfg["height"],
            "fps":           _vcam_cfg["fps"],
            "enabled":       _vcam_cfg["enabled"],
            "device":        _vcam_cfg["device"],
            "active_device": _vcam_cfg.get("_active_device"),
            "has_vcam":      _HAS_VCAM,
        })


@app.post("/api/vcam")
async def post_vcam(req: Request):
    body = await req.json()
    changed = False
    with _vcam_cfg_lock:
        for key, cast in [("width", int), ("height", int), ("fps", int), ("enabled", bool)]:
            if key in body:
                new_val = cast(body[key])
                if _vcam_cfg[key] != new_val:
                    _vcam_cfg[key] = new_val
                    changed = True
        if "device" in body:
            new_device = (body["device"] or "").strip()
            if _vcam_cfg["device"] != new_device:
                _vcam_cfg["device"] = new_device
                changed = True

    if changed:
        _vcam_restart_event.set()

    with _vcam_cfg_lock:
        return JSONResponse({
            "ok":            True,
            "width":         _vcam_cfg["width"],
            "height":        _vcam_cfg["height"],
            "fps":           _vcam_cfg["fps"],
            "enabled":       _vcam_cfg["enabled"],
            "device":        _vcam_cfg["device"],
            "active_device": _vcam_cfg.get("_active_device"),
        })


@app.get("/api/vcam/devices")
def list_vcam_devices():
    """
    Return only pyvirtualcam-compatible virtual camera device names.
    We do NOT enumerate DirectShow devices — those include real webcams
    that pyvirtualcam cannot use, which causes the 'unsupported device' error.
    Instead we return the name that pyvirtualcam itself confirmed when it
    opened successfully (stored in _active_device), plus the known OBS name.
    """
    seen: list[str] = []

    # Highest priority: device name pyvirtualcam already opened successfully
    with _vcam_cfg_lock:
        active = _vcam_cfg.get("_active_device")
    if active:
        seen.append(active)

    # Always include the canonical OBS name if not already present
    obs_name = "OBS Virtual Camera"
    if obs_name not in seen:
        seen.append(obs_name)

    devices = [{"name": n} for n in seen]
    return JSONResponse({"devices": devices, "has_vcam": _HAS_VCAM})


# ── Static file serving ──────────────────────────────────────────────────────────
@app.get("/")
def serve_index():
    return FileResponse(_STATIC / "index.html")


@app.get("/{filename:path}")
def serve_static(filename: str):
    if filename.startswith(("api/", "vcam-feed")):
        return Response(status_code=404)
    target = (_STATIC / filename).resolve()
    try:
        target.relative_to(_STATIC.resolve())
    except ValueError:
        return Response(status_code=403)
    if target.exists() and target.is_file():
        return FileResponse(target)
    return Response(status_code=404)


def _find_free_port(start: int) -> int:
    import socket
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + 19}")


# ── Entry point ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-Live-Cam Local Runner")
    parser.add_argument("--port", type=int, default=8080, help="Preferred UI port (default 8080)")
    args = parser.parse_args()

    # Auto-bump port if the preferred one is already taken
    port = _find_free_port(args.port)
    if port != args.port:
        print(f"  [warn] Port {args.port} in use — using {port} instead")
    args.port = port

    threading.Thread(target=_vcam_supervisor, daemon=True).start()

    url = f"http://localhost:{args.port}/"
    vcam_status = (
        "OBS VirtualCam ready  (configure in UI)"
        if _HAS_VCAM
        else "DISABLED — install OBS Studio: https://obsproject.com"
    )

    print()
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║           Deep Live Cam — Local Runner                    ║")
    print("  ╠═══════════════════════════════════════════════════════════╣")
    print(f"  ║  UI     :  {url:<50}║")
    print(f"  ║  VirtCam:  {vcam_status:<50}║")
    print("  ║  Ctrl+C :  quit                                           ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print()

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _on_signal(sig, frame):
        print("\n  Shutting down.")
        server.should_exit = True

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    server.run()


if __name__ == "__main__":
    main()
