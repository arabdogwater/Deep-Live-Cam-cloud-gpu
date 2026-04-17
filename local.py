#!/usr/bin/env python3
"""
local.py — Deep-Live-Cam Local Runner

Single entry-point for the local machine. Run with just:
    python local.py

It will:
  1. Auto-install any missing Python dependencies
  2. Serve the web UI on http://localhost:8080
  3. Proxy WebSocket connections to the GPU server (browser <-> GPU)
  4. Intercept face-swapped frames and feed them to a Windows virtual camera
     (OBS VirtualCam driver — install OBS Studio: https://obsproject.com)

Virtual camera settings (resolution, FPS) are configurable live from the UI.
The real camera feed is NEVER written to the virtual camera — only swapped frames.
On disconnect the virtual camera outputs solid black.

Usage:
    python local.py [--port 8080]
"""

# ── Step 1: Auto-install all dependencies ──────────────────────────────────────
import sys
import subprocess

_REQUIRED = [
    ("fastapi",        "fastapi>=0.111.0"),
    ("uvicorn",        "uvicorn[standard]>=0.29.0"),
    ("websockets",     "websockets>=12.0"),
    ("cv2",            "opencv-python>=4.8.0"),
    ("numpy",          "numpy>=1.23.5,<2"),
    ("pyvirtualcam",   "pyvirtualcam"),
]

def _ensure_deps():
    missing = []
    for (import_name, pip_spec) in _REQUIRED:
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print("  [setup] Done.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n  [setup] pip install failed:\n{e.stderr.decode()}")
        print("  Some features may not work. Install manually:")
        for pkg in missing:
            print(f"    pip install {pkg}")
        print()

    # pyvirtualcam needs OBS driver — warn if pyvirtualcam still missing after install
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        print("  [setup] pyvirtualcam installed but no virtual camera driver found.")
        print("          Install OBS Studio to get the OBS-VirtualCam driver:")
        print("          https://obsproject.com")
        print()

_ensure_deps()

# ── Step 2: Normal imports (all guaranteed to be installed now) ────────────────
import argparse
import asyncio
import json
import os
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

import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ───────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_STATIC = _HERE / "static"

# ── Virtual cam config (mutable, applied on next restart of vcam thread) ────────
_vcam_cfg_lock = threading.Lock()
_vcam_cfg = {
    "width":   1280,
    "height":  720,
    "fps":     30,
    "enabled": _HAS_VCAM,
}

# ── Shared frame state ───────────────────────────────────────────────────────────
_vcam_state_lock = threading.Lock()
_vcam_frame_rgb  = None   # latest swapped frame (numpy RGB uint8)
_vcam_frame_ts   = 0.0    # time.monotonic() when it arrived
_vcam_connected  = False  # True while a GPU WS proxy is active

# ── Signals to restart the vcam thread when settings change ─────────────────────
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


# ── Virtual camera supervisor (restarts inner thread when settings change) ───────
def _vcam_supervisor() -> None:
    """Outer loop: starts/restarts the actual pyvirtualcam loop when cfg changes."""
    while True:
        with _vcam_cfg_lock:
            cfg = dict(_vcam_cfg)

        if not cfg["enabled"] or not _HAS_VCAM:
            _vcam_restart_event.wait(timeout=2.0)
            _vcam_restart_event.clear()
            continue

        _vcam_restart_event.clear()
        _run_vcam_once(cfg["width"], cfg["height"], cfg["fps"])
        # _run_vcam_once exits either on error or when _vcam_restart_event fires


def _run_vcam_once(width: int, height: int, fps: int) -> None:
    """Inner loop: opens the virtual camera and pumps frames until told to restart."""
    black = np.zeros((height, width, 3), dtype=np.uint8)
    stale_limit = 1.5  # seconds before showing black

    try:
        with pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
        ) as cam:
            print(f"  [vcam] Active: {cam.device}  ({width}x{height} @ {fps}fps)")

            while not _vcam_restart_event.is_set():
                with _vcam_state_lock:
                    frame = _vcam_frame_rgb
                    age   = time.monotonic() - _vcam_frame_ts
                    conn  = _vcam_connected

                if frame is None or not conn or age > stale_limit:
                    out = black
                else:
                    if frame.shape[1] != width or frame.shape[0] != height:
                        out = cv2.resize(frame, (width, height))
                    else:
                        out = frame

                cam.send(out)
                cam.sleep_until_next_frame()

        print("  [vcam] Closed (settings changed or restart requested)")

    except Exception as exc:
        msg = str(exc).lower()
        print(f"\n  [vcam] Error: {exc}")
        if any(k in msg for k in ("no virtual camera", "no device", "failed to open",
                                   "obs", "not found")):
            print("  [vcam] No virtual camera driver found.")
            print("         Install OBS Studio: https://obsproject.com")
        # non-fatal; supervisor will retry after a delay
        time.sleep(3.0)


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ──────────────────────────────────────────────────────────
@app.get("/")
def serve_index():
    return FileResponse(_STATIC / "index.html")


@app.get("/{path:path}")
def serve_static(path: str):
    # Skip API/WebSocket routes so they don't get caught here
    if path.startswith(("api/", "proxy")):
        return Response(status_code=404)
    target = (_STATIC / path).resolve()
    try:
        target.relative_to(_STATIC.resolve())
    except ValueError:
        return Response(status_code=403)
    if target.exists() and target.is_file():
        return FileResponse(target)
    return Response(status_code=404)


# ── Virtual cam settings API ─────────────────────────────────────────────────────
@app.get("/api/vcam")
def get_vcam():
    with _vcam_cfg_lock:
        return JSONResponse(_vcam_cfg)


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

    if changed:
        _vcam_restart_event.set()   # tell supervisor to apply new settings
        print(f"  [vcam] Settings updated: {_vcam_cfg}")

    with _vcam_cfg_lock:
        return JSONResponse({"ok": True, **_vcam_cfg})


# ── WebSocket proxy ──────────────────────────────────────────────────────────────
@app.websocket("/proxy")
async def ws_proxy(browser_ws: WebSocket) -> None:
    """
    Bridges browser <-> GPU server WS.
    Query param: ?url=<gpu_ws_url>

    - Forwards browser frames (camera data, settings, pings) to GPU unchanged
    - Forwards GPU text messages (log, status, pong) to browser unchanged
    - Intercepts GPU binary (JPEG) frames → push to virtual cam, then forward to browser
    - Real webcam data flows browser→GPU only, NEVER touches the virtual cam
    """
    gpu_url: str = browser_ws.query_params.get("url", "")
    if not gpu_url:
        await browser_ws.close(code=4000)
        return

    await browser_ws.accept()
    loop = asyncio.get_running_loop()

    async def _browser_to_gpu(gpu_ws) -> None:
        try:
            while True:
                msg = await browser_ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    await gpu_ws.send(bytes(msg["bytes"]))
                elif "text" in msg and msg["text"]:
                    await gpu_ws.send(msg["text"])
        except Exception:
            pass
        finally:
            try:
                await gpu_ws.close()
            except Exception:
                pass

    async def _gpu_to_browser(gpu_ws) -> None:
        try:
            async for message in gpu_ws:
                if isinstance(message, bytes):
                    # Push to vcam in thread-pool (non-blocking)
                    await loop.run_in_executor(None, _decode_and_push, message)
                    try:
                        await browser_ws.send_bytes(message)
                    except Exception:
                        break
                else:
                    try:
                        await browser_ws.send_text(message)
                    except Exception:
                        break
        except Exception:
            pass

    ssl_ctx = None
    if gpu_url.startswith("wss://"):
        import ssl as _ssl
        ssl_ctx = _ssl.create_default_context()

    try:
        async with websockets.connect(
            gpu_url,
            ssl=ssl_ctx,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        ) as gpu_ws:
            _set_vcam_connected(True)
            await asyncio.gather(
                _browser_to_gpu(gpu_ws),
                _gpu_to_browser(gpu_ws),
                return_exceptions=True,
            )
    except Exception as exc:
        try:
            await browser_ws.send_text(
                json.dumps({"type": "error", "message": f"Proxy error: {exc}"})
            )
        except Exception:
            pass
    finally:
        _set_vcam_connected(False)
        try:
            await browser_ws.close()
        except Exception:
            pass


def _decode_and_push(data: bytes) -> None:
    """Decode a JPEG frame and push RGB to virtual cam. Runs in thread-pool."""
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        _push_vcam_frame(frame)


# ── Entry point ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-Live-Cam Local Runner")
    parser.add_argument("--port", type=int, default=8080, help="UI port (default 8080)")
    args = parser.parse_args()

    # Start virtual cam supervisor daemon
    sup = threading.Thread(target=_vcam_supervisor, daemon=True)
    sup.start()

    url = f"http://localhost:{args.port}/"
    vcam_status = (
        "OBS VirtualCam ready  (configure in UI)"
        if _HAS_VCAM
        else "DISABLED — install OBS Studio: https://obsproject.com"
    )

    print()
    print("  ╔═════════════════════════════════════════════════════════╗")
    print("  ║           Deep Live Cam — Local Runner                  ║")
    print("  ╠═════════════════════════════════════════════════════════╣")
    print(f"  ║  UI     :  {url:<47}║")
    print(f"  ║  VirtCam:  {vcam_status:<47}║")
    print("  ║  Ctrl+C :  quit                                         ║")
    print("  ╚═════════════════════════════════════════════════════════╝")
    print()

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
        ws_ping_interval=None,
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
