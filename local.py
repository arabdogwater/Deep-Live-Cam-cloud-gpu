#!/usr/bin/env python3
"""
local.py — Deep-Live-Cam Local Runner

Single entry-point for the local machine:
  • Serves the web UI (static/) on localhost
  • Proxies WebSocket connections to the GPU server so frames can be intercepted
  • Writes every face-swapped frame to a Windows virtual camera (OBS VirtualCam
    or any pyvirtualcam-compatible backend)
  • Outputs solid-black frames when the GPU connection is lost — the real camera
    feed is NEVER sent to the virtual camera

Requirements (local machine only, no GPU needed):
    pip install fastapi "uvicorn[standard]" websockets opencv-python numpy pyvirtualcam

Windows virtual camera backend (one of):
    • OBS Studio  →  https://obsproject.com  (includes OBS-VirtualCam)
    • Unity Capture  →  https://github.com/schellingb/UnityCapture

Usage:
    python local.py [--port 8080] [--cam-width 1280] [--cam-height 720] [--cam-fps 30]
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np

# ── Virtual camera ──────────────────────────────────────────────────────────────
try:
    import pyvirtualcam
    _HAS_VCAM = True
except ImportError:
    _HAS_VCAM = False

# ── WebSocket client (upstream to GPU) ─────────────────────────────────────────
try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed: pip install websockets")
    sys.exit(1)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ───────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_STATIC = _HERE / "static"

# ── Shared virtual-cam state ────────────────────────────────────────────────────
_vcam_lock      = threading.Lock()
_vcam_frame_rgb = None   # latest swapped frame as numpy uint8 RGB
_vcam_frame_ts  = 0.0    # time.monotonic() when last frame arrived
_vcam_connected = False  # True while a GPU WS is active


def _push_vcam_frame(bgr: np.ndarray) -> None:
    global _vcam_frame_rgb, _vcam_frame_ts
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with _vcam_lock:
        _vcam_frame_rgb = rgb
        _vcam_frame_ts  = time.monotonic()


def _set_vcam_connected(v: bool) -> None:
    global _vcam_connected
    with _vcam_lock:
        _vcam_connected = v


# ── Virtual camera thread ───────────────────────────────────────────────────────
def _vcam_thread(width: int, height: int, fps: int) -> None:
    """
    Runs forever inside a daemon thread.
    Sends the latest swapped frame to the virtual camera at `fps`.
    Falls back to a solid-black frame when:
      - no GPU connection is active, OR
      - no new frame arrived within 1 second (stale / connection lag)
    """
    black = np.zeros((height, width, 3), dtype=np.uint8)  # RGB black
    stale_limit = 1.0  # seconds before switching to black

    try:
        with pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
        ) as cam:
            print(f"  Virtual camera: {cam.device}")

            while True:
                with _vcam_lock:
                    frame   = _vcam_frame_rgb
                    age     = time.monotonic() - _vcam_frame_ts
                    conn    = _vcam_connected

                if frame is None or not conn or age > stale_limit:
                    output = black
                else:
                    # Resize to camera resolution if the GPU frame is a different size
                    if frame.shape[1] != width or frame.shape[0] != height:
                        output = cv2.resize(frame, (width, height))
                    else:
                        output = frame

                cam.send(output)
                cam.sleep_until_next_frame()

    except Exception as exc:
        msg = str(exc).lower()
        print(f"\n[VCAM ERROR] {exc}")
        if any(k in msg for k in ("no virtual camera", "no device", "failed to open")):
            print("[VCAM] Make sure OBS Studio (OBS-VirtualCam) is installed.")
            print("       https://obsproject.com")
        # Virtual cam failure is non-fatal — UI and proxy still work


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    return FileResponse(_STATIC / "index.html")


@app.get("/{path:path}")
def serve_static(path: str):
    target = (_STATIC / path).resolve()
    # Prevent path traversal outside static/
    try:
        target.relative_to(_STATIC.resolve())
    except ValueError:
        return Response(status_code=403)
    if target.exists() and target.is_file():
        return FileResponse(target)
    return Response(status_code=404)


# ── WebSocket proxy ─────────────────────────────────────────────────────────────
@app.websocket("/proxy")
async def ws_proxy(browser_ws: WebSocket) -> None:
    """
    Bridges  browser ←→ GPU server WebSocket.

    The browser connects to  ws://localhost:{port}/proxy?url=<gpu_ws_url>
    This proxy:
      1. Forwards browser frames (camera data, settings, pings) upstream to GPU
      2. Forwards GPU frames back to the browser for display
      3. Intercepts binary JPEG frames from the GPU and pushes them to the
         virtual camera (without ever sending raw camera data to the vcam)
    """
    gpu_url: str = browser_ws.query_params.get("url", "")
    if not gpu_url:
        await browser_ws.close(code=4000)
        return

    await browser_ws.accept()
    loop = asyncio.get_running_loop()

    async def _browser_to_gpu(gpu_ws) -> None:
        """Forward browser → GPU: camera frames, settings, pings."""
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
            # Signal the GPU side to close so _gpu_to_browser exits
            try:
                await gpu_ws.close()
            except Exception:
                pass

    async def _gpu_to_browser(gpu_ws) -> None:
        """Forward GPU → browser; intercept JPEG frames for virtual cam."""
        try:
            async for message in gpu_ws:
                if isinstance(message, bytes):
                    # Decode and push to virtual cam (non-blocking)
                    await loop.run_in_executor(None, _decode_and_push, message)
                    # Forward raw bytes to browser for canvas display
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

    try:
        ssl_ctx = None
        if gpu_url.startswith("wss://"):
            import ssl
            ssl_ctx = ssl.create_default_context()

        async with websockets.connect(
            gpu_url,
            ssl=ssl_ctx,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10 MB frame limit
        ) as gpu_ws:
            _set_vcam_connected(True)
            b2g = asyncio.create_task(_browser_to_gpu(gpu_ws))
            g2b = asyncio.create_task(_gpu_to_browser(gpu_ws))
            await asyncio.gather(b2g, g2b, return_exceptions=True)

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
    """Decode a JPEG frame and push it to the virtual cam. Runs in thread-pool."""
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        _push_vcam_frame(frame)


# ── Entry point ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-Live-Cam Local Runner")
    parser.add_argument("--port",       type=int, default=8080,  help="Local UI port (default 8080)")
    parser.add_argument("--cam-width",  type=int, default=1280,  help="Virtual camera width")
    parser.add_argument("--cam-height", type=int, default=720,   help="Virtual camera height")
    parser.add_argument("--cam-fps",    type=int, default=30,    help="Virtual camera FPS")
    args = parser.parse_args()

    # ── Start virtual camera thread ────────────────────────────────────────────
    if _HAS_VCAM:
        vcam_t = threading.Thread(
            target=_vcam_thread,
            args=(args.cam_width, args.cam_height, args.cam_fps),
            daemon=True,
        )
        vcam_t.start()
    else:
        print("[WARN] pyvirtualcam not installed — virtual camera disabled")
        print("       pip install pyvirtualcam")
        print("       Then install OBS Studio for the camera driver: https://obsproject.com")
        print()

    # ── Banner ─────────────────────────────────────────────────────────────────
    url = f"http://localhost:{args.port}/"
    vcam_line = (
        f"  VirtCam : {args.cam_width}x{args.cam_height} @ {args.cam_fps}fps"
        if _HAS_VCAM
        else "  VirtCam : DISABLED  (pip install pyvirtualcam)"
    )
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║          Deep Live Cam — Local Runner                 ║")
    print("  ╠═══════════════════════════════════════════════════════╣")
    print(f"  ║  UI      : {url:<46}║")
    print(f"  ║{vcam_line:<55}║")
    print("  ║  Ctrl+C  : quit                                       ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    # ── Uvicorn ────────────────────────────────────────────────────────────────
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
        ws_ping_interval=None,   # we manage our own pings upstream
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
