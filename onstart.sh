#!/bin/bash
set -e

# ─── Deep-Live-Cam – vast.ai onstart script ───────────────────────────────────
# Clones the repo, installs deps, downloads models, then launches the app.
# Designed for: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#
# Access the GUI in your browser at:  http://<instance-ip>:8080/vnc.html
# Make sure port 8080 is open in your vast.ai instance port settings.
# ──────────────────────────────────────────────────────────────────────────────

NOVNC_PORT=8080

REPO_URL="https://github.com/arabdogwater/Deep-Live-Cam-cloud-gpu"
APP_DIR="/workspace/Deep-Live-Cam"
VENV_DIR="$APP_DIR/venv"
MODELS_DIR="$APP_DIR/models"

# ── 1. System packages ─────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    python3-tk \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    wget \
    xvfb \
    x11vnc \
    novnc \
    websockify 2>&1 | tail -5

# ── 2. Clone repo ──────────────────────────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
    echo "[onstart] Repo already cloned – pulling latest..."
    git -C "$APP_DIR" pull
else
    echo "[onstart] Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

# ── 3. Python venv ─────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 4. Install Python dependencies ────────────────────────────────────────────
pip install --upgrade pip -q

# Install PyTorch for CUDA 12.4 first (ensures correct CUDA-linked torch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Install the rest of the project requirements
pip install -r requirements.txt -q

# ── 5. Download models (skips if already present) ─────────────────────────────
mkdir -p "$MODELS_DIR"

INSWAPPER="$MODELS_DIR/inswapper_128_fp16.onnx"
if [ ! -f "$INSWAPPER" ]; then
    echo "[onstart] Downloading inswapper_128_fp16.onnx ..."
    wget -q --show-progress \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx" \
        -O "$INSWAPPER"
fi

GFPGAN="$MODELS_DIR/GFPGANv1.4.onnx"
if [ ! -f "$GFPGAN" ]; then
    echo "[onstart] Downloading GFPGANv1.4.onnx ..."
    wget -q --show-progress \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx" \
        -O "$GFPGAN"
fi

echo "[onstart] Models ready."

# ── 6. Launch ──────────────────────────────────────────────────────────────────
# Xvfb: virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1

# x11vnc: exposes the virtual display as a VNC server (localhost only)
x11vnc -display :99 -nopw -listen localhost -forever -quiet &
sleep 1

# noVNC + websockify: serves the browser-based GUI on port $NOVNC_PORT
# Access at: http://<instance-ip>:8080/vnc.html
NOVNC_PATH=$(find /usr/share -name "vnc.html" 2>/dev/null | head -1 | xargs dirname || echo "/usr/share/novnc")
websockify --web="$NOVNC_PATH" $NOVNC_PORT localhost:5900 &
echo "[onstart] noVNC running on port $NOVNC_PORT  →  http://<instance-ip>:$NOVNC_PORT/vnc.html"

echo "[onstart] Starting Deep-Live-Cam with CUDA..."
python run.py --execution-provider cuda
