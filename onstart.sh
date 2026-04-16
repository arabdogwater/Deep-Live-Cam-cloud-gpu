#!/bin/bash
set -e

# ─── Deep-Live-Cam – vast.ai onstart script ───────────────────────────────────
# Designed for: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# GUI in browser: http://<instance-ip>:8080/vnc.html
# ──────────────────────────────────────────────────────────────────────────────

NOVNC_PORT=1111
RTSP_PORT=48207       # port your local machine pushes webcam stream to
VIRTUAL_CAM=/dev/video10  # virtual webcam device Deep-Live-Cam will open

REPO_URL="https://github.com/arabdogwater/Deep-Live-Cam-cloud-gpu"
APP_DIR="/workspace/Deep-Live-Cam"
VENV_DIR="$APP_DIR/venv"
MODELS_DIR="$APP_DIR/models"

# ── Helpers ────────────────────────────────────────────────────────────────────
STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP $STEP — $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
ok()   { echo "  ✔  $1"; }
info() { echo "  →  $1"; }

START_TIME=$(date +%s)
elapsed() {
    echo "  ⏱  $(($(date +%s) - START_TIME))s elapsed"
}

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║        Deep-Live-Cam  —  vast.ai startup          ║"
echo "╚═══════════════════════════════════════════════════╝"
echo "  Started at $(date)"

# ── 1. System packages ─────────────────────────────────────────────────────────
step "Installing system packages"
info "Running apt-get update..."
apt-get update -qq
info "Installing ffmpeg, tkinter, xvfb, x11vnc, noVNC, v4l2loopback..."
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
    websockify \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    linux-headers-$(uname -r) 2>&1 | grep -E "^(Get|Unpacking|Setting up|Processing)" | sed 's/^/  | /' || true
ok "System packages ready"
elapsed

# ── 2. Clone / update repo ────────────────────────────────────────────────────
step "Fetching repository"
if [ -d "$APP_DIR/.git" ]; then
    info "Repo exists — pulling latest changes..."
    git -C "$APP_DIR" pull
    ok "Repository up to date"
else
    info "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$APP_DIR"
    ok "Repository cloned"
fi
elapsed

cd "$APP_DIR"

# ── 3. Python virtual environment ─────────────────────────────────────────────
step "Setting up Python venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    ok "venv created"
else
    ok "venv already exists — skipping"
fi
source "$VENV_DIR/bin/activate"
info "Python: $(python --version)"
elapsed

# ── 4. Python dependencies ────────────────────────────────────────────────────
step "Installing Python dependencies"

info "Upgrading pip..."
pip install --upgrade pip -q

info "Installing PyTorch (CUDA 12.4) — this may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 \
    | grep -E "^(Collecting|Downloading|Installing|Successfully)" | sed 's/^/  | /' || true
ok "PyTorch installed"

info "Installing project requirements..."
pip install -r requirements.txt 2>&1 \
    | grep -E "^(Collecting|Downloading|Installing|Successfully)" | sed 's/^/  | /' || true
ok "All Python dependencies installed"
elapsed

# ── 5. Download models ────────────────────────────────────────────────────────
step "Downloading models"
mkdir -p "$MODELS_DIR"

INSWAPPER="$MODELS_DIR/inswapper_128_fp16.onnx"
if [ -f "$INSWAPPER" ]; then
    ok "inswapper_128_fp16.onnx already present — skipping"
else
    info "Downloading inswapper_128_fp16.onnx (~170 MB)..."
    wget --progress=bar:force:noscroll \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx" \
        -O "$INSWAPPER" 2>&1 | tail -3
    ok "inswapper_128_fp16.onnx downloaded"
fi

GFPGAN="$MODELS_DIR/GFPGANv1.4.onnx"
if [ -f "$GFPGAN" ]; then
    ok "GFPGANv1.4.onnx already present — skipping"
else
    info "Downloading GFPGANv1.4.onnx (~350 MB)..."
    wget --progress=bar:force:noscroll \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx" \
        -O "$GFPGAN" 2>&1 | tail -3
    ok "GFPGANv1.4.onnx downloaded"
fi
elapsed

# ── 6. Virtual webcam + RTSP server (for live streaming from local machine) ────
step "Setting up virtual webcam (v4l2loopback + MediaMTX)"

# Load v4l2loopback kernel module to create a virtual webcam device
if ! lsmod | grep -q v4l2loopback; then
    info "Loading v4l2loopback kernel module..."
    modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1 || \
        { echo "  ✘  v4l2loopback failed — ensure instance is set to 'Privileged' in vast.ai"; }
fi
ok "Virtual webcam at $VIRTUAL_CAM"

# Install MediaMTX (lightweight RTSP server) if not already present
MEDIAMTX_BIN="/usr/local/bin/mediamtx"
if [ ! -f "$MEDIAMTX_BIN" ]; then
    info "Downloading MediaMTX RTSP server..."
    MEDIAMTX_VER="v1.9.3"
    wget -q "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VER}/mediamtx_${MEDIAMTX_VER}_linux_amd64.tar.gz" \
        -O /tmp/mediamtx.tar.gz
    tar -xzf /tmp/mediamtx.tar.gz -C /usr/local/bin mediamtx
    chmod +x "$MEDIAMTX_BIN"
    rm /tmp/mediamtx.tar.gz
fi
ok "MediaMTX ready"

# Start MediaMTX (accepts RTSP pushes on port $RTSP_PORT)
info "Starting MediaMTX RTSP server on port $RTSP_PORT ..."
mediamtx &>/tmp/mediamtx.log &
sleep 1
ok "RTSP server listening on port $RTSP_PORT"

# Bridge: once a stream arrives at rtsp://localhost:$RTSP_PORT/webcam, feed it into v4l2loopback
info "Starting ffmpeg bridge (RTSP → $VIRTUAL_CAM)..."
(
    while true; do
        ffmpeg -rtsp_transport tcp \
               -i "rtsp://localhost:$RTSP_PORT/webcam" \
               -vf "scale=1280:720" \
               -f v4l2 "$VIRTUAL_CAM" \
               -loglevel error 2>/tmp/ffmpeg-bridge.log || true
        sleep 2  # retry if stream drops
    done
) &
ok "Webcam bridge ready — push your stream to rtsp://<instance-ip>:$RTSP_PORT/webcam"
elapsed

# ── 7. Launch virtual display + noVNC ─────────────────────────────────────────
step "Starting virtual display and noVNC"

export DISPLAY=:99
info "Starting Xvfb on $DISPLAY ..."
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1
ok "Xvfb running"

info "Starting x11vnc (VNC server)..."
x11vnc -display :99 -nopw -listen localhost -forever -quiet &
sleep 1
ok "x11vnc running"

info "Starting noVNC on port $NOVNC_PORT ..."
NOVNC_PATH=$(find /usr/share -name "vnc.html" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "/usr/share/novnc")
websockify --web="$NOVNC_PATH" $NOVNC_PORT localhost:5900 &
sleep 1
ok "noVNC running"
elapsed

# ── 8. Launch Deep-Live-Cam ───────────────────────────────────────────────────
step "Launching Deep-Live-Cam"

TOTAL=$(($(date +%s) - START_TIME))
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Setup complete in ${TOTAL}s"
echo "║"
echo "║  GUI (browser):  http://77.48.24.250:48253/vnc.html"
echo "║  Push webcam to: rtsp://77.48.24.250:48207/webcam"
echo "║"
echo "║  Ports already open on this instance."
echo "╚═══════════════════════════════════════════════════╝"
echo ""

exec python run.py --execution-provider cuda
