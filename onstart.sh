#!/bin/bash
set -e

# ─── Deep-Live-Cam – vast.ai onstart script ───────────────────────────────────
# Designed for: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# GUI in browser: open the port you set as OPEN_BUTTON_PORT (default 8080)
# Ports to open in your vast.ai template: 8080 (Web UI), 8554 (RTSP)
# ──────────────────────────────────────────────────────────────────────────────

# ── Port config (driven by vast.ai env vars) ─────────────────────────────────
# vast.ai injects these into every container:
#   OPEN_BUTTON_PORT    — internal port the "Open" button in the UI points to
#   VAST_TCP_PORT_XXXX  — external (host-side) port mapped to internal port XXXX
#   PUBLIC_IPADDR       — public IP of the instance
#
# In your vast.ai template, add open ports: 8080 (Web UI) and 8554 (RTSP).
# Set OPEN_BUTTON_PORT=8080 as the primary port in the template.
# Override RTSP_PORT here only if you open a different port in the template.
WEBUI_PORT=${OPEN_BUTTON_PORT:-8080}   # internal port the FastAPI web UI binds on
RTSP_PORT=${RTSP_PORT:-8554}           # internal port MediaMTX RTSP listens on
INSTANCE_IP=${PUBLIC_IPADDR:-$(hostname -I | awk '{print $1}')}
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
info "Installing ffmpeg, v4l2loopback, and build deps..."
apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    wget \
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
    # --system-site-packages lets the venv see Docker's pre-installed torch/CUDA
    python3 -m venv --system-site-packages "$VENV_DIR"
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

# PyTorch is pre-installed in pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# Only install if missing (e.g. someone used a different base image)
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "PyTorch $(python -c 'import torch; print(torch.__version__)') with CUDA already present — skipping"
else
    info "PyTorch not found — installing for CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 \
        | grep -E "^(Collecting|Downloading|Installing|Successfully)" | sed 's/^/  | /' || true
    ok "PyTorch installed"
fi

info "Installing project requirements..."
pip install -r requirements.txt 2>&1 \
    | grep -E "^(Collecting|Downloading|Installing|Successfully)" | sed 's/^/  | /' || true

info "Installing web UI dependencies (FastAPI, Uvicorn)..."
pip install fastapi "uvicorn[standard]" 2>&1 \
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

# Build v4l2loopback via DKMS if module isn't loadable yet
if ! modinfo v4l2loopback &>/dev/null; then
    info "Building v4l2loopback kernel module via DKMS..."
    dkms autoinstall 2>&1 | tail -5 || true
fi

# Load the module
if ! lsmod | grep -q v4l2loopback; then
    info "Loading v4l2loopback kernel module..."
    if ! modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1 2>/dev/null; then
        echo "  ✘  v4l2loopback could not load — webcam streaming will not work"
        echo "     Fix: enable 'Privileged' on the vast.ai instance and restart"
    else
        ok "Virtual webcam at $VIRTUAL_CAM"
    fi
else
    ok "Virtual webcam at $VIRTUAL_CAM (already loaded)"
fi

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
# MTX_RTSPADDRESS tells MediaMTX which port to bind without needing a config file
export MTX_RTSPADDRESS=":${RTSP_PORT}"
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

# ── 7. Kill anything already on the web UI port ───────────────────────────────
step "Clearing web UI port"
fuser -k ${WEBUI_PORT}/tcp 2>/dev/null || true
ok "Port $WEBUI_PORT is free"
elapsed

# ── 8. Launch Deep-Live-Cam web UI ────────────────────────────────────────────
step "Launching Deep-Live-Cam (FastAPI Web UI)"

# Resolve external ports: vast.ai sets VAST_TCP_PORT_XXXX = external port for internal port XXXX
_webui_ext=$(eval "echo \${VAST_TCP_PORT_${WEBUI_PORT}:-${WEBUI_PORT}}")
_rtsp_ext=$(eval "echo \${VAST_TCP_PORT_${RTSP_PORT}:-${RTSP_PORT}}")

TOTAL=$(($(date +%s) - START_TIME))
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Setup complete in ${TOTAL}s"
echo "║"
echo "║  Web UI:         http://${INSTANCE_IP}:${_webui_ext}/"
echo "║  Push webcam to: rtsp://${INSTANCE_IP}:${_rtsp_ext}/webcam"
echo "║"
echo "║  WebUI internal:${WEBUI_PORT}  external:${_webui_ext}"
echo "║  RTSP  internal:${RTSP_PORT}  external:${_rtsp_ext}"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

exec python webui.py --execution-provider cuda
