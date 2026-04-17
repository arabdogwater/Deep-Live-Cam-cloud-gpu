#!/bin/bash
set -e

# ─── Deep-Live-Cam – vast.ai onstart script ───────────────────────────────────
# Designed for: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# Port: uses OPEN_BUTTON_PORT if set in the vast.ai template, otherwise picks
#       a free port automatically. Never kills unrelated processes.
# ──────────────────────────────────────────────────────────────────────────────

# ── Port config (driven by vast.ai env vars) ─────────────────────────────────
# vast.ai injects these into every container:
#   OPEN_BUTTON_PORT    — internal port the "Open" button in the UI points to
#   VAST_TCP_PORT_XXXX  — external (host-side) port mapped to internal port XXXX
#   PUBLIC_IPADDR       — public IP of the instance
#
# If OPEN_BUTTON_PORT is set (vast.ai template), use it.
# Otherwise find a free port automatically — never steal an occupied one.
_find_free_port() {
    python3 -c "
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
"
}
if [ -n "${OPEN_BUTTON_PORT:-}" ]; then
    WEBUI_PORT=${OPEN_BUTTON_PORT}
else
    WEBUI_PORT=$(_find_free_port)
fi
INSTANCE_IP=${PUBLIC_IPADDR:-$(hostname -I | awk '{print $1}')}

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
info "Installing ffmpeg and build deps..."
apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    wget \
    psmisc 2>&1 | grep -E "^(Get|Unpacking|Setting up|Processing)" | sed 's/^/  | /' || true
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

# ── 6. Kill any previous instance of THIS script's server ───────────────────
step "Stopping any previous gpu_server.py instance"
pkill -9 -f "gpu_server.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
sleep 1

# Find the first free port from a candidate list.
# If OPEN_BUTTON_PORT is set but occupied (e.g. taken by vast.ai itself), skip it.
_try_bind() {
    python3 - <<EOF 2>/dev/null
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('', $1))
s.close()
EOF
}

WEBUI_PORT=""
# Prefer OPEN_BUTTON_PORT if set and actually free
if [ -n "${OPEN_BUTTON_PORT:-}" ] && _try_bind "${OPEN_BUTTON_PORT}"; then
    WEBUI_PORT=${OPEN_BUTTON_PORT}
else
    if [ -n "${OPEN_BUTTON_PORT:-}" ]; then
        info "OPEN_BUTTON_PORT=${OPEN_BUTTON_PORT} is occupied — trying fallbacks"
    fi
    # Try well-known ports first (so VAST_TCP_PORT_XXXX mapping is predictable)
    for _p in 8080 8888 7860 6006 5000; do
        if _try_bind $_p; then
            WEBUI_PORT=$_p
            break
        fi
    done
    # Absolute last resort: let OS pick any free port
    if [ -z "$WEBUI_PORT" ]; then
        WEBUI_PORT=$(_find_free_port)
    fi
fi

ok "Will bind on port $WEBUI_PORT"
elapsed

# ── 7. Launch Deep-Live-Cam GPU server ────────────────────────────────────────
step "Launching Deep-Live-Cam (GPU Server)"

# Resolve external port: vast.ai sets VAST_TCP_PORT_XXXX = external port for internal port XXXX
_webui_ext=$(eval "echo \${VAST_TCP_PORT_${WEBUI_PORT}:-${WEBUI_PORT}}")

TOTAL=$(($(date +%s) - START_TIME))
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Setup complete in ${TOTAL}s"
echo "║"
echo "║  GPU Server: http://${INSTANCE_IP}:${_webui_ext}/"
echo "║"
echo "║  On your local machine, run:  python webui.py"
echo "║  Then enter this address in the connect screen:"
echo "║    ${INSTANCE_IP}:${_webui_ext}"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

exec python gpu_server.py --port "$WEBUI_PORT" --execution-provider cuda
