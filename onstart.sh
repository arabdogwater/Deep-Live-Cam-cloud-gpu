#!/bin/bash
set -e

# ─── Deep-Live-Cam – vast.ai onstart script ───────────────────────────────────
# Designed for: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# GUI in browser: open the port you set as OPEN_BUTTON_PORT (default 8080)
# Ports to open in your vast.ai template: 8080 (GPU Server)
# ──────────────────────────────────────────────────────────────────────────────

# ── Port config (driven by vast.ai env vars) ─────────────────────────────────
# vast.ai injects these into every container:
#   OPEN_BUTTON_PORT    — internal port the "Open" button in the UI points to
#   VAST_TCP_PORT_XXXX  — external (host-side) port mapped to internal port XXXX
#   PUBLIC_IPADDR       — public IP of the instance
#
# In your vast.ai template, add open port: 8080 (GPU Server).
# Set OPEN_BUTTON_PORT=8080 as the primary port in the template.
WEBUI_PORT=${OPEN_BUTTON_PORT:-8080}   # internal port the GPU server binds on
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
    wget 2>&1 | grep -E "^(Get|Unpacking|Setting up|Processing)" | sed 's/^/  | /' || true
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

# ── 6. Kill anything already on the GPU server port ──────────────────────────
step "Clearing GPU server port"
fuser -k ${WEBUI_PORT}/tcp 2>/dev/null || true
ok "Port $WEBUI_PORT is free"
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

exec python gpu_server.py --execution-provider cuda
