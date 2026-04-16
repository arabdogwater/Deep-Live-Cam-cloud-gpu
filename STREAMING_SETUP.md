# Real-Time Webcam Streaming to vast.ai GPU

How to pipe your local webcam into Deep-Live-Cam running on a remote GPU, with the processed output visible live in your browser.

---

## How It Works

```
[Your Webcam]
     │
     │  ffmpeg (local machine)
     │  push RTSP stream
     ▼
[vast.ai GPU  :8554]  ← MediaMTX RTSP server receives stream
     │
     │  ffmpeg bridge
     ▼
[/dev/video10]  ← v4l2loopback virtual webcam device
     │
     ▼
[Deep-Live-Cam]  ← reads /dev/video10, applies face swap with CUDA
     │
     ▼
[noVNC :8080]  ← you watch the result in your browser
```

---

## Step 1 — vast.ai Instance Setup

### Docker Image
```
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

### Required Settings (in the vast.ai instance config)
| Setting | Value |
|---|---|
| **Run as Privileged** | ✔ ON — required for v4l2loopback kernel module |
| **On-Start Script** | *(see below)* |
| **Disk Space** | 30 GB minimum |

> Note: vast.ai uses port `8080` for Jupyter by default — noVNC runs on `6080` to avoid conflict.

### Open Ports
In **Edit Instance → Expose Ports**, add:
- `6080` — noVNC browser GUI
- `8554` — RTSP webcam stream input

### On-Start Script
Paste this into the vast.ai "On-Start Script" field:
```bash
git clone https://github.com/arabdogwater/Deep-Live-Cam-cloud-gpu /workspace/Deep-Live-Cam 2>/dev/null || git -C /workspace/Deep-Live-Cam pull; bash /workspace/Deep-Live-Cam/onstart.sh
```

---

## Step 2 — Install ffmpeg Locally

### Windows
Download from https://ffmpeg.org/download.html or run:
```powershell
winget install ffmpeg
```

### macOS
```bash
brew install ffmpeg
```

### Linux
```bash
sudo apt install ffmpeg
```

---

## Step 3 — Find Your Webcam Name

### Windows (PowerShell)
```powershell
ffmpeg -list_devices true -f dshow -i dummy 2>&1 | Select-String "video"
```
Look for something like `"HD Webcam"` or `"Integrated Camera"`.

### macOS
```bash
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -i video
```
Note the index number e.g. `[0]`.

### Linux
```bash
v4l2-ctl --list-devices
# Usually /dev/video0
```

---

## Step 4 — Push Your Webcam to the Server

Replace `<instance-ip>` with your vast.ai instance's public IP.

### Windows
```powershell
ffmpeg -f dshow -i video="YOUR WEBCAM NAME HERE" `
  -vcodec libx264 -preset ultrafast -tune zerolatency `
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M `
  -f rtsp rtsp://<instance-ip>:8554/webcam
```

### macOS
```bash
ffmpeg -f avfoundation -framerate 30 -i "0" \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M \
  -f rtsp rtsp://<instance-ip>:8554/webcam
```

### Linux
```bash
ffmpeg -f v4l2 -framerate 30 -i /dev/video0 \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M \
  -f rtsp rtsp://<instance-ip>:8554/webcam
```

Keep this terminal open — it must stay running while you use Deep-Live-Cam.

---

## Step 5 — Use the GUI

1. Open your browser and go to:
   ```
   http://<instance-ip>:6080/vnc.html
   ```
2. Click **Connect**
3. The Deep-Live-Cam GUI will appear
4. Click **Select Face** → pick a source face image
5. Click the **Camera dropdown** and select **`/dev/video10`** (the virtual webcam)
6. Click **Live** → face swap starts in real time

---

## Latency Expectations

| Component | Added Latency |
|---|---|
| Local ffmpeg encode | ~50ms |
| Internet upload to server | ~50–200ms (depends on your connection) |
| RTSP server + bridge | ~50ms |
| Deep-Live-Cam processing (GPU) | ~20–50ms per frame |
| noVNC display | ~50–100ms |
| **Total round-trip** | **~200–500ms** |

This is not zero-latency but is totally usable for streaming/recording. For Zoom/OBS use, the latency is fine since it's one-way.

---

## Troubleshooting

**`modprobe v4l2loopback` fails on boot**
→ Make sure "Run as Privileged" is enabled in the vast.ai instance settings. Without it, kernel module loading is blocked.

**Deep-Live-Cam doesn't show `/dev/video10` in the camera list**
→ The virtual webcam only appears once a stream is actively being pushed. Start your local ffmpeg push first, then open the camera dropdown.

**Stream lags or stutters**
→ Lower the bitrate on your local push:
```bash
-b:v 1M -maxrate 1M -bufsize 2M
```
Or add `-s 854x480` to downscale to 480p before sending.

**ffmpeg bridge log**
On the server, check `/tmp/ffmpeg-bridge.log` for errors:
```bash
tail -f /tmp/ffmpeg-bridge.log
```

**MediaMTX log**
```bash
tail -f /tmp/mediamtx.log
```
