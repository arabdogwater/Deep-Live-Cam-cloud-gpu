# Local Machine Setup — Real-Time Webcam to Deep-Live-Cam

Everything the GPU does is handled automatically by `onstart.sh`. This guide is **only what you run on your local machine**.

---

## Step 1 — Install ffmpeg

### Windows (PowerShell — run once)
```powershell
winget install ffmpeg
```
Then **close and reopen PowerShell** so `ffmpeg` is on your PATH.

### macOS
```bash
brew install ffmpeg
```

### Linux
```bash
sudo apt install ffmpeg
```

---

## Step 2 — Find Your Webcam Name

### Windows
```powershell
ffmpeg -list_devices true -f dshow -i dummy 2>&1 | Select-String "video"
```
You'll see something like `"HD Webcam"` or `"Integrated Camera"`. Copy the exact name inside the quotes.

### macOS
```bash
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -i video
```
Note the number in brackets, e.g. `[0]`.

### Linux
```bash
v4l2-ctl --list-devices
# Usually /dev/video0
```

---

## Step 3 — Push Your Webcam to the GPU

Open a terminal and run the command for your OS. **Keep it running the whole time you use the app.**

### Windows
```powershell
ffmpeg -f dshow -i video="YOUR WEBCAM NAME HERE" `
  -vcodec libx264 -preset ultrafast -tune zerolatency `
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M `
  -f rtsp rtsp://77.48.24.250:48207/webcam
```
Replace `YOUR WEBCAM NAME HERE` with the exact name from Step 2.

### macOS
```bash
ffmpeg -f avfoundation -framerate 30 -i "0" \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M \
  -f rtsp rtsp://77.48.24.250:48207/webcam
```
Replace `0` with your device index from Step 2.

### Linux
```bash
ffmpeg -f v4l2 -framerate 30 -i /dev/video0 \
  -vcodec libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -b:v 2M -maxrate 2M -bufsize 4M \
  -f rtsp rtsp://77.48.24.250:48207/webcam
```

When it's working you'll see output like:
```
frame=  42 fps= 30 q=28.0 size=    512kB time=00:00:01.40 bitrate=2994.3kbits/s
```

---

## Step 4 — Open the GUI

Once the ffmpeg stream is running, open your browser and go to:

```
http://77.48.24.250:48253/vnc.html
```

Click **Connect** (no password). The Deep-Live-Cam window will appear.

1. Click **Select Face** → choose a face image from your machine (upload via the noVNC clipboard or use a pre-uploaded file)
2. Click the **Camera dropdown** → select **`/dev/video10`**
3. Click **Live** → real-time face swap starts

---

## Troubleshooting

**ffmpeg exits immediately with `dshow` error (Windows)**
→ The webcam name doesn't match exactly. Re-run Step 2 and copy it character-for-character.

**`/dev/video10` not in the camera list**
→ The stream isn't reaching the server yet. Check your ffmpeg terminal — it should show frame output. Try restarting it.

**Stream lags or stutters**
→ Lower the bitrate:
```powershell
# replace -b:v 2M -maxrate 2M -bufsize 4M with:
-b:v 1M -maxrate 1M -bufsize 2M
```
Or add `-s 854x480` to send 480p instead of full res.

**GUI won't load**
→ The server may still be setting up (first boot takes ~5–10 min). Wait, then refresh.
