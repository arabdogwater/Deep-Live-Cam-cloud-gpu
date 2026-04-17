# Run Instructions

## Local Machine

```bash
python webui.py
```

Opens `http://localhost:8080` in your browser automatically.

---

## Cloud GPU (vast.ai)

### Option A — Automatic (recommended)
Paste the contents of `onstart.sh` into the **On-start Script** field when creating your instance. Set **Open Port** to `8080`. The server starts automatically on boot.

### Option B — Manual (run in the GPU terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/arabdogwater/Deep-Live-Cam-cloud-gpu/main/onstart.sh | bash
```

This clones the repo, installs dependencies, downloads models, and starts the server.

---

Once running, your GPU address is:

```
http://<PUBLIC_IPADDR>:8080
```

Enter that in the browser connect screen on your local machine.
