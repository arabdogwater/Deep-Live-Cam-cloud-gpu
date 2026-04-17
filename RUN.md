# Run Instructions

## Local Machine

```bash
python webui.py
```

Opens `http://localhost:8080` in your browser automatically.

---

## Cloud GPU (vast.ai)

Paste `onstart.sh` contents into the **On-start Script** field when creating your instance.

Set **Open Port** to `8080`.

Once the instance is running, click **Open** in the vast.ai UI — or connect manually:

```
http://<PUBLIC_IPADDR>:8080
```

The server starts automatically. Enter that address in the browser connect screen on your local machine.
