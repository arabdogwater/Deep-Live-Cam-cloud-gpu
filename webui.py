#!/usr/bin/env python3
"""
webui.py — Deep-Live-Cam Local UI Server

Serves the web interface on localhost so the browser can access your webcam
(getUserMedia requires a secure context — localhost counts).
No GPU, no AI libraries, no pip install needed — just Python 3.

The browser UI connects to your vast.ai GPU server for the heavy processing.

Usage:  python webui.py [--port 8080]
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import signal

PORT = 8080
DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serve files from the 'static/' directory, suppress request logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep console clean


class _Server(socketserver.ThreadingTCPServer):
    """Threaded localhost server that ignores normal browser disconnect noise."""

    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        etype, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    port = PORT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv) - 1:
            port = int(sys.argv[i + 1])
            break
        try:
            port = int(arg)
            break
        except ValueError:
            continue

    with _Server(("127.0.0.1", port), _Handler) as httpd:
        url = f"http://localhost:{port}/"
        print()
        print("  ╔═══════════════════════════════════════════════════╗")
        print("  ║         Deep Live Cam — Local UI                  ║")
        print("  ╠═══════════════════════════════════════════════════╣")
        print(f"  ║  Open: {url:<44}║")
        print("  ║  Press Ctrl+C to quit                             ║")
        print("  ╚═══════════════════════════════════════════════════╝")
        print()

        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

        def _shutdown(sig, frame):
            print("\n  Shutting down.")
            threading.Thread(target=httpd.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        httpd.serve_forever()


if __name__ == "__main__":
    main()