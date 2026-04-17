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

PORT = 8080
DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serve files from the 'static/' directory, suppress request logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep console clean


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

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", port), _Handler) as httpd:
        url = f"http://localhost:{port}/"
        print()
        print("  ╔═══════════════════════════════════════════════════╗")
        print("  ║         Deep Live Cam — Local UI                  ║")
        print("  ╠═══════════════════════════════════════════════════╣")
        print(f"  ║  Open: {url:<44}║")
        print("  ║  Press Ctrl+C to quit                             ║")
        print("  ╚═══════════════════════════════════════════════════╝")
        print()

        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Shutting down.")


if __name__ == "__main__":
    main()