#!/usr/bin/env python3
"""
Simple HTTP server for testing the static site locally.

Usage:
    python serve.py
    
Then open: http://localhost:8000
"""

import http.server
import socketserver
import os
import sys

# Port selection: env var PORT or first CLI arg, default 8000
PORT = int(os.environ.get("PORT", 8000))
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except Exception:
        pass

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler


class ReuseAddrTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReuseAddrTCPServer(("", PORT), Handler) as httpd:
    print(f"✓ Server running at http://localhost:{PORT}")
    print(f"  Open this URL in your browser to view the site")
    print(f"  Press Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
