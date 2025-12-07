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

PORT = 8000

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"✓ Server running at http://localhost:{PORT}")
    print(f"  Open this URL in your browser to view the site")
    print(f"  Press Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
