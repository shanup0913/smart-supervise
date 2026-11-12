"""
Smart Supervise — Responsive Cross-Platform Web Entry Point (Mobile, Tablet & Desktop)
Shares core/ AI, VideoEngine, LocationTracker, and MapVisualizer directly with app.py.
Deploys seamlessly on Hugging Face Spaces (Docker port 7860), local Wi-Fi, or Cloudflare Tunnel.
"""

import os
import sys
import cv2
import socket
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware

# Prevent OpenCV internal thread thrashing alongside VideoEngine workers
cv2.setNumThreads(1)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from webapp import (  # noqa: E402
    SecurityAndAuthMiddleware,
    core_routes,
    get_state,
    lifespan,
    stream_routes,
)

routes = [*core_routes, *stream_routes]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(SecurityAndAuthMiddleware)],
)


def _detect_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    default_port = 7860 if os.environ.get("SPACE_ID") else 8000
    port = int(os.environ.get("PORT", default_port))
    lan_ip = _detect_lan_ip()
    print("=" * 64)
    print("  SMART SUPERVISE — Responsive Web Console (Desktop + Mobile)")
    print(f"  Local URL:  http://127.0.0.1:{port}")
    print(f"  Phone URL:  http://{lan_ip}:{port}  (on same Wi-Fi)")
    print("=" * 64)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
