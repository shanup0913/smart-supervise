"""
Modular SmartSupervise web server package:
- security: Session cookies, brute-force rate limit, path validation, Windows asyncio filter
- browser_stream: Zero-pre-upload browser Video->JPEG on-demand capture bridge
- engine_state: Thread-safe WebAppState coordinator & Starlette lifespan
- routes_core: Auth, state, target, playback, tracker export, map, and host folder APIs
- routes_streams: Camera slot, MJPEG stream, and browser feeder WebSocket endpoints
"""

from webapp.engine_state import WebAppState, get_state, lifespan
from webapp.routes_core import core_routes
from webapp.routes_streams import stream_routes
from webapp.security import SecurityAndAuthMiddleware

__all__ = [
    "WebAppState",
    "get_state",
    "lifespan",
    "core_routes",
    "stream_routes",
    "SecurityAndAuthMiddleware",
]
