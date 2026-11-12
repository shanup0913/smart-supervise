"""
Security, authentication, safe filesystem validation, streaming upload helpers,
and Windows asyncio disconnect noise suppression for the SmartSupervise web server.
"""

import os
import re
import hmac
import time
import uuid
import hashlib
import secrets
import threading
import urllib.parse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".ts", ".flv", ".wmv"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024

_COOKIE_NAME = "ss_session"
_SESSION_TTL_SEC = 12 * 3600
_SESSION_SECRET = os.environ.get("SMARTSUPERVISE_SESSION_SECRET", "").encode("utf-8") or secrets.token_bytes(32)
_AUTH_FAIL_LOCK = threading.Lock()
_AUTH_FAILS = {}  # ip -> [fail_count, window_expires_at]

_BLOCKED_METADATA_HOSTS = {"169.254.169.254", "169.254.170.2", "metadata.google.internal"}


def get_access_code() -> str:
    """Return the active passcode if public/session protection is enabled."""
    return os.environ.get("SMARTSUPERVISE_ACCESS_CODE", "").strip()


def auth_enabled() -> bool:
    """True when an access passcode protects the web console."""
    return bool(get_access_code())


def client_ip(conn) -> str:
    """Extract originating client IP (honoring X-Forwarded-For behind reverse proxies)."""
    fwd = conn.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if fwd:
        return fwd
    return conn.client.host if getattr(conn, "client", None) else "unknown"


def check_login_rate_limit(ip: str) -> bool:
    """Return True if the client IP is temporarily rate-limited due to failed logins."""
    now = time.time()
    with _AUTH_FAIL_LOCK:
        entry = _AUTH_FAILS.get(ip)
        return bool(entry and entry[1] > now and entry[0] >= 8)


def record_login_failure(ip: str):
    """Increment failed login attempts within the 60-second sliding window."""
    now = time.time()
    with _AUTH_FAIL_LOCK:
        entry = _AUTH_FAILS.get(ip)
        if not entry or entry[1] <= now:
            _AUTH_FAILS[ip] = [1, now + 60.0]
        else:
            entry[0] += 1


def clear_login_failures(ip: str):
    """Clear login failure tracking after a successful authentication."""
    with _AUTH_FAIL_LOCK:
        _AUTH_FAILS.pop(ip, None)


def sign_session_token() -> str:
    """Create an expiring HMAC-SHA256 signed session cookie value."""
    exp = int(time.time()) + _SESSION_TTL_SEC
    nonce = secrets.token_hex(16)
    msg = f"{exp}.{nonce}".encode("ascii")
    sig = hmac.new(_SESSION_SECRET, msg, hashlib.sha256).hexdigest()
    return f"{exp}.{nonce}.{sig}"


def verify_session_token(token: str) -> bool:
    """Validate expiration and HMAC signature of a session cookie value."""
    if not token or not isinstance(token, str):
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    exp_s, nonce, sig = parts
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < int(time.time()):
        return False
    expected = hmac.new(_SESSION_SECRET, f"{exp_s}.{nonce}".encode("ascii"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def is_authenticated(conn) -> bool:
    """Return True if auth is disabled or the request carries a valid session cookie."""
    if not auth_enabled():
        return True
    tok = conn.cookies.get(_COOKIE_NAME, "")
    return verify_session_token(tok)


def is_under_root(path_real: str, root_real: str) -> bool:
    """Verify containment inside root_real without prefix-sibling bypass."""
    try:
        return os.path.commonpath([root_real, path_real]) == root_real
    except ValueError:
        return False


def allowed_media_roots():
    """Return normalized safe directories permitted for host disk video selection."""
    roots = [UPLOADS_DIR, BASE_DIR]
    home = os.path.expanduser("~")
    for sub in ("Videos", "Desktop", "Downloads", "Movies"):
        roots.append(os.path.join(home, sub))
    extra = os.environ.get("SMARTSUPERVISE_MEDIA_DIRS", "")
    for raw in extra.split(os.pathsep):
        raw = raw.strip()
        if raw:
            roots.append(raw)
    out = []
    seen = set()
    for p in roots:
        try:
            rp = os.path.realpath(p)
            if os.path.isdir(rp) and rp not in seen:
                seen.add(rp)
                out.append(rp)
        except OSError:
            continue
    return out


def is_safe_media_path(raw_path: str, allowed_exts=None):
    """Resolve and validate that raw_path points to an allowed media file under safe roots."""
    if not raw_path or not isinstance(raw_path, str):
        return None
    try:
        cand = os.path.realpath(raw_path.strip())
    except OSError:
        return None
    if not os.path.isfile(cand):
        return None
    if allowed_exts is not None:
        ext = os.path.splitext(cand)[1].lower()
        if ext not in allowed_exts:
            return None
    for root in allowed_media_roots():
        if is_under_root(cand, root):
            return cand
    return None


def is_cloud_container() -> bool:
    """True when deployed inside a cloud container runtime (Hugging Face Spaces or Railway)."""
    return bool(
        os.environ.get("SPACE_ID")
        or os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("RAILWAY_PROJECT_ID")
        or os.environ.get("RAILWAY_SERVICE_ID")
    )


def host_fs_allowed() -> bool:
    """True unless running in container mode or explicitly disabled by env."""
    if is_cloud_container():
        return False
    return os.environ.get("SMARTSUPERVISE_ALLOW_HOST_FS", "1").strip() != "0"


def can_browse_host_fs(conn) -> bool:
    """Authorize host media folder browsing for authenticated sessions or local loopback."""
    if not host_fs_allowed():
        return False
    if auth_enabled():
        return is_authenticated(conn)
    if conn.headers.get("x-forwarded-for"):
        return False
    host = conn.client.host if getattr(conn, "client", None) else ""
    return host in ("127.0.0.1", "::1", "localhost")


def validate_slot_source(raw_source: str, conn=None) -> tuple[bool, str]:
    """Validate camera slot source against webcam index, stream URL, or allowed local media."""
    src = str(raw_source or "").strip()
    if not src:
        return True, ""
    if src.isdigit() and len(src) <= 2:
        return True, src
    if src.startswith("browser://"):
        tail = src[len("browser://"):]
        if tail.isdigit() and 0 <= int(tail) < 4:
            return True, src
        return False, "Invalid browser stream reference."
    lower = src.lower()
    if lower.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        parsed = urllib.parse.urlparse(src)
        host = (parsed.hostname or "").lower()
        if host in _BLOCKED_METADATA_HOSTS:
            return False, "Stream host is blocked for security."
        return True, src
    if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?(?:/.*)?$", src):
        host_ip = src.split("/", 1)[0].split(":", 1)[0]
        if host_ip in _BLOCKED_METADATA_HOSTS:
            return False, "Stream host is blocked for security."
        return True, src
    if conn is not None and not can_browse_host_fs(conn):
        return False, "Host file paths require authenticated local mode."
    safe = is_safe_media_path(src, allowed_exts=_VIDEO_EXTS)
    if not safe:
        return False, "Video file not found in allowed media folders or unsupported video format."
    return True, safe


def sanitize_filename(filename, default_ext=".bin"):
    """Generate a collision-free filesystem-safe filename."""
    base = os.path.basename(str(filename or "upload")).strip()
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    if not safe:
        safe = f"file{default_ext}"
    return f"{uuid.uuid4().hex[:8]}_{safe}"


async def save_upload_file(upload_file, subdir, allowed_exts=None, max_bytes=MAX_VIDEO_BYTES):
    """Stream uploaded file in 1 MB chunks while enforcing extension allowlist and byte cap."""
    orig_name = getattr(upload_file, "filename", "upload")
    ext = os.path.splitext(str(orig_name or ""))[1].lower()
    if allowed_exts is not None and ext not in allowed_exts:
        raise ValueError(f"Unsupported file format '{ext or 'unknown'}'.")
    dest_dir = os.path.join(UPLOADS_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = sanitize_filename(orig_name)
    dest_path = os.path.join(dest_dir, safe_name)
    total = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Upload exceeds size limit ({max_bytes // (1024 * 1024)} MB).")
                f.write(chunk)
    except Exception:
        try:
            if os.path.isfile(dest_path):
                os.remove(dest_path)
        except OSError:
            pass
        raise
    return dest_path


def install_windows_proactor_filter(loop):
    """
    Suppress harmless Windows ProactorEventLoop ConnectionResetError tracebacks
    (e.g. WinError 10054 when a browser tab closes or switches an MJPEG feed).
    """
    if os.name != "nt" or loop is None:
        return
    if getattr(loop, "_ss_proactor_filtered", False):
        return
    prev_handler = loop.get_exception_handler()

    def _handler(target_loop, context):
        exc = context.get("exception")
        msg = str(context.get("message") or "")
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (10054, 10053, 121):
            return
        if "_ProactorBasePipeTransport._call_connection_lost" in msg:
            return
        if prev_handler is not None:
            prev_handler(target_loop, context)
        else:
            target_loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    loop._ss_proactor_filtered = True


class SecurityAndAuthMiddleware(BaseHTTPMiddleware):
    """Protect all private endpoints behind session auth and attach standard security headers."""

    _PUBLIC_HTTP_PATHS = {"/", "/api/health", "/api/auth/status", "/api/auth/login", "/api/auth/logout"}

    async def dispatch(self, request: Request, call_next):
        if auth_enabled() and request.url.path not in self._PUBLIC_HTTP_PATHS:
            if not is_authenticated(request):
                return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if os.environ.get("SPACE_ID"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "frame-ancestors 'self' https://huggingface.co https://*.hf.space",
            )
        else:
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
