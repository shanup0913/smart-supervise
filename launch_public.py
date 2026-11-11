"""
SmartSupervise Public Launcher
Starts web_app.py locally + Cloudflare Quick Tunnel (free $0 HTTPS public link, zero account needed).
"""

import os
import re
import sys
import time
import socket
import signal
import threading
import subprocess
import urllib.request

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CLOUDFLARED_PATH = os.path.join(TOOLS_DIR, "cloudflared.exe" if os.name == "nt" else "cloudflared")
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    if os.name == "nt"
    else "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
PORT = int(os.environ.get("PORT", "8000"))


def ensure_cloudflared():
    if os.path.isfile(CLOUDFLARED_PATH) and os.path.getsize(CLOUDFLARED_PATH) > 5_000_000:
        return CLOUDFLARED_PATH
    os.makedirs(TOOLS_DIR, exist_ok=True)
    print(f"[Launcher] Downloading official Cloudflare Quick Tunnel binary...", flush=True)
    tmp_path = CLOUDFLARED_PATH + ".part"
    urllib.request.urlretrieve(CLOUDFLARED_URL, tmp_path)
    if os.path.getsize(tmp_path) < 5_000_000:
        raise RuntimeError("Downloaded cloudflared binary looks incomplete.")
    os.replace(tmp_path, CLOUDFLARED_PATH)
    print(f"[Launcher] Saved to {CLOUDFLARED_PATH}", flush=True)
    return CLOUDFLARED_PATH


def detect_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            srv.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def choose_listen_port(preferred_port, explicit=False):
    if _is_port_free(preferred_port):
        return preferred_port
    if explicit:
        raise RuntimeError(f"Requested PORT={preferred_port} is already in use.")
    for candidate in range(preferred_port + 1, preferred_port + 25):
        if _is_port_free(candidate):
            print(f"[Launcher] Port {preferred_port} is busy; using free port {candidate}.", flush=True)
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("0.0.0.0", 0))
        fallback = srv.getsockname()[1]
    print(f"[Launcher] Port {preferred_port} is busy; using free port {fallback}.", flush=True)
    return fallback


def wait_local_ready(port, proc=None, timeout_sec=60):
    deadline = time.time() + timeout_sec
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"web_app.py exited early with exit code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"web_app.py failed to start on port {port} within {timeout_sec}s")


def main():
    cf_bin = ensure_cloudflared()
    port = choose_listen_port(PORT, explicit=("PORT" in os.environ))
    env = os.environ.copy()
    env["PORT"] = str(port)

    print("[Launcher] Starting SmartSupervise server (web_app.py)...", flush=True)
    web_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", str(port), "--proxy-headers", "--log-level", "warning"],
        cwd=BASE_DIR,
        env=env,
    )

    cf_proc = None
    stop_watch = threading.Event()
    try:
        wait_local_ready(port, proc=web_proc, timeout_sec=90)
        lan_ip = detect_lan_ip()
        print(f"[Launcher] Local server ready at http://127.0.0.1:{port} (LAN: http://{lan_ip}:{port})", flush=True)

        cf_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _watch_web_proc():
            while not stop_watch.wait(2.0):
                if web_proc.poll() is not None:
                    print("[Launcher] web_app.py exited unexpectedly.", flush=True)
                    if cf_proc.poll() is None:
                        cf_proc.terminate()
                    break

        threading.Thread(target=_watch_web_proc, daemon=True).start()

        url_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        announced = False
        for line in cf_proc.stdout:
            if not announced:
                m = url_re.search(line)
                if m:
                    public_url = m.group(0)
                    announced = True
                    print("\n" + "=" * 70, flush=True)
                    print("  SMART SUPERVISE IS LIVE!", flush=True)
                    print(f"  Public URL (Phone / Any PC): {public_url}", flush=True)
                    print(f"  Local URL  (This PC):        http://127.0.0.1:{port}", flush=True)
                    print(f"  Wi-Fi URL  (Same Network):   http://{lan_ip}:{port}", flush=True)
                    print("  Keep this window open to stay live. Press Ctrl+C to stop.", flush=True)
                    print("=" * 70 + "\n", flush=True)
            if web_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down cleanly...", flush=True)
    finally:
        stop_watch.set()
        for p in (cf_proc, web_proc):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(0.5)
        for p in (cf_proc, web_proc):
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
