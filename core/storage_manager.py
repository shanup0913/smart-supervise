"""
Bounded edge storage retention and cleanup helpers for alerts, uploads, and reports.
"""

import os
import time
import threading

_MAINT_LOCK = threading.Lock()
_DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def enforce_retention(
    directory: str,
    max_age_days: float = 7.0,
    max_size_mb: float = 500.0,
    extensions=None,
    skip_paths=None,
) -> dict:
    """
    Prunes old or oversized files under `directory` cleanly without touching `skip_paths`.
    Phase 1 deletes files older than `max_age_days`; Phase 2 deletes oldest files until
    remaining disk usage is within `max_size_mb`.
    """
    stats = {"deleted_count": 0, "freed_bytes": 0, "remaining_bytes": 0}
    if not directory or not os.path.isdir(directory):
        return stats

    ext_set = {e.lower() for e in extensions} if extensions else None
    protected = {os.path.realpath(p) for p in (skip_paths or ()) if p}
    now = time.time()
    cutoff = now - (max_age_days * 86400.0) if max_age_days > 0 else None
    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb > 0 else 0

    kept = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.islink(fpath):
                continue
            real_path = os.path.realpath(fpath)
            if real_path in protected:
                continue
            if ext_set is not None and os.path.splitext(fname)[1].lower() not in ext_set:
                continue
            try:
                st = os.stat(fpath)
            except OSError:
                continue

            if cutoff is not None and st.st_mtime < cutoff:
                try:
                    os.remove(fpath)
                    stats["deleted_count"] += 1
                    stats["freed_bytes"] += st.st_size
                except OSError:
                    kept.append((st.st_mtime, st.st_size, fpath))
            else:
                kept.append((st.st_mtime, st.st_size, fpath))

    total_bytes = sum(sz for _, sz, _ in kept)
    if max_bytes > 0 and total_bytes > max_bytes:
        kept.sort(key=lambda item: item[0])  # Oldest first
        remaining = []
        for mtime, sz, fpath in kept:
            if total_bytes > max_bytes:
                try:
                    os.remove(fpath)
                    total_bytes -= sz
                    stats["deleted_count"] += 1
                    stats["freed_bytes"] += sz
                    continue
                except OSError:
                    pass
            remaining.append((mtime, sz, fpath))
        total_bytes = sum(sz for _, sz, _ in remaining)

    stats["remaining_bytes"] = total_bytes
    return stats


def run_storage_maintenance(base_dir: str = None, active_paths=None) -> dict:
    """Enforces bounded retention across alerts/, uploads/, and reports/."""
    root = os.path.abspath(base_dir or _DEFAULT_BASE_DIR)
    skip = [p for p in (active_paths or ()) if p]
    with _MAINT_LOCK:
        return {
            "alerts": enforce_retention(
                os.path.join(root, "alerts"),
                max_age_days=_env_float("SMARTSUPERVISE_ALERT_MAX_DAYS", 7.0),
                max_size_mb=_env_float("SMARTSUPERVISE_ALERT_MAX_MB", 250.0),
                extensions={".jpg", ".jpeg", ".png"},
                skip_paths=skip,
            ),
            "uploads": enforce_retention(
                os.path.join(root, "uploads"),
                max_age_days=_env_float("SMARTSUPERVISE_UPLOAD_MAX_DAYS", 3.0),
                max_size_mb=_env_float("SMARTSUPERVISE_UPLOAD_MAX_MB", 1024.0),
                skip_paths=skip,
            ),
            "reports": enforce_retention(
                os.path.join(root, "reports"),
                max_age_days=_env_float("SMARTSUPERVISE_REPORT_MAX_DAYS", 14.0),
                max_size_mb=_env_float("SMARTSUPERVISE_REPORT_MAX_MB", 100.0),
                extensions={".csv", ".json"},
                skip_paths=skip,
            ),
        }
