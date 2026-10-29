import os
import csv
import time
import json
import threading
from datetime import datetime


class LocationTracker:
    """
    LocationTracker: Intelligent multi-camera journey path and station engine.
    
    Features:
    - Smart Station Dwell: While a target remains in the same camera, updates dwell
      duration without creating duplicate hops or spamming the timeline.
    - Clean Transition Paths: Draws movement path only when the target actually
      changes cameras (e.g. Cam 1 -> Cam 2).
    - Return Path Support: When target returns to a previous camera (Cam 2 -> Cam 1),
      records the return transit cleanly.
    - Ping-Pong Protection: Prevents single-frame false alarms from causing fake hops.
    - Thread-Safe: Protected with threading.Lock().
    """
    def __init__(self, lost_timeout_seconds=30, session_name=None):
        self._lock = threading.Lock()
        self.lost_timeout_seconds = lost_timeout_seconds
        self.session_name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self.timeline = []       # Detailed transition & dwell log
        self.journey_route = []  # Clean sequence of unique station stops [Cam 1, Cam 2, Cam 1]
        
        # State tracking
        self.current_camera_id = None
        self.current_camera_name = None
        self.last_transition_epoch = 0.0
        self.candidate_switch_count = 0
        self.candidate_switch_cam = None

        self.last_known = {
            "camera": None,
            "camera_id": None,
            "timestamp": None,
            "video_time": None,
            "confidence": 0.0,
            "snapshot_path": None,
            "time_epoch": None
        }

        self.target_name = "Unknown"
        self.search_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "reports")
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def set_target(self, target_name):
        """Sets the target name and resets the timeline for a new search."""
        with self._lock:
            self.target_name = target_name
            self.timeline.clear()
            self.journey_route.clear()
            self.current_camera_id = None
            self.current_camera_name = None
            self.last_transition_epoch = 0.0
            self.candidate_switch_count = 0
            self.candidate_switch_cam = None
            self.search_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_known = {
                "camera": None,
                "camera_id": None,
                "timestamp": None,
                "video_time": None,
                "confidence": 0.0,
                "snapshot_path": None,
                "time_epoch": None
            }

    def log_sighting(self, camera_name, confidence, snapshot_path=None, video_time=None, camera_id=0):
        """
        Logs a target sighting with smart station dwell and transit gating:
        - If target is still in the same camera: updates dwell time, NO duplicate hop drawn.
        - If target moves to a different camera: logs transition and creates the new path hop.
        - If target returns to an earlier camera: logs return transit and draws the return path.
        """
        now_epoch = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_time_str = video_time if video_time else now_str.split(" ")[1]

        with self._lock:
            # Always update real-time last known location
            self.last_known = {
                "camera": camera_name,
                "camera_id": camera_id,
                "timestamp": now_str,
                "video_time": video_time_str,
                "confidence": round(confidence, 3),
                "snapshot_path": snapshot_path or "",
                "time_epoch": now_epoch
            }

            # ── Scenario 1: First Ever Sighting (Trip Origin) ──
            if self.current_camera_id is None:
                self.current_camera_id = camera_id
                self.current_camera_name = camera_name
                self.last_transition_epoch = now_epoch
                self.candidate_switch_count = 0

                entry = {
                    "index": 1,
                    "type": "ORIGIN",
                    "camera": camera_name,
                    "camera_id": camera_id,
                    "entry_time": video_time_str,
                    "last_seen_time": video_time_str,
                    "dwell_str": "Present",
                    "timestamp": now_str,
                    "video_time": video_time_str,
                    "confidence": round(confidence, 3),
                    "snapshot_path": snapshot_path or "",
                    "time_epoch": now_epoch
                }
                self.timeline.append(entry)
                self.journey_route.append({
                    "step": 1,
                    "camera": camera_name,
                    "camera_id": camera_id,
                    "video_time": video_time_str,
                    "timestamp": now_str,
                    "confidence": round(confidence, 3),
                    "snapshot_path": snapshot_path or ""
                })
                return True

            # ── Scenario 2: Object Still in the Same Camera ──
            if camera_id == self.current_camera_id:
                self.candidate_switch_count = 0
                self.candidate_switch_cam = None

                # Update the active station entry's dwell time and peak confidence
                if self.timeline:
                    active_entry = self.timeline[-1]
                    active_entry["last_seen_time"] = video_time_str
                    if confidence > active_entry["confidence"]:
                        active_entry["confidence"] = round(confidence, 3)
                    if snapshot_path:
                        active_entry["snapshot_path"] = snapshot_path

                    start_t = active_entry.get("entry_time", video_time_str)
                    if start_t != video_time_str:
                        active_entry["dwell_str"] = f"{start_t} -> {video_time_str}"

                # Also update route destination exit time
                if self.journey_route:
                    self.journey_route[-1]["exit_time"] = video_time_str
                    if confidence > self.journey_route[-1]["confidence"]:
                        self.journey_route[-1]["confidence"] = round(confidence, 3)

                return False  # Target is stationary in current camera; no new hop!

            # ── Scenario 3: Potential Camera Transition (Transit / Return) ──
            # Anti-flicker filter: If seen in a different camera, require either:
            # 1. High confidence (>= 0.62)
            # 2. Or confirmed over 2 consecutive detections
            # 3. Or at least 2.0 seconds elapsed since previous station
            is_high_conf = confidence >= 0.62
            elapsed_since_prev = (now_epoch - self.last_transition_epoch)

            if not is_high_conf and elapsed_since_prev < 2.0:
                if self.candidate_switch_cam == camera_id:
                    self.candidate_switch_count += 1
                else:
                    self.candidate_switch_cam = camera_id
                    self.candidate_switch_count = 1

                # Require 2 consecutive hits to confirm transit if confidence is moderate
                if self.candidate_switch_count < 2:
                    return False

            # Confirmed real transition to the new camera!
            from_cam_name = self.current_camera_name
            self.current_camera_id = camera_id
            self.current_camera_name = camera_name
            self.last_transition_epoch = now_epoch
            self.candidate_switch_count = 0
            self.candidate_switch_cam = None

            # Check if this is returning to a previously visited camera
            visited_cams = [r["camera_id"] for r in self.journey_route]
            is_return = camera_id in visited_cams
            event_type = "RETURN" if is_return else "TRANSIT"

            entry = {
                "index": len(self.timeline) + 1,
                "type": event_type,
                "camera": camera_name,
                "camera_id": camera_id,
                "from_camera": from_cam_name,
                "entry_time": video_time_str,
                "last_seen_time": video_time_str,
                "dwell_str": "Arrived",
                "timestamp": now_str,
                "video_time": video_time_str,
                "confidence": round(confidence, 3),
                "snapshot_path": snapshot_path or "",
                "time_epoch": now_epoch
            }
            self.timeline.append(entry)

            self.journey_route.append({
                "step": len(self.journey_route) + 1,
                "camera": camera_name,
                "camera_id": camera_id,
                "from_camera": from_cam_name,
                "is_return": is_return,
                "video_time": video_time_str,
                "timestamp": now_str,
                "confidence": round(confidence, 3),
                "snapshot_path": snapshot_path or ""
            })
            return True

    def get_last_known_location(self):
        """Returns the dictionary representing the last known sighting location."""
        with self._lock:
            return dict(self.last_known)

    def get_journey_route(self):
        """
        Extracts the condensed sequence of unique camera transitions.
        Example: [Cam1, Cam2, Cam1]
        """
        with self._lock:
            return list(self.journey_route)

    def get_journey_arrow_str(self):
        """Returns clean single-line route: e.g. [Camera 1] -> [Camera 2] -> [Camera 1]"""
        route = self.get_journey_route()
        if not route:
            return "No movement recorded yet."
        return " -> ".join([f"[{r['camera']}]" for r in route])

    def get_formatted_journey(self):
        """
        Generates a clear visual path showing origin, transit points, and destination.
        """
        route = self.get_journey_route()
        if not route:
            return "No sightings recorded yet."

        lines = []
        lines.append("=" * 48)
        lines.append("        COMPLETE MOVEMENT JOURNEY PATH")
        lines.append("=" * 48)
        lines.append("")

        total_steps = len(route)
        for i, step in enumerate(route):
            cam = step["camera"]
            vt = step.get("video_time", "")
            ts = step["timestamp"].split(" ")[1] if " " in step["timestamp"] else step["timestamp"]
            conf = int(step["confidence"] * 100)
            exit_t = step.get("exit_time", "")
            is_ret = step.get("is_return", False)

            if exit_t and exit_t != vt:
                time_label = f"Video: {vt} -> {exit_t}"
            else:
                time_label = f"Video: {vt} (Clock: {ts})" if vt else f"Clock: {ts}"

            if i == 0:
                header = f"[START / ORIGIN] (Step 1):"
            elif i == total_steps - 1:
                header = f"[DESTINATION / LAST SEEN] (Step {i+1}):"
            elif is_ret:
                header = f"[RETURN STATION] (Step {i+1}):"
            else:
                header = f"[TRANSIT STATION] (Step {i+1}):"

            lines.append(f" {header}")
            lines.append(f"    Location: {cam}")
            lines.append(f"    Time:     {time_label}  |  Confidence: {conf}%")
            lines.append("")

            if i < total_steps - 1:
                lines.append("               |")
                lines.append("               v")
                lines.append("")

        lines.append("-" * 48)
        lines.append(f"ROUTE: {self.get_journey_arrow_str()}")
        lines.append("-" * 48)
        return "\n".join(lines)

    def get_movement_trail(self):
        """Returns detailed chronological sightings list."""
        with self._lock:
            if not self.timeline:
                return "No sightings recorded yet."

            lines = []
            for i, entry in enumerate(self.timeline):
                vt = entry.get("video_time", "")
                ts = entry["timestamp"].split(" ")[1] if " " in entry["timestamp"] else entry["timestamp"]
                time_str = f"[{vt}]" if vt else f"[{ts}]"
                is_last = (i == len(self.timeline) - 1)
                suffix = "  <- LAST SEEN" if is_last else ""
                etype = entry.get("type", "TRANSIT")
                dwell = entry.get("dwell_str", "")
                dwell_info = f" ({dwell})" if dwell and dwell not in ["Present", "Arrived"] else ""
                lines.append(
                    f" #{entry['index']:>2} {time_str:<10} {entry['camera']:<22} "
                    f"({int(entry['confidence']*100)}%){dwell_info}{suffix}"
                )
            return "\n".join(lines)

    def get_formatted_timeline(self):
        """Returns clean formatted timeline with station dwell for GUI display."""
        with self._lock:
            if not self.timeline:
                return "No target sightings recorded yet.\nStart scanning to begin tracking."

            lines = []
            for i, entry in enumerate(self.timeline):
                vt = entry.get("video_time", "")
                cam = entry["camera"]
                conf = int(entry["confidence"] * 100)
                etype = entry.get("type", "SIGHTING")
                dwell = entry.get("dwell_str", "")

                is_last = (i == len(self.timeline) - 1)
                marker = "[ACTIVE]" if is_last else "[PASS]"

                if etype == "ORIGIN":
                    lines.append(f"{marker} [{vt}] FIRST SIGHTING: {cam} ({conf}%)")
                    if dwell and dwell != "Present":
                        lines.append(f"     Station Dwell: {dwell}")
                elif etype == "RETURN":
                    from_c = entry.get("from_camera", "")
                    lines.append(f"{marker} [{vt}] <- RETURN TO: {cam} (from {from_c}) ({conf}%)")
                    if dwell and dwell != "Arrived":
                        lines.append(f"     Station Dwell: {dwell}")
                else:
                    from_c = entry.get("from_camera", "")
                    lines.append(f"{marker} [{vt}] -> TRANSIT TO: {cam} (from {from_c}) ({conf}%)")
                    if dwell and dwell != "Arrived":
                        lines.append(f"     Station Dwell: {dwell}")
                lines.append("")

            return "\n".join(lines)

    def get_timeline_summary(self):
        with self._lock:
            elapsed = None
            if self.last_known["time_epoch"]:
                s = int(time.time() - self.last_known["time_epoch"])
                elapsed = f"{s}s" if s < 60 else f"{s // 60}m {s % 60}s"

            return {
                "total_sightings": len(self.timeline),
                "last_known_camera": self.last_known["camera"],
                "last_known_time": self.last_known["timestamp"],
                "time_since_last_seen": elapsed,
                "route_summary": self.get_journey_arrow_str()
            }

    def seconds_since_last_seen(self):
        """Returns integer seconds elapsed since target was last sighted."""
        with self._lock:
            if not self.last_known["time_epoch"]:
                return None
            return int(time.time() - self.last_known["time_epoch"])

    def is_target_lost(self):
        with self._lock:
            if not self.last_known["time_epoch"]:
                return False
            return (time.time() - self.last_known["time_epoch"]) > self.lost_timeout_seconds

    def export_to_csv_and_json(self):
        """Exports timeline and complete journey path to CSV and JSON."""
        csv_path = self.export_to_csv()
        json_path = self.export_to_json()
        return csv_path, json_path

    def export_to_csv(self):
        with self._lock:
            if not self.timeline:
                return None

            filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_path.csv"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["index", "type", "camera", "camera_id", "from_camera", "entry_time", "last_seen_time", "dwell_str", "video_time", "timestamp", "confidence", "snapshot_path"]
                )
                writer.writeheader()
                for entry in self.timeline:
                    writer.writerow({k: v for k, v in entry.items() if k != "time_epoch"})
            return filepath

    def export_to_json(self):
        with self._lock:
            if not self.timeline:
                return None

            filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_path.json"
            filepath = os.path.join(self.output_dir, filename)

            report = {
                "target_name": self.target_name,
                "search_started_at": self.search_started_at,
                "complete_journey_route": list(self.journey_route),
                "route_arrow_summary": self.get_journey_arrow_str(),
                "last_known_location": dict(self.last_known),
                "total_sightings": len(self.timeline),
                "timeline": [
                    {k: v for k, v in e.items() if k != "time_epoch"}
                    for e in self.timeline
                ]
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return filepath
