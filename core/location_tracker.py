import os
import csv
import time
import json
from datetime import datetime


class LocationTracker:
    """
    LocationTracker: Builds a multi-camera movement timeline and calculates
    the complete journey path (e.g. Camera 1 -> Camera 4 -> Camera 3).
    
    Features:
    - Records every sighting: camera name, video timestamp (HH:MM:SS), clock timestamp, confidence, snapshot
    - Builds a condensed journey route (Origin -> Transits -> Destination)
    - Exports the timeline and path summary to CSV & JSON
    - Detects lost target timeout
    """
    def __init__(self, lost_timeout_seconds=30, session_name=None):
        self.lost_timeout_seconds = lost_timeout_seconds
        self.session_name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self.timeline = []

        self.last_known = {
            "camera": None,
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
        self.target_name = target_name
        self.timeline.clear()
        self.search_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_known = {
            "camera": None,
            "timestamp": None,
            "video_time": None,
            "confidence": 0.0,
            "snapshot_path": None,
            "time_epoch": None
        }

    def log_sighting(self, camera_name, confidence, snapshot_path=None, video_time=None):
        """
        Logs a confirmed sighting of the target.
        Throttles consecutive entries for the same camera to avoid spam,
        while always capturing camera changes immediately.
        """
        now_epoch = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_time_str = video_time if video_time else now_str.split(" ")[1]

        if self.timeline:
            last = self.timeline[-1]
            same_camera = (last["camera"] == camera_name)
            within_2s = (now_epoch - last["time_epoch"]) < 2.0
            if same_camera and within_2s:
                if confidence > last["confidence"]:
                    last["confidence"] = round(confidence, 3)
                return False

        entry = {
            "index": len(self.timeline) + 1,
            "camera": camera_name,
            "timestamp": now_str,
            "video_time": video_time_str,
            "confidence": round(confidence, 3),
            "snapshot_path": snapshot_path or "",
            "time_epoch": now_epoch
        }
        self.timeline.append(entry)

        self.last_known = {
            "camera": camera_name,
            "timestamp": now_str,
            "video_time": video_time_str,
            "confidence": round(confidence, 3),
            "snapshot_path": snapshot_path or "",
            "time_epoch": now_epoch
        }

        return True

    def get_journey_route(self):
        """
        Extracts the condensed sequence of unique camera transitions.
        Example: [Cam1, Cam1, Cam4, Cam4, Cam3] -> [Cam1, Cam4, Cam3]
        """
        if not self.timeline:
            return []

        route = []
        for entry in self.timeline:
            if not route or route[-1]["camera"] != entry["camera"]:
                route.append({
                    "step": len(route) + 1,
                    "camera": entry["camera"],
                    "video_time": entry.get("video_time", ""),
                    "timestamp": entry["timestamp"],
                    "confidence": entry["confidence"],
                    "snapshot_path": entry.get("snapshot_path", "")
                })
            else:
                # Update exit timestamp/confidence for the same camera stop
                route[-1]["exit_time"] = entry.get("video_time", "")
                if entry["confidence"] > route[-1]["confidence"]:
                    route[-1]["confidence"] = entry["confidence"]
        return route

    def get_journey_arrow_str(self):
        """Returns clean single-line route: e.g. [Camera 1] -> [Camera 4] -> [Camera 3]"""
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

            time_label = f"Video: {vt} (Clock: {ts})" if vt else f"Clock: {ts}"

            if i == 0:
                header = f"[START / ORIGIN] (Step 1):"
            elif i == total_steps - 1:
                header = f"[DESTINATION / LAST SEEN] (Step {i+1}):"
            else:
                header = f"[TRANSIT POINT] (Step {i+1}):"

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
        if not self.timeline:
            return "No sightings recorded yet."

        lines = []
        for i, entry in enumerate(self.timeline):
            vt = entry.get("video_time", "")
            ts = entry["timestamp"].split(" ")[1] if " " in entry["timestamp"] else entry["timestamp"]
            time_str = f"[{vt}]" if vt else f"[{ts}]"
            is_last = (i == len(self.timeline) - 1)
            suffix = "  ← LAST SEEN" if is_last else ""
            lines.append(
                f" #{entry['index']:>2} {time_str:<10} {entry['camera']:<24} "
                f"({int(entry['confidence']*100)}%){suffix}"
            )
        return "\n".join(lines)

    def get_timeline_summary(self):
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

    def is_target_lost(self):
        if not self.last_known["time_epoch"]:
            return False
        return (time.time() - self.last_known["time_epoch"]) > self.lost_timeout_seconds

    def export_to_csv_and_json(self):
        """Exports timeline and complete journey path to CSV and JSON."""
        csv_path = self.export_to_csv()
        json_path = self.export_to_json()
        return csv_path, json_path

    def export_to_csv(self):
        if not self.timeline:
            return None

        filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_path.csv"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "camera", "video_time", "timestamp", "confidence", "snapshot_path"]
            )
            writer.writeheader()
            for entry in self.timeline:
                writer.writerow({k: v for k, v in entry.items() if k != "time_epoch"})
        return filepath

    def export_to_json(self):
        if not self.timeline:
            return None

        filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_path.json"
        filepath = os.path.join(self.output_dir, filename)

        report = {
            "target_name": self.target_name,
            "search_started_at": self.search_started_at,
            "complete_journey_route": self.get_journey_route(),
            "route_arrow_summary": self.get_journey_arrow_str(),
            "last_known_location": self.last_known,
            "total_sightings": len(self.timeline),
            "timeline": [
                {k: v for k, v in e.items() if k != "time_epoch"}
                for e in self.timeline
            ]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return filepath
