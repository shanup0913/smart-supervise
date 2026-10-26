import os
import csv
import time
import json
from datetime import datetime


class LocationTracker:
    """
    LocationTracker: Builds a movement timeline for the target person
    across multiple cameras and maintains the Last Known Location.
    
    Features:
    - Records every sighting: camera name, timestamp, match confidence, snapshot path
    - Builds an ordered movement trail (Camera 1 → Camera 2 → Last Seen Camera)
    - Exports the timeline to CSV / JSON for reports
    - Detects if the target has NOT been seen for more than N seconds (lost alert)
    """
    def __init__(self, lost_timeout_seconds=30, session_name=None):
        # How many seconds without a sighting before declaring target "Lost"
        self.lost_timeout_seconds = lost_timeout_seconds

        # Session label for this search (used in file names)
        self.session_name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")

        # Ordered list of sighting events
        self.timeline = []

        # Last known location details
        self.last_known = {
            "camera": None,
            "timestamp": None,
            "confidence": 0.0,
            "snapshot_path": None,
            "time_epoch": None
        }

        # Target info
        self.target_name = "Unknown"
        self.search_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Output directory for logs and reports
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
            "confidence": 0.0,
            "snapshot_path": None,
            "time_epoch": None
        }

    def log_sighting(self, camera_name, confidence, snapshot_path=None):
        """
        Logs a confirmed sighting of the target.
        Call this every time the scanner confirms a match in a frame.
        Throttles to one log entry per 3 seconds per camera.
        """
        now_epoch = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Throttle: only log a new entry if camera changed OR 3 seconds have passed
        if self.timeline:
            last = self.timeline[-1]
            same_camera = last["camera"] == camera_name
            within_3s = (now_epoch - last["time_epoch"]) < 3.0
            if same_camera and within_3s:
                # Just silently update confidence if higher
                if confidence > last["confidence"]:
                    last["confidence"] = round(confidence, 3)
                return False  # Not a new log entry

        entry = {
            "index": len(self.timeline) + 1,
            "camera": camera_name,
            "timestamp": now_str,
            "confidence": round(confidence, 3),
            "snapshot_path": snapshot_path or "",
            "time_epoch": now_epoch
        }
        self.timeline.append(entry)

        # Update Last Known Location
        self.last_known = {
            "camera": camera_name,
            "timestamp": now_str,
            "confidence": round(confidence, 3),
            "snapshot_path": snapshot_path or "",
            "time_epoch": now_epoch
        }

        return True  # New entry was logged

    def get_last_known_location(self):
        """Returns a formatted string of the last known location."""
        if not self.last_known["camera"]:
            return "Target not yet found."
        return (
            f"Last Seen: {self.last_known['camera']}  |  "
            f"{self.last_known['timestamp']}  |  "
            f"Confidence: {int(self.last_known['confidence'] * 100)}%"
        )

    def is_target_lost(self):
        """Returns True if no sighting has been logged for more than lost_timeout_seconds."""
        if not self.last_known["time_epoch"]:
            return False  # Never found yet — not 'lost', just 'searching'
        elapsed = time.time() - self.last_known["time_epoch"]
        return elapsed > self.lost_timeout_seconds

    def seconds_since_last_seen(self):
        """Returns how many seconds have passed since the last confirmed sighting."""
        if not self.last_known["time_epoch"]:
            return None
        return round(time.time() - self.last_known["time_epoch"], 1)

    def get_movement_trail(self):
        """
        Returns a clean, human-readable movement trail string.
        Example:
          #1  School Main Gate   →   14:30:10  (91%)
          #2  Corridor B         →   14:35:40  (88%)
          #3  Food Court Exit    →   14:42:05  (94%)  ← LAST KNOWN LOCATION
        """
        if not self.timeline:
            return "No sightings recorded yet."

        lines = []
        for i, entry in enumerate(self.timeline):
            ts = entry["timestamp"].split(" ")[1]  # Show just the time HH:MM:SS
            is_last = (i == len(self.timeline) - 1)
            suffix = "  ← LAST KNOWN LOCATION" if is_last else ""
            lines.append(
                f"  #{entry['index']:>2}  {entry['camera']:<28}  {ts}  "
                f"({int(entry['confidence']*100)}%){suffix}"
            )
        return "\n".join(lines)

    def export_to_csv(self):
        """Exports the full sightings timeline to a CSV file."""
        if not self.timeline:
            return None
        
        filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_timeline.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["index", "camera", "timestamp", "confidence", "snapshot_path"]
            )
            writer.writeheader()
            for entry in self.timeline:
                writer.writerow({k: v for k, v in entry.items() if k != "time_epoch"})
        
        return filepath

    def export_to_json(self):
        """Exports the full sightings timeline to a JSON file."""
        if not self.timeline:
            return None

        filename = f"{self.session_name}_{self.target_name.replace(' ', '_')}_timeline.json"
        filepath = os.path.join(self.output_dir, filename)

        report = {
            "target_name": self.target_name,
            "search_started_at": self.search_started_at,
            "last_known_location": self.last_known["camera"],
            "last_seen_at": self.last_known["timestamp"],
            "total_sightings": len(self.timeline),
            "timeline": [
                {k: v for k, v in e.items() if k != "time_epoch"}
                for e in self.timeline
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return filepath

    def clear(self):
        """Resets the tracker for a new search session."""
        self.timeline.clear()
        self.last_known = {
            "camera": None,
            "timestamp": None,
            "confidence": 0.0,
            "snapshot_path": None,
            "time_epoch": None
        }
