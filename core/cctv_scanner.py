import cv2
import time
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from core.reid_matcher import TargetMatcher

class CCTVScanner:
    """
    CCTVScanner: Real-time video analysis and target tracking engine.
    
    Supports:
    - Live Webcams (0, 1)
    - IP CCTV Camera Streams (RTSP / HTTP)
    - Recorded Video Files (.mp4, .avi, .mkv)
    
    Capabilities:
    1. Search for a specific Person by reference photo (Re-ID)
    2. Detect specific unauthorized classes (e.g., Stray Dog, Animal Intrusion)
    3. Keep track of Last Known Location, Timestamps, and Sightings
    """
    def __init__(self, model_name=None, match_threshold=0.72):
        if model_name is None:
            local_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt"))
            model_name = local_model if os.path.exists(local_model) else "yolov8n.pt"

        # Load lightweight YOLOv8 (nano model for fast CPU execution)
        self.model = YOLO(model_name)
        self.matcher = TargetMatcher(match_threshold=match_threshold)
        
        # Tracking mode: 'person_photo' (search by photo) or 'class_alert' (e.g. dog intrusion)
        self.tracking_mode = "person_photo"
        self.target_class_name = "person" # default target class
        
        # Last Known Location state
        self.last_sighting = {
            "found": False,
            "target_name": None,
            "camera_name": "Camera 1",
            "timestamp": None,
            "confidence": 0.0,
            "snapshot": None,
            "bbox": None
        }
        
        # Sightings timeline log
        self.sightings_history = []
        
        # Directory for saving alert snapshots
        self.alerts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "alerts"))
        os.makedirs(self.alerts_dir, exist_ok=True)

    def set_person_target(self, photo_path_or_crop, person_name="Target Person"):
        """Sets a target person by reference photo."""
        self.tracking_mode = "person_photo"
        self.matcher.set_target_photo(photo_path_or_crop, target_name=person_name)
        self.last_sighting["found"] = False
        self.last_sighting["target_name"] = person_name
        self.sightings_history.clear()
        return True

    def set_class_target(self, class_name="dog"):
        """Sets tracking for an animal/intruder class (e.g. 'dog' for school safety)."""
        self.tracking_mode = "class_alert"
        self.target_class_name = class_name.lower()
        self.matcher.clear_target()
        self.last_sighting["found"] = False
        self.last_sighting["target_name"] = f"Intruder ({class_name.capitalize()})"
        self.sightings_history.clear()
        return True

    def process_frame(self, frame, camera_name="Camera 1"):
        """
        Analyzes a single video frame with YOLO + Matcher.
        Returns annotated frame with bounding boxes and alert status.
        """
        if frame is None:
            return None, False

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        target_spotted_in_frame = False
        best_match_info = None

        # Run YOLO detection (classes 0=person, 15=cat, 16=dog, etc.)
        results = self.model(frame, verbose=False, conf=0.35)[0]
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Clip coordinates to frame boundary
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Case A: Searching for a specific Person by Photo
            if self.tracking_mode == "person_photo" and cls_name == "person":
                is_match, similarity = self.matcher.compare(crop)
                
                if is_match:
                    target_spotted_in_frame = True
                    # Bright Green Box for Found Target
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"TARGET FOUND: {self.matcher.target_name} ({int(similarity*100)}%)"
                    
                    # Label banner
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 28)), (x1 + len(label) * 11, y1), (0, 200, 0), -1)
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                    
                    best_match_info = {
                        "confidence": similarity,
                        "crop": crop,
                        "bbox": (x1, y1, x2, y2),
                        "target_name": self.matcher.target_name
                    }
                else:
                    # Clear Mismatch Indicator showing exact similarity score
                    match_pct = int(similarity * 100)
                    label = f"MISMATCH ({match_pct}%)"
                    box_color = (40, 50, 210)  # Distinct Red/Orange for mismatch
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    text_w = len(label) * 9 + 8
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + text_w, y1), box_color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 4, y1 - 6), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Case B: Animal Intrusion / Specific Class Detection (e.g. Dog in school)
            elif self.tracking_mode == "class_alert" and cls_name == self.target_class_name:
                target_spotted_in_frame = True
                # Red Alert Box for Intrusion
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"ALERT: {cls_name.upper()} DETECTED! ({int(conf*100)}%)"
                
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 28)), (x1 + len(label) * 11, y1), (0, 0, 220), -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                best_match_info = {
                    "confidence": conf,
                    "crop": crop,
                    "bbox": (x1, y1, x2, y2),
                    "target_name": f"{cls_name.capitalize()} Intrusion"
                }

        # Update Last Known Location if target was spotted
        if target_spotted_in_frame and best_match_info:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_sighting["found"] = True
            self.last_sighting["camera_name"] = camera_name
            self.last_sighting["timestamp"] = now_str
            self.last_sighting["confidence"] = best_match_info["confidence"]
            self.last_sighting["snapshot"] = best_match_info["crop"]
            self.last_sighting["bbox"] = best_match_info["bbox"]
            self.last_sighting["target_name"] = best_match_info["target_name"]
            
            # Save sighting log entry (throttle: only log once every 2 seconds)
            if not self.sightings_history or (time.time() - self.sightings_history[-1].get("time_epoch", 0)) > 2.0:
                snapshot_filename = f"sighting_{int(time.time())}.jpg"
                snapshot_path = os.path.join(self.alerts_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, best_match_info["crop"])
                
                entry = {
                    "camera": camera_name,
                    "timestamp": now_str,
                    "confidence": round(best_match_info["confidence"], 2),
                    "snapshot_file": snapshot_path,
                    "time_epoch": time.time()
                }
                self.sightings_history.append(entry)

        # Overlay status bar at bottom of frame
        self._draw_status_overlay(annotated_frame, target_spotted_in_frame, camera_name)
        return annotated_frame, target_spotted_in_frame

    def _draw_status_overlay(self, frame, spotted_now, camera_name):
        """Draws a sleek status banner showing current camera & Last Known Location."""
        h, w = frame.shape[:2]
        banner_h = 42
        
        # Bottom banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Camera tag (Left)
        cv2.putText(frame, f"CAM: {camera_name}", (12, h - 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # Last Known Location tag (Right)
        if self.last_sighting["found"]:
            status_text = f"LAST SEEN: {self.last_sighting['camera_name']} @ {self.last_sighting['timestamp']}"
            tag_color = (0, 255, 0) if spotted_now else (0, 180, 255)
        else:
            status_text = "SEARCHING FOR TARGET..."
            tag_color = (180, 180, 180)

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, status_text, (w - text_size[0] - 15, h - 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 1)
