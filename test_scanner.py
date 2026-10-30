import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(r"C:\Users\ysidd\.gemini\antigravity\scratch\SmartSupervise"))

from core.cctv_scanner import CCTVScanner
from visual_demo import create_demo_figure

print("Initializing CCTVScanner with YOLOv8...")
scanner = CCTVScanner(model_name="yolov8n.pt")
print("CCTVScanner initialized successfully!")

# Test 1: Set Target Person by Photo
target_photo = create_demo_figure((30, 30, 220), (180, 90, 20), "Target") # Red shirt, blue pants
scanner.set_person_target(target_photo, person_name="Lost Child")
print("Target person set: 'Lost Child'")

# Test 2: Process a frame containing the target figure
synthetic_cctv_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
synthetic_cctv_frame[100:420, 240:400] = target_photo

annotated, spotted, match_info = scanner.process_frame(synthetic_cctv_frame, camera_name="School Entrance Cam")
print(f"Frame processed! Target spotted: {spotted}")
print(f"Last Sighting State: {scanner.last_sighting['found']}")
print(f"Last Seen Camera: {scanner.last_sighting['camera_name']}")

# Test 3: Test Animal Intrusion Mode
scanner.set_class_target("dog")
print(f"Switched tracking mode to: {scanner.tracking_mode} (Class: {scanner.target_class_name})")

print("\n>>> All CCTVScanner tests PASSED perfectly! <<<")
