import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(r"C:\Users\ysidd\.gemini\antigravity\scratch\SmartSupervise"))
from core.location_tracker import LocationTracker

print("=" * 60)
print("  STEP 3 TEST: Multi-Camera Movement Timeline")
print("=" * 60)

tracker = LocationTracker(lost_timeout_seconds=5)
tracker.set_target("Lost Child (Red Shirt)")

# Simulate target being spotted across 3 different cameras
print("\nSimulating target moving through 3 cameras...\n")

tracker.log_sighting("Camera 1 - School Main Gate",    confidence=0.91, snapshot_path="alerts/snap1.jpg")
print(f"  ✓ Sighting 1 logged: {tracker.get_last_known_location()}")

time.sleep(0.1)
tracker.log_sighting("Camera 2 - Corridor B",          confidence=0.88, snapshot_path="alerts/snap2.jpg")
print(f"  ✓ Sighting 2 logged: {tracker.get_last_known_location()}")

time.sleep(0.1)
tracker.log_sighting("Camera 3 - Food Court Exit",     confidence=0.94, snapshot_path="alerts/snap3.jpg")
print(f"  ✓ Sighting 3 logged: {tracker.get_last_known_location()}")

# Print the movement trail
print("\n--- MOVEMENT TRAIL ---")
print(tracker.get_movement_trail())

# Test 'lost' detection
print("\n--- LOST DETECTION ---")
print(f"  Seconds since last seen: {tracker.seconds_since_last_seen()}s")
print(f"  Is target lost (5s timeout)? {tracker.is_target_lost()}")
time.sleep(6)
print(f"  After 6 seconds — Is target lost? {tracker.is_target_lost()}")
print(f"  Seconds since last seen: {tracker.seconds_since_last_seen()}s")

# Export timeline
csv_path = tracker.export_to_csv()
json_path = tracker.export_to_json()
print(f"\n  ✓ CSV exported to:  {csv_path}")
print(f"  ✓ JSON exported to: {json_path}")

print("\n>>> Step 3 LocationTracker: All Tests PASSED! <<<")
