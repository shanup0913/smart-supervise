import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(r"C:\Users\ysidd\.gemini\antigravity\scratch\SmartSupervise"))

from core.reid_matcher import TargetMatcher

print("Testing TargetMatcher initialization...")
matcher = TargetMatcher()
print("Initialized successfully!")

# Create two sample dummy person images (e.g., Person in red shirt vs Person in blue shirt)
img_red = np.zeros((200, 100, 3), dtype=np.uint8)
img_red[:100, :] = [0, 0, 200]    # Red upper body
img_red[100:, :] = [100, 50, 0]   # Blue lower body

img_red_variant = np.zeros((200, 100, 3), dtype=np.uint8)
img_red_variant[:100, :] = [10, 10, 190]   # Slightly different red
img_red_variant[100:, :] = [90, 45, 10]    # Slightly different blue

img_green = np.zeros((200, 100, 3), dtype=np.uint8)
img_green[:100, :] = [0, 200, 0]   # Green upper body
img_green[100:, :] = [0, 0, 0]     # Black lower body

# 1. Set target photo
matcher.set_target_photo(img_red, target_name="Child in Red")
print("Target photo set: Child in Red")

# 2. Compare matching photo
is_match, score = matcher.compare(img_red_variant)
print(f"Compare against similar person: Match={is_match}, Score={score * 100:.1f}%")

# 3. Compare non-matching photo
is_match_diff, score_diff = matcher.compare(img_green)
print(f"Compare against different person: Match={is_match_diff}, Score={score_diff * 100:.1f}%")

assert score > score_diff, "Similar person should have significantly higher score than different person!"
print("\n>>> All TargetMatcher tests PASSED perfectly! <<<")
