import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.reid_matcher import TargetMatcher

def create_demo_figure(color_shirt, color_pants, label):
    """Draws a clean humanoid avatar with specific clothing colors."""
    canvas = np.full((320, 160, 3), 245, dtype=np.uint8)
    
    # Head
    cv2.circle(canvas, (80, 50), 28, (200, 180, 160), -1)
    cv2.circle(canvas, (80, 50), 28, (50, 50, 50), 2)
    
    # Torso (Shirt)
    cv2.rectangle(canvas, (45, 85), (115, 195), color_shirt, -1)
    cv2.rectangle(canvas, (45, 85), (115, 195), (40, 40, 40), 2)
    
    # Legs (Pants)
    cv2.rectangle(canvas, (48, 195), (75, 300), color_pants, -1)
    cv2.rectangle(canvas, (85, 195), (112, 300), color_pants, -1)
    cv2.rectangle(canvas, (48, 195), (75, 300), (40, 40, 40), 2)
    cv2.rectangle(canvas, (85, 195), (112, 300), (40, 40, 40), 2)
    
    return canvas

def generate_display():
    matcher = TargetMatcher(match_threshold=0.70)

    # 1. Target Person: Red Jacket & Blue Jeans
    target_img = create_demo_figure((30, 30, 220), (180, 90, 20), "Target")
    matcher.set_target_photo(target_img, target_name="Target Person")

    # 2. Candidate 1 (Match): Same Red Jacket & Blue Jeans
    cand1_img = create_demo_figure((25, 25, 210), (170, 85, 25), "Candidate 1")
    is_match1, score1 = matcher.compare(cand1_img)

    # 3. Candidate 2 (Mismatch): Green Jacket & Black Pants
    cand2_img = create_demo_figure((40, 190, 40), (40, 40, 40), "Candidate 2")
    is_match2, score2 = matcher.compare(cand2_img)

    # Build side-by-side presentation display
    h, w = 450, 720
    display = np.full((h, w, 3), 30, dtype=np.uint8) # Dark modern background

    # Header
    cv2.rectangle(display, (0, 0), (w, 60), (45, 45, 45), -1)
    cv2.putText(display, "SMART SUPERVISE - AI Photo Matcher (Step 1 Demo)", (20, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # Place Target Photo (Left)
    display[100:420, 40:200] = target_img
    cv2.rectangle(display, (38, 98), (202, 422), (255, 180, 0), 2)
    cv2.putText(display, "TARGET PHOTO", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
    cv2.putText(display, "(Uploaded by User)", (42, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Place Candidate 1 (Middle)
    display[100:420, 280:440] = cand1_img
    c1_color = (0, 230, 0) if is_match1 else (0, 0, 230)
    cv2.rectangle(display, (278, 98), (442, 422), c1_color, 3)
    cv2.putText(display, f"MATCH: {score1*100:.1f}%", (280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c1_color, 2)
    status1 = ">> TARGET FOUND <<" if is_match1 else "No Match"
    cv2.putText(display, status1, (285, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c1_color, 2)

    # Place Candidate 2 (Right)
    display[100:420, 520:680] = cand2_img
    c2_color = (0, 230, 0) if is_match2 else (0, 0, 220)
    cv2.rectangle(display, (518, 98), (682, 422), c2_color, 2)
    cv2.putText(display, f"MATCH: {score2*100:.1f}%", (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c2_color, 2)
    status2 = ">> TARGET FOUND <<" if is_match2 else "Different Person"
    cv2.putText(display, status2, (525, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Save visual snapshot
    snapshot_path = os.path.join(os.path.dirname(__file__), "step1_visual_demo.png")
    cv2.imwrite(snapshot_path, display)
    return display, snapshot_path

if __name__ == "__main__":
    display, snapshot_path = generate_display()
    print(f"Snapshot image generated: {snapshot_path}")
    print("\n[INFO] Opening display window on your screen! Press any key or 'q' to close it.")
    cv2.imshow("Smart Supervise - Step 1 Verification", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
