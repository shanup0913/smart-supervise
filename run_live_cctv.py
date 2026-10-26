import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.cctv_scanner import CCTVScanner

def main():
    print("=" * 60)
    print("      SMART SUPERVISE - LIVE CCTV & VIDEO TESTER")
    print("=" * 60)
    print("Choose video source:")
    print("  [1] Live Computer Webcam (Default)")
    print("  [2] Enter IP Camera Stream (RTSP / HTTP)")
    print("  [3] Test with Sample Recorded Video File")
    
    choice = input("\nEnter choice (1, 2, or 3) [Default 1]: ").strip()
    
    if choice == "2":
        source = input("Enter RTSP or HTTP camera stream URL: ").strip()
        cam_name = "IP Camera 1"
    elif choice == "3":
        video_path = input("Enter path to video file (.mp4/.avi): ").strip()
        source = video_path.strip('"').strip("'")
        cam_name = os.path.basename(source)
    else:
        source = 0
        cam_name = "Live Webcam (Cam 0)"

    print(f"\n[INFO] Connecting to: {cam_name}...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    scanner = CCTVScanner()
    
    print("\n" + "=" * 60)
    print("KEYBOARD SHORTCUTS:")
    print("  Press 's' -> CAPTURE CURRENT PERSON as TARGET TO SEARCH!")
    print("  Press 'd' -> Switch to DOG / ANIMAL INTRUSION MODE")
    print("  Press 'p' -> Switch back to PERSON SEARCH MODE")
    print("  Press 'q' -> EXIT")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or no frame received.")
            break

        # Process frame with YOLOv8 + TargetMatcher
        annotated_frame, spotted = scanner.process_frame(frame, camera_name=cam_name)

        # Show on desktop window
        cv2.imshow("Smart Supervise - CCTV Feed (Press 'q' to quit)", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Freeze and grab the largest person in frame as target
            results = scanner.model(frame, verbose=False, conf=0.4)[0]
            person_boxes = [b for b in results.boxes if int(b.cls[0].item()) == 0]
            if person_boxes:
                # pick largest person
                b = person_boxes[0]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    scanner.set_person_target(crop, person_name="Captured Target")
                    print("\n[SUCCESS] New Target Captured and set from camera!")
            else:
                print("\n[WARNING] No person detected in frame to capture! Stand in front of camera and press 's'.")
        elif key == ord('d'):
            scanner.set_class_target("dog")
            print("\n[MODE] Switched to DOG / ANIMAL INTRUSION DETECTION!")
        elif key == ord('p'):
            print("\n[MODE] Switched to PERSON SEARCH MODE! Press 's' to capture a target.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] CCTV Scanner closed safely.")

if __name__ == "__main__":
    main()
