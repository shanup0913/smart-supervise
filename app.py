import sys
import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reid_matcher import TargetMatcher
from core.cctv_scanner import CCTVScanner
from core.location_tracker import LocationTracker

# ─── App Theme ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DARK_BG      = "#1a1a2e"
PANEL_BG     = "#16213e"
CARD_BG      = "#0f3460"
ACCENT_BLUE  = "#4361ee"
ACCENT_GREEN = "#06d6a0"
ACCENT_RED   = "#ef233c"
ACCENT_AMBER = "#f9c74f"
TEXT_LIGHT   = "#e0e0e0"
TEXT_DIM     = "#888899"


class SmartSuperviseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # ── Window Setup ────────────────────────────────────────────────────
        self.title("Smart Supervise — Multi-Camera AI CCTV Intelligence System")
        self.geometry("1340x820")
        self.minsize(1100, 720)
        self.configure(fg_color=DARK_BG)

        # ── Core AI Components ──────────────────────────────────────────────
        self.scanner  = CCTVScanner()
        self.tracker  = LocationTracker(lost_timeout_seconds=30)

        # ── State Variables ──────────────────────────────────────────────────
        self.is_running     = False
        self.scan_thread    = None
        self.target_photo   = None  # numpy BGR
        self.tracking_mode  = ctk.StringVar(value="person")  # "person" or "class"
        self.view_mode      = ctk.StringVar(value="grid")    # "grid", "cam0", "cam1", "cam2", "cam3"

        # ── 4 Camera Slots ──────────────────────────────────────────────────
        default_locations = [
            "Floor 1 - Main Gate",
            "Floor 1 - Corridor",
            "Floor 2 - Room 204",
            "Ground Floor - Cafeteria"
        ]
        self.camera_slots = []
        for i in range(4):
            self.camera_slots.append({
                "id": i,
                "enabled": ctk.BooleanVar(value=(i == 0)), # Cam 1 active by default
                "source_var": ctk.StringVar(value="0" if i == 0 else ""),
                "location_var": ctk.StringVar(value=default_locations[i]),
                "cap": None,
                "current_frame": None,
                "last_annotated": None,
                "spotted": False,
                "frame_count": 0
            })

        # ── Build Layout ─────────────────────────────────────────────────────
        self._build_titlebar()
        self._build_main_layout()
        self._update_last_seen_banner()

        # Close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT BUILDERS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_titlebar(self):
        bar = ctk.CTkFrame(self, fg_color=PANEL_BG, height=52, corner_radius=0)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text="⬡  SMART SUPERVISE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT_BLUE
        ).pack(side="left", padx=18, pady=10)

        ctk.CTkLabel(
            bar, text="Multi-Camera AI CCTV Target Search & Biometric Tracking",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM
        ).pack(side="left", padx=0, pady=10)

        # Status indicator (top-right)
        self.status_dot = ctk.CTkLabel(
            bar, text="●  IDLE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_DIM
        )
        self.status_dot.pack(side="right", padx=18)

    def _build_main_layout(self):
        """3-column layout: Left (Setup) | Center (Multi-Feed Video) | Right (Timeline)"""
        main = ctk.CTkFrame(self, fg_color=DARK_BG)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        main.grid_columnconfigure(0, weight=0, minsize=290)  # Left panel
        main.grid_columnconfigure(1, weight=1)               # Center video
        main.grid_columnconfigure(2, weight=0, minsize=280)  # Right panel
        main.grid_rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_center_panel(main)
        self._build_right_panel(main)

    # ── Left Panel: Scrollable Configuration ────────────────────────────────
    def _build_left_panel(self, parent):
        panel = ctk.CTkScrollableFrame(parent, fg_color=PANEL_BG, corner_radius=12, width=280)
        panel.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="TARGET & CAMERAS SETUP",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(8, 6), padx=10, anchor="w")

        # ── 1. Target Mode Selector ──
        mode_frame = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        mode_frame.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(mode_frame, text="Tracking Mode", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(6, 2), padx=10, anchor="w")

        ctk.CTkRadioButton(
            mode_frame, text="Search Person (Face + Clothing)",
            variable=self.tracking_mode, value="person",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=11), fg_color=ACCENT_BLUE
        ).pack(anchor="w", padx=10, pady=3)

        ctk.CTkRadioButton(
            mode_frame, text="Animal / Intruder Alert",
            variable=self.tracking_mode, value="class",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=11), fg_color=ACCENT_RED
        ).pack(anchor="w", padx=10, pady=(3, 8))

        # ── Dynamic Target Container ──
        self.target_container = ctk.CTkFrame(panel, fg_color="transparent")
        self.target_container.pack(fill="x", padx=0, pady=0)

        # Class mode entry
        self.class_name_var = ctk.StringVar(value="dog")
        self.class_entry_frame = ctk.CTkFrame(self.target_container, fg_color=CARD_BG, corner_radius=10)
        ctk.CTkLabel(self.class_entry_frame, text="Alert Class (e.g. dog, cat)",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(pady=(8, 2), padx=10, anchor="w")
        ctk.CTkEntry(
            self.class_entry_frame, textvariable=self.class_name_var,
            width=210, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(0, 8), padx=10)

        # Person photo upload frame
        self.photo_frame = ctk.CTkFrame(self.target_container, fg_color=CARD_BG, corner_radius=10)
        self.photo_frame.pack(fill="x", padx=4, pady=(0, 8))

        ctk.CTkLabel(self.photo_frame, text="Target Photo", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(6, 4), padx=10, anchor="w")

        self.photo_canvas = ctk.CTkLabel(
            self.photo_frame, text="No Photo Uploaded\nClick Upload below",
            width=120, height=130,
            fg_color=DARK_BG, corner_radius=8,
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.photo_canvas.pack(pady=(0, 6), padx=10)

        ctk.CTkButton(
            self.photo_frame, text="Upload Target Photo",
            command=self._upload_photo,
            fg_color=ACCENT_BLUE, hover_color="#2d4bcc",
            corner_radius=8, height=30,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(fill="x", padx=10, pady=(0, 6))

        self.target_name_var = ctk.StringVar(value="Target Person")
        ctk.CTkLabel(self.photo_frame, text="Target Name",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(padx=10, anchor="w")
        ctk.CTkEntry(
            self.photo_frame, textvariable=self.target_name_var,
            width=210, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(2, 8), padx=10)

        # ── 2. Multi-Camera Slots (Up to 4 Videos) ──
        ctk.CTkLabel(
            panel, text="CCTV CAMERAS (UP TO 4)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(6, 4), padx=10, anchor="w")

        for i, cam in enumerate(self.camera_slots):
            cframe = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=8)
            cframe.pack(fill="x", padx=4, pady=(0, 6))

            # Header with Checkbox
            hdr = ctk.CTkFrame(cframe, fg_color="transparent")
            hdr.pack(fill="x", padx=8, pady=(4, 2))

            chk = ctk.CTkCheckBox(
                hdr, text=f"Camera {i+1}",
                variable=cam["enabled"],
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=ACCENT_GREEN,
                text_color=TEXT_LIGHT
            )
            chk.pack(side="left")

            # Source entry & Browse
            src_row = ctk.CTkFrame(cframe, fg_color="transparent")
            src_row.pack(fill="x", padx=8, pady=(2, 2))

            ctk.CTkEntry(
                src_row, textvariable=cam["source_var"],
                placeholder_text="0 | rtsp | video.mp4",
                width=140, height=26, fg_color=DARK_BG, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=(0, 4))

            ctk.CTkButton(
                src_row, text="Browse",
                command=lambda idx=i: self._browse_camera_video(idx),
                width=65, height=26, fg_color=DARK_BG, hover_color=PANEL_BG,
                border_width=1, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(side="left")

            # Location label
            ctk.CTkEntry(
                cframe, textvariable=cam["location_var"],
                placeholder_text="Room / Floor Location",
                width=210, height=26, fg_color=DARK_BG, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(padx=8, pady=(2, 6))

        # ── Start / Stop Buttons ──
        self.start_btn = ctk.CTkButton(
            panel, text="▶  START SCANNING",
            command=self._start_scanning,
            fg_color=ACCENT_GREEN, hover_color="#04a57a",
            text_color="#000000",
            corner_radius=10, height=38,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.start_btn.pack(fill="x", padx=4, pady=(6, 4))

        self.stop_btn = ctk.CTkButton(
            panel, text="■  STOP",
            command=self._stop_scanning,
            fg_color="#333355", hover_color=ACCENT_RED,
            corner_radius=10, height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.stop_btn.pack(fill="x", padx=4, pady=(0, 10))

    # ── Center Panel: Multi-Camera Video Grid ───────────────────────────────
    def _build_center_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=12)
        panel.grid(row=0, column=1, padx=4, pady=0, sticky="nsew")

        # Top Control Bar: Mode Toggle (Grid vs Single Camera) & FPS
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            header, text="CCTV SURVEILLANCE MATRIX",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_GREEN
        ).pack(side="left")

        # View Mode Segmented Buttons
        self.view_buttons = ctk.CTkSegmentedButton(
            header,
            values=["⊞ 4-Grid", "Cam 1", "Cam 2", "Cam 3", "Cam 4"],
            command=self._on_view_mode_change,
            selected_color=ACCENT_BLUE,
            height=26,
            font=ctk.CTkFont(size=11)
        )
        self.view_buttons.set("⊞ 4-Grid")
        self.view_buttons.pack(side="left", padx=14)

        self.fps_label = ctk.CTkLabel(
            header, text="FPS: —",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.fps_label.pack(side="right")

        # ── 2x2 Grid Container ──
        self.grid_container = ctk.CTkFrame(panel, fg_color="transparent")
        self.grid_container.pack(fill="both", expand=True, padx=8, pady=(2, 6))

        self.grid_container.grid_rowconfigure(0, weight=1)
        self.grid_container.grid_rowconfigure(1, weight=1)
        self.grid_container.grid_columnconfigure(0, weight=1)
        self.grid_container.grid_columnconfigure(1, weight=1)

        self.tiles = []
        for i in range(4):
            r = i // 2
            c = i % 2
            tile_card = ctk.CTkFrame(self.grid_container, fg_color=DARK_BG, corner_radius=8, border_width=1, border_color="#22223a")
            tile_card.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

            # Tile header
            hdr = ctk.CTkLabel(
                tile_card, text=f"CAM {i+1} — Idle",
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=CARD_BG, height=22, corner_radius=6,
                text_color=TEXT_LIGHT
            )
            hdr.pack(fill="x", padx=2, pady=(2, 0))

            # Tile video canvas
            vlabel = ctk.CTkLabel(
                tile_card, text=f"Camera {i+1}\nDisabled / No Source",
                font=ctk.CTkFont(size=11), text_color=TEXT_DIM
            )
            vlabel.pack(fill="both", expand=True, padx=2, pady=2)

            self.tiles.append({
                "frame": tile_card,
                "header": hdr,
                "label": vlabel
            })

        # ── Single Camera Container (Hidden until switched) ──
        self.single_container = ctk.CTkFrame(panel, fg_color="transparent")
        self.single_header = ctk.CTkLabel(
            self.single_container, text="Focused Camera",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=CARD_BG, height=24, corner_radius=6,
            text_color=TEXT_LIGHT
        )
        self.single_header.pack(fill="x", padx=2, pady=(2, 4))
        self.single_label = ctk.CTkLabel(
            self.single_container, text="No Feed",
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM,
            fg_color=DARK_BG, corner_radius=8
        )
        self.single_label.pack(fill="both", expand=True, padx=2, pady=2)

        # ── LAST KNOWN LOCATION Banner ──
        self.last_seen_banner = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=8, height=42)
        self.last_seen_banner.pack(fill="x", padx=8, pady=(0, 8))
        self.last_seen_banner.pack_propagate(False)

        self.last_seen_label = ctk.CTkLabel(
            self.last_seen_banner,
            text="Last Known Location: Not yet found",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_DIM
        )
        self.last_seen_label.pack(expand=True)

    # ── Right Panel: Movement Trail ─────────────────────────────────────────
    def _build_right_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=12)
        panel.grid(row=0, column=2, padx=(4, 0), pady=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="MOVEMENT TIMELINE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(10, 4), padx=12, anchor="w")

        # Timeline text box
        self.timeline_box = ctk.CTkTextbox(
            panel, fg_color=DARK_BG, corner_radius=8,
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=TEXT_LIGHT,
            wrap="word", state="disabled"
        )
        self.timeline_box.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        # Alert stats strip
        stats = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=8)
        stats.pack(fill="x", padx=8, pady=(0, 6))

        self.sightings_label = ctk.CTkLabel(
            stats, text="Total Sightings: 0",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.sightings_label.pack(pady=4, padx=8, anchor="w")

        self.time_since_label = ctk.CTkLabel(
            stats, text="Last seen: —",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.time_since_label.pack(pady=(0, 4), padx=8, anchor="w")

        # Export & Clear buttons
        ctk.CTkButton(
            panel, text="Export Movement Report (CSV+JSON)",
            command=self._export_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_AMBER,
            corner_radius=8, height=30,
            font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=8, pady=(0, 4))

        ctk.CTkButton(
            panel, text="Clear Timeline",
            command=self._clear_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_RED,
            corner_radius=8, height=28,
            font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=8, pady=(0, 8))

    # ══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_mode_change(self):
        if self.tracking_mode.get() == "class":
            self.photo_frame.pack_forget()
            self.class_entry_frame.pack(fill="x", padx=4, pady=(0, 8))
        else:
            self.class_entry_frame.pack_forget()
            self.photo_frame.pack(fill="x", padx=4, pady=(0, 8))

    def _on_view_mode_change(self, selected_mode):
        if selected_mode == "⊞ 4-Grid":
            self.single_container.pack_forget()
            self.grid_container.pack(fill="both", expand=True, padx=8, pady=(2, 6))
            self.view_mode.set("grid")
        else:
            # Cam 1, Cam 2, Cam 3, Cam 4
            idx = int(selected_mode.replace("Cam ", "")) - 1
            self.grid_container.pack_forget()
            self.single_container.pack(fill="both", expand=True, padx=8, pady=(2, 6))
            self.view_mode.set(f"cam{idx}")

    def _upload_photo(self):
        path = filedialog.askopenfilename(
            title="Select Target Person Photo",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Could not load the selected image file.")
            return

        self.target_photo = img_bgr
        name = self.target_name_var.get().strip() or "Target Person"

        # Update thumbnail
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_img.thumbnail((120, 130), Image.LANCZOS)
        ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
        self.photo_canvas.configure(image=ctk_img, text="")
        self.photo_canvas.image = ctk_img

        # Update scanner & tracker
        self.scanner.set_person_target(img_bgr, person_name=name)
        self.tracker.set_target(name)

        has_face = self.scanner.matcher.target_has_face
        face_status = "Biometric Face + Clothing" if has_face else "Clothing Re-ID"
        self._set_status(f"Target: {name} ({face_status})", color=ACCENT_GREEN)

    def _browse_camera_video(self, slot_idx):
        path = filedialog.askopenfilename(
            title=f"Select Video File for Camera {slot_idx+1}",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if path:
            cam = self.camera_slots[slot_idx]
            cam["source_var"].set(path)
            cam["enabled"].set(True)  # Auto-enable when source is picked
            
            # Suggest a clean location name based on filename if still default
            clean_name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
            curr_loc = cam["location_var"].get()
            if not curr_loc or curr_loc.startswith("Floor") or curr_loc.startswith("Ground"):
                cam["location_var"].set(f"Camera {slot_idx+1} ({clean_name})")

    def _start_scanning(self):
        if self.tracking_mode.get() == "person" and self.target_photo is None:
            messagebox.showwarning("No Target", "Please upload a target photo first!")
            return

        if self.tracking_mode.get() == "class":
            class_name = self.class_name_var.get().strip() or "dog"
            self.scanner.set_class_target(class_name)
            self.tracker.set_target(f"{class_name.capitalize()} Intrusion Detection")

        # Open video capture for all enabled cameras
        active_cams = 0
        for i, cam in enumerate(self.camera_slots):
            if cam["enabled"].get():
                raw_src = cam["source_var"].get().strip()
                if raw_src == "" or raw_src == "0":
                    src = 0
                else:
                    try:
                        src = int(raw_src)
                    except ValueError:
                        src = raw_src

                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    cam["cap"] = cap
                    cam["frame_count"] = 0
                    cam["spotted"] = False
                    active_cams += 1
                else:
                    messagebox.showerror("Error", f"Could not connect to Camera {i+1} source:\n{raw_src}")
                    self._stop_scanning()
                    return
            else:
                cam["cap"] = None

        if active_cams == 0:
            messagebox.showwarning("No Cameras", "Please enable at least one Camera slot!")
            return

        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status(f"MONITORING {active_cams} CAMERA(S)", color=ACCENT_GREEN)

        # Start multi-camera scanning thread
        self.scan_thread = threading.Thread(target=self._multi_scan_loop, daemon=True)
        self.scan_thread.start()

        # Start GUI update loop
        self._update_gui()

    def _stop_scanning(self):
        self.is_running = False
        for cam in self.camera_slots:
            if cam["cap"]:
                cam["cap"].release()
                cam["cap"] = None
            cam["spotted"] = False

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_status("IDLE", color=TEXT_DIM)
        for tile in self.tiles:
            tile["header"].configure(text="Offline", fg_color=CARD_BG)
            tile["frame"].configure(border_color="#22223a")
            tile["label"].configure(image=None, text="Camera Offline")

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-CAMERA SCANNING LOOP (Background Thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _multi_scan_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            any_active = False

            for i, cam in enumerate(self.camera_slots):
                if not self.is_running:
                    break
                cap = cam["cap"]
                if cap is None or not cap.isOpened():
                    continue

                any_active = True
                ret, frame = cap.read()
                
                # If video file reached end, loop seamlessly back to start
                if not ret:
                    src = cam["source_var"].get().strip()
                    if isinstance(src, str) and os.path.isfile(src):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()

                if not ret or frame is None:
                    continue

                cam["frame_count"] += 1
                loc_name = cam["location_var"].get().strip() or f"Camera {i+1}"

                # Real-Time CPU Optimization:
                # Run full YOLO AI detection every 3rd frame per camera (maintaining smooth FPS)
                # On intervening frames, display frame with previous detections
                if cam["frame_count"] % 3 == 0 or cam["last_annotated"] is None:
                    annotated_frame, spotted = self.scanner.process_frame(frame, camera_name=loc_name)
                    cam["last_annotated"] = annotated_frame
                    cam["spotted"] = spotted

                    if spotted and self.scanner.last_sighting["found"]:
                        self.tracker.log_sighting(
                            camera_name=self.scanner.last_sighting["camera_name"],
                            confidence=self.scanner.last_sighting["confidence"],
                            snapshot_path=self.scanner.last_sighting.get("snapshot_path", "")
                        )
                else:
                    # Quick copy for playback fluidity
                    annotated_frame = cam["last_annotated"] if cam["last_annotated"] is not None else frame

                cam["current_frame"] = annotated_frame

            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                self.after(0, self.fps_label.configure, {"text": f"FPS: {fps:.1f}"})
                fps_counter = 0
                fps_timer = time.time()

            # Slight yield to avoid pegging 100% of single CPU core
            time.sleep(0.015)

    # ══════════════════════════════════════════════════════════════════════════
    # GUI UPDATE LOOP (Main Thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_gui(self):
        if not self.is_running:
            return

        current_view = self.view_mode.get()

        # ── Case A: 4-Grid View ──
        if current_view == "grid":
            for i, cam in enumerate(self.camera_slots):
                tile = self.tiles[i]
                if cam["enabled"].get() and cam["current_frame"] is not None:
                    frame = cam["current_frame"]
                    loc = cam["location_var"].get().strip() or f"Camera {i+1}"
                    
                    if cam["spotted"]:
                        tile["header"].configure(text=f"🚨 CAM {i+1}: {loc} [TARGET FOUND!]", fg_color="#801010", text_color="#ffffff")
                        tile["frame"].configure(border_color=ACCENT_GREEN, border_width=2)
                    else:
                        tile["header"].configure(text=f"● CAM {i+1}: {loc}", fg_color=CARD_BG, text_color=TEXT_LIGHT)
                        tile["frame"].configure(border_color="#22223a", border_width=1)

                    # Resize frame to tile size
                    tw = max(10, tile["label"].winfo_width() or 320)
                    th = max(10, tile["label"].winfo_height() or 220)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pimg = Image.fromarray(frame_rgb)
                    pimg.thumbnail((tw, th), Image.BILINEAR)
                    cimg = ctk.CTkImage(pimg, size=pimg.size)
                    tile["label"].configure(image=cimg, text="")
                    tile["label"].image = cimg
                elif not cam["enabled"].get():
                    tile["header"].configure(text=f"CAM {i+1} — Disabled", fg_color="#181828", text_color=TEXT_DIM)
                    tile["frame"].configure(border_color="#181828", border_width=1)
                    tile["label"].configure(image=None, text=f"Camera {i+1}\n(Disabled in Settings)")

        # ── Case B: Single Camera Zoom View ──
        elif current_view.startswith("cam"):
            idx = int(current_view.replace("cam", ""))
            cam = self.camera_slots[idx]
            loc = cam["location_var"].get().strip() or f"Camera {idx+1}"

            if cam["spotted"]:
                self.single_header.configure(text=f"🚨 FOCUSED: CAM {idx+1} — {loc} [TARGET SPOTTED!]", fg_color="#801010")
            else:
                self.single_header.configure(text=f"FOCUSED: CAM {idx+1} — {loc}", fg_color=CARD_BG)

            if cam["current_frame"] is not None:
                frame = cam["current_frame"]
                sw = max(10, self.single_label.winfo_width() or 640)
                sh = max(10, self.single_label.winfo_height() or 460)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pimg = Image.fromarray(frame_rgb)
                pimg.thumbnail((sw, sh), Image.BILINEAR)
                cimg = ctk.CTkImage(pimg, size=pimg.size)
                self.single_label.configure(image=cimg, text="")
                self.single_label.image = cimg

        # Update Last Seen banner & Timeline
        self._update_last_seen_banner()
        self._update_timeline_display()

        # Schedule next update (~33 FPS GUI refresh)
        self.after(30, self._update_gui)

    def _update_last_seen_banner(self):
        if self.tracker.last_known["camera"]:
            loc_text = (
                f"📍 Last Known Location: {self.tracker.last_known['camera']}  "
                f"  ·  {self.tracker.last_known['timestamp']}  "
                f"  ·  {int(self.tracker.last_known['confidence']*100)}% confidence"
            )
            is_lost = self.tracker.is_target_lost() if self.is_running else False
            color = ACCENT_RED if is_lost else ACCENT_GREEN
            self.last_seen_banner.configure(fg_color="#0a2e1a" if not is_lost else "#2e0a0a")
            self.last_seen_label.configure(text=loc_text, text_color=color)
        else:
            self.last_seen_label.configure(
                text="Last Known Location: Searching all active camera feeds...",
                text_color=TEXT_DIM
            )
            self.last_seen_banner.configure(fg_color=CARD_BG)

    def _update_timeline_display(self):
        summary = self.tracker.get_timeline_summary()
        self.sightings_label.configure(text=f"Total Sightings: {summary['total_sightings']}")

        if summary["time_since_last_seen"]:
            self.time_since_label.configure(text=f"Last seen: {summary['time_since_last_seen']} ago")
        else:
            self.time_since_label.configure(text="Last seen: —")

        trail_str = self.tracker.get_movement_trail()

        self.timeline_box.configure(state="normal")
        self.timeline_box.delete("1.0", "end")
        self.timeline_box.insert("1.0", trail_str)
        self.timeline_box.see("end")
        self.timeline_box.configure(state="disabled")

    def _export_timeline(self):
        if not self.tracker.timeline:
            messagebox.showinfo("Export", "No sightings recorded yet to export.")
            return

        csv_path, json_path = self.tracker.export_to_csv_and_json()
        messagebox.showinfo(
            "Export Successful",
            f"Movement timeline exported successfully!\n\nCSV:  {os.path.basename(csv_path)}\nJSON: {os.path.basename(json_path)}\n\nSaved in reports/ folder."
        )

    def _clear_timeline(self):
        name = self.target_name_var.get().strip() or "Target Person"
        self.tracker.set_target(name)
        self._update_last_seen_banner()
        self._update_timeline_display()

    def _set_status(self, text, color=TEXT_DIM):
        self.status_dot.configure(text=f"●  {text}", text_color=color)

    def _on_close(self):
        self.is_running = False
        for cam in self.camera_slots:
            if cam["cap"]:
                cam["cap"].release()
        self.destroy()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartSuperviseApp()
    app.mainloop()
