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
        self.title("Smart Supervise — AI CCTV Intelligence System")
        self.geometry("1280x780")
        self.minsize(1100, 700)
        self.configure(fg_color=DARK_BG)

        # ── Core Components ─────────────────────────────────────────────────
        self.scanner  = CCTVScanner()
        self.tracker  = LocationTracker(lost_timeout_seconds=30)

        # ── State Variables ──────────────────────────────────────────────────
        self.cap            = None
        self.is_running     = False
        self.scan_thread    = None
        self.current_frame  = None
        self.target_photo   = None  # numpy BGR crop
        self.tracking_mode  = ctk.StringVar(value="person")  # "person" or "class"
        self.camera_source  = None  # 0, rtsp url, or file path

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
            bar, text="AI-Powered CCTV Target Search & Tracking",
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
        """3-column layout: Left (Target) | Center (Video) | Right (Timeline)"""
        main = ctk.CTkFrame(self, fg_color=DARK_BG)
        main.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        main.grid_columnconfigure(0, weight=0, minsize=240)  # Left panel
        main.grid_columnconfigure(1, weight=1)               # Center video
        main.grid_columnconfigure(2, weight=0, minsize=280)  # Right panel
        main.grid_rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_center_panel(main)
        self._build_right_panel(main)

    # ── Left Panel ─────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        panel.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="TARGET SETUP",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(14, 6), padx=14, anchor="w")

        # ── Mode Selector ──
        mode_frame = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        mode_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(mode_frame, text="Tracking Mode", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).pack(pady=(8, 2), padx=10, anchor="w")

        ctk.CTkRadioButton(
            mode_frame, text="Search Person by Photo",
            variable=self.tracking_mode, value="person",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=12), fg_color=ACCENT_BLUE
        ).pack(anchor="w", padx=14, pady=3)

        ctk.CTkRadioButton(
            mode_frame, text="Animal / Intruder Alert",
            variable=self.tracking_mode, value="class",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=12), fg_color=ACCENT_RED
        ).pack(anchor="w", padx=14, pady=(3, 10))

        # ── Class Name (shows only in class mode) ──
        self.class_name_var = ctk.StringVar(value="dog")
        self.class_entry_frame = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        ctk.CTkLabel(self.class_entry_frame, text="Alert Class (e.g. dog, cat)",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(pady=(8, 2), padx=10, anchor="w")
        ctk.CTkEntry(
            self.class_entry_frame, textvariable=self.class_name_var,
            width=180, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(0, 10), padx=10)

        # ── Photo Upload Section ──
        self.photo_frame = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        self.photo_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(self.photo_frame, text="Target Photo", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).pack(pady=(8, 4), padx=10, anchor="w")

        # Thumbnail placeholder
        self.photo_canvas = ctk.CTkLabel(
            self.photo_frame, text="No Photo\nUploaded",
            width=140, height=170,
            fg_color=DARK_BG, corner_radius=8,
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.photo_canvas.pack(pady=(0, 6), padx=10)

        ctk.CTkButton(
            self.photo_frame, text="Upload Target Photo",
            command=self._upload_photo,
            fg_color=ACCENT_BLUE, hover_color="#2d4bcc",
            corner_radius=8, height=34,
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(fill="x", padx=10, pady=(0, 6))

        # Target name input
        self.target_name_var = ctk.StringVar(value="Target Person")
        ctk.CTkLabel(self.photo_frame, text="Target Name",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(padx=10, anchor="w")
        ctk.CTkEntry(
            self.photo_frame, textvariable=self.target_name_var,
            width=180, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(2, 10), padx=10)

        # ── Camera Source ──
        src_frame = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        src_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(src_frame, text="Video Source", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).pack(pady=(8, 4), padx=10, anchor="w")

        self.source_var = ctk.StringVar(value="0")
        ctk.CTkEntry(
            src_frame, textvariable=self.source_var,
            width=180, fg_color=DARK_BG, border_color=ACCENT_BLUE,
            placeholder_text="0 | rtsp://... | video.mp4"
        ).pack(pady=(0, 4), padx=10)

        ctk.CTkButton(
            src_frame, text="Browse Video File",
            command=self._browse_video,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_BLUE,
            corner_radius=8, height=30, font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=10, pady=(0, 10))

        # ── Start / Stop Buttons ──
        self.start_btn = ctk.CTkButton(
            panel, text="▶  START SCANNING",
            command=self._start_scanning,
            fg_color=ACCENT_GREEN, hover_color="#04a57a",
            text_color="#000000",
            corner_radius=10, height=42,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.start_btn.pack(fill="x", padx=10, pady=(4, 4))

        self.stop_btn = ctk.CTkButton(
            panel, text="■  STOP",
            command=self._stop_scanning,
            fg_color="#333355", hover_color=ACCENT_RED,
            corner_radius=10, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.stop_btn.pack(fill="x", padx=10, pady=(0, 10))

    # ── Center Panel ────────────────────────────────────────────────────────
    def _build_center_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        panel.grid(row=0, column=1, padx=6, pady=0, sticky="nsew")

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            header, text="LIVE CCTV FEED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_GREEN
        ).pack(side="left")

        self.cam_label = ctk.CTkLabel(
            header, text="— No Source Connected",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.cam_label.pack(side="left", padx=8)

        # FPS counter
        self.fps_label = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.fps_label.pack(side="right")

        # Video display canvas
        self.video_label = ctk.CTkLabel(
            panel, text="No Feed\n\nStart scanning to connect\nto your camera or video file",
            fg_color=DARK_BG, corner_radius=10,
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # ── LAST KNOWN LOCATION Banner ──
        self.last_seen_banner = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10, height=46)
        self.last_seen_banner.pack(fill="x", padx=10, pady=(0, 10))
        self.last_seen_banner.pack_propagate(False)

        self.last_seen_label = ctk.CTkLabel(
            self.last_seen_banner,
            text="Last Known Location: Not yet found",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_DIM
        )
        self.last_seen_label.pack(expand=True)

    # ── Right Panel ─────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        panel.grid(row=0, column=2, padx=(6, 0), pady=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="MOVEMENT TRAIL",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(14, 6), padx=14, anchor="w")

        # Timeline text box
        self.timeline_box = ctk.CTkTextbox(
            panel, fg_color=DARK_BG, corner_radius=10,
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=TEXT_LIGHT,
            wrap="word", state="disabled"
        )
        self.timeline_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # Alert stats strip
        stats = ctk.CTkFrame(panel, fg_color=CARD_BG, corner_radius=10)
        stats.pack(fill="x", padx=10, pady=(0, 6))

        self.sightings_label = ctk.CTkLabel(
            stats, text="Total Sightings: 0",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.sightings_label.pack(pady=6, padx=10, anchor="w")

        self.time_since_label = ctk.CTkLabel(
            stats, text="Last seen: —",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.time_since_label.pack(pady=(0, 6), padx=10, anchor="w")

        # Export & Clear buttons
        ctk.CTkButton(
            panel, text="Export Timeline (CSV + JSON)",
            command=self._export_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_AMBER,
            corner_radius=8, height=34,
            font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkButton(
            panel, text="Clear Timeline",
            command=self._clear_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_RED,
            corner_radius=8, height=30,
            font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=10, pady=(0, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_mode_change(self):
        if self.tracking_mode.get() == "class":
            self.photo_frame.pack_forget()
            self.class_entry_frame.pack(fill="x", padx=10, pady=(0, 10), after=None)
            self.class_entry_frame.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.class_entry_frame.pack_forget()
            self.photo_frame.pack(fill="x", padx=10, pady=(0, 10))

    def _upload_photo(self):
        path = filedialog.askopenfilename(
            title="Select Target Photo",
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
        pil_img.thumbnail((140, 170), Image.LANCZOS)
        ctk_img = ctk.CTkImage(pil_img, size=(140, 170))
        self.photo_canvas.configure(image=ctk_img, text="")
        self.photo_canvas.image = ctk_img

        # Update scanner & tracker
        self.scanner.set_person_target(img_bgr, person_name=name)
        self.tracker.set_target(name)

        self._set_status(f"Target set: {name}", color=ACCENT_GREEN)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if path:
            self.source_var.set(path)

    def _start_scanning(self):
        # Validate
        if self.tracking_mode.get() == "person" and self.target_photo is None:
            messagebox.showwarning("No Target", "Please upload a target photo first!")
            return

        if self.tracking_mode.get() == "class":
            class_name = self.class_name_var.get().strip() or "dog"
            self.scanner.set_class_target(class_name)
            self.tracker.set_target(f"{class_name.capitalize()} Intrusion Detection")

        # Determine video source
        src_raw = self.source_var.get().strip()
        if src_raw == "0" or src_raw == "":
            self.camera_source = 0
            src_label = "Webcam (Live)"
        else:
            try:
                self.camera_source = int(src_raw)
                src_label = f"Camera {src_raw}"
            except ValueError:
                self.camera_source = src_raw
                src_label = os.path.basename(src_raw) if os.path.isfile(src_raw) else src_raw

        # Open capture
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            messagebox.showerror("Connection Failed", f"Could not open video source:\n{self.camera_source}")
            return

        self.cam_label.configure(text=f"— {src_label}")
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status("SCANNING", color=ACCENT_GREEN)

        # Start scanning in background thread
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()

        # Start GUI update loop
        self._update_gui()

    def _stop_scanning(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.cam_label.configure(text="— Stopped")
        self._set_status("IDLE", color=TEXT_DIM)
        self.video_label.configure(image=None, text="Feed stopped.\nPress START to reconnect.")

    # ══════════════════════════════════════════════════════════════════════════
    # SCANNING LOOP (runs in background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _scan_loop(self):
        fps_counter = 0
        fps_timer = time.time()
        cam_name = self.source_var.get().strip() or "Live Camera"

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Run YOLO + Matcher
            annotated_frame, spotted = self.scanner.process_frame(frame, camera_name=cam_name)

            # Log sighting to tracker
            if spotted and self.scanner.last_sighting["found"]:
                self.tracker.log_sighting(
                    camera_name=self.scanner.last_sighting["camera_name"],
                    confidence=self.scanner.last_sighting["confidence"],
                    snapshot_path=self.scanner.last_sighting.get("snapshot_path", "")
                )

            # Store frame for GUI thread
            self.current_frame = annotated_frame

            # FPS calc
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                self.after(0, self.fps_label.configure, {"text": f"FPS: {fps:.1f}"})
                fps_counter = 0
                fps_timer = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # GUI UPDATE LOOP (runs on main thread via after())
    # ══════════════════════════════════════════════════════════════════════════

    def _update_gui(self):
        if not self.is_running:
            return

        # Display video frame
        if self.current_frame is not None:
            frame = self.current_frame
            # Resize frame to fit the label widget
            widget_w = self.video_label.winfo_width() or 640
            widget_h = self.video_label.winfo_height() or 480
            if widget_w > 10 and widget_h > 10:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img.thumbnail((widget_w, widget_h), Image.BILINEAR)
                ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

        # Update Last Seen banner
        self._update_last_seen_banner()

        # Update timeline box
        self._update_timeline_display()

        # Schedule next update (30ms ≈ 33 FPS GUI refresh)
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
                text="Last Known Location: Searching for target...",
                text_color=TEXT_DIM
            )
            self.last_seen_banner.configure(fg_color=CARD_BG)

    def _update_timeline_display(self):
        trail = self.tracker.get_movement_trail()
        count = len(self.tracker.timeline)
        self.sightings_label.configure(text=f"Total Sightings: {count}")

        elapsed = self.tracker.seconds_since_last_seen()
        if elapsed is not None:
            self.time_since_label.configure(text=f"Last seen: {elapsed:.0f}s ago")
        else:
            self.time_since_label.configure(text="Last seen: —")

        self.timeline_box.configure(state="normal")
        self.timeline_box.delete("1.0", "end")
        self.timeline_box.insert("end", trail)
        self.timeline_box.configure(state="disabled")
        self.timeline_box.see("end")

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITY
    # ══════════════════════════════════════════════════════════════════════════

    def _export_timeline(self):
        csv_path = self.tracker.export_to_csv()
        json_path = self.tracker.export_to_json()
        if csv_path:
            messagebox.showinfo(
                "Export Successful",
                f"Timeline exported!\n\nCSV:\n{csv_path}\n\nJSON:\n{json_path}"
            )
        else:
            messagebox.showinfo("Nothing to Export", "No sightings recorded yet.")

    def _clear_timeline(self):
        self.tracker.clear()
        self._update_timeline_display()
        self._update_last_seen_banner()

    def _set_status(self, text, color=TEXT_DIM):
        self.status_dot.configure(text=f"●  {text}", text_color=color)

    def _on_close(self):
        self._stop_scanning()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartSuperviseApp()
    app.mainloop()
