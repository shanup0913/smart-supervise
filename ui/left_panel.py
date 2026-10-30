"""
LeftPanel: Target setup, 360-degree multi-angle gallery, camera slots,
per-camera speed selectors, and scan controls.
"""

import os
import cv2
import threading
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

from ui.theme import (
    DARK_BG, PANEL_BG, CARD_BG, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_LIGHT, TEXT_DIM
)


class LeftPanel(ctk.CTkScrollableFrame):
    """Left sidebar containing target biometrics, camera slots, and control buttons."""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=PANEL_BG, corner_radius=12, width=285)
        self.app = app

        # Reactive State
        self.tracking_mode = ctk.StringVar(value="person")
        self.target_name_var = ctk.StringVar(value="Target Person")
        self.class_name_var = ctk.StringVar(value="dog")
        self.global_speed_var = ctk.StringVar(value="10x")
        self.match_sensitivity_var = ctk.StringVar(value="Strict (70%)")

        # Camera Slot UI bindings
        default_locations = [
            "Camera 1 (Main Gate)",
            "Camera 2 (Corridor)",
            "Camera 3 (Room 204)",
            "Camera 4 (Cafeteria)"
        ]
        self.cam_ui_slots = []
        for i in range(4):
            self.cam_ui_slots.append({
                "id": i,
                "enabled": ctk.BooleanVar(value=(i == 0)),
                "source_var": ctk.StringVar(value="0" if i == 0 else ""),
                "location_var": ctk.StringVar(value=default_locations[i]),
                "speed_var": ctk.StringVar(value="10x")
            })

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self, text="TARGET & CAMERAS SETUP",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(6, 4), padx=10, anchor="w")

        # ── 1. Target Mode ──
        mode_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8)
        mode_frame.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(mode_frame, text="Tracking Mode", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_DIM).pack(pady=(4, 2), padx=8, anchor="w")

        ctk.CTkRadioButton(
            mode_frame, text="Search Person (Biometrics)",
            variable=self.tracking_mode, value="person",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=11), fg_color=ACCENT_BLUE
        ).pack(anchor="w", padx=8, pady=2)

        ctk.CTkRadioButton(
            mode_frame, text="Animal / Intruder Alert",
            variable=self.tracking_mode, value="class",
            command=self._on_mode_change,
            font=ctk.CTkFont(size=11), fg_color=ACCENT_RED
        ).pack(anchor="w", padx=8, pady=(2, 6))

        # Dynamic Target Container
        self.target_container = ctk.CTkFrame(self, fg_color="transparent")
        self.target_container.pack(fill="x", padx=0, pady=0)

        # Class Intrusion entry
        self.class_entry_frame = ctk.CTkFrame(self.target_container, fg_color=CARD_BG, corner_radius=8)
        ctk.CTkLabel(self.class_entry_frame, text="Alert Class (e.g. dog, cat)",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(pady=(6, 2), padx=8, anchor="w")
        ctk.CTkEntry(
            self.class_entry_frame, textvariable=self.class_name_var,
            width=210, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(0, 6), padx=8)

        # Photo Gallery container
        self.photo_frame = ctk.CTkFrame(self.target_container, fg_color=CARD_BG, corner_radius=8)
        self.photo_frame.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(
            self.photo_frame, text="🎯 TARGET GALLERY (360° Poses)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(4, 1), padx=8, anchor="w")

        ctk.CTkLabel(
            self.photo_frame, text="Upload Front, Side & Back photos to detect\nsuspect when turning or walking away.",
            font=ctk.CTkFont(size=9), text_color=TEXT_DIM, justify="left"
        ).pack(pady=(0, 4), padx=8, anchor="w")

        self.gallery_container = ctk.CTkFrame(self.photo_frame, fg_color=DARK_BG, corner_radius=6)
        self.gallery_container.pack(fill="x", padx=8, pady=(0, 6))

        self.gallery_placeholder = ctk.CTkLabel(
            self.gallery_container,
            text="No Photos Loaded\nUpload Front, Side or Back views",
            height=75, font=ctk.CTkFont(size=10), text_color=TEXT_DIM
        )
        self.gallery_placeholder.pack(pady=8, padx=8)

        btn_row = ctk.CTkFrame(self.photo_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 4))

        ctk.CTkButton(
            btn_row, text="📁 Upload Photo(s)",
            command=self._upload_photo,
            fg_color=ACCENT_BLUE, hover_color="#2d4bcc",
            corner_radius=6, height=26,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ctk.CTkButton(
            btn_row, text="➕ Add Angle",
            command=self._add_angle_photo,
            fg_color="#30304a", hover_color="#404060",
            corner_radius=6, height=26,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.clear_photos_btn = ctk.CTkButton(
            self.photo_frame, text="✕ Clear Target Gallery",
            command=self._clear_target_gallery,
            fg_color="transparent", hover_color="#301515", text_color="#d05050",
            corner_radius=4, height=20,
            font=ctk.CTkFont(size=9)
        )
        self.clear_photos_btn.pack(pady=(0, 4), padx=8)

        ctk.CTkLabel(self.photo_frame, text="Target Name",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM).pack(padx=8, anchor="w")
        ctk.CTkEntry(
            self.photo_frame, textvariable=self.target_name_var,
            width=210, height=26, fg_color=DARK_BG, border_color=ACCENT_BLUE
        ).pack(pady=(1, 6), padx=8)

        # ── 2. Match Sensitivity (Precision vs Recall) ──
        sens_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8)
        sens_frame.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(
            sens_frame, text="🎯 TARGET MATCH SENSITIVITY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(4, 2), padx=8, anchor="w")

        self.sens_selector = ctk.CTkSegmentedButton(
            sens_frame,
            values=["Strict (70%)", "Balanced (60%)", "Flexible (50%)"],
            variable=self.match_sensitivity_var,
            command=self._on_sensitivity_change,
            selected_color=ACCENT_BLUE,
            height=24,
            font=ctk.CTkFont(size=10)
        )
        self.sens_selector.pack(fill="x", padx=8, pady=(2, 4))

        self.sens_hint_label = ctk.CTkLabel(
            sens_frame, text="Strict: Zero false alarms (Ignores hands & random clothes)",
            font=ctk.CTkFont(size=9),
            text_color="#9999bb"
        )
        self.sens_hint_label.pack(padx=8, pady=(0, 4), anchor="w")

        # ── 3. Fast Search Speed ──
        speed_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8)
        speed_frame.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(
            speed_frame, text="⚡ FAST SEARCH SPEED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(4, 2), padx=8, anchor="w")

        self.speed_selector = ctk.CTkSegmentedButton(
            speed_frame,
            values=["1x", "5x", "10x", "20x", "40x"],
            variable=self.global_speed_var,
            command=self._on_global_speed_change,
            selected_color=ACCENT_BLUE,
            height=24,
            font=ctk.CTkFont(size=10)
        )
        self.speed_selector.pack(fill="x", padx=8, pady=(2, 4))

        ctk.CTkLabel(
            speed_frame, text="10x = ~4 min / 40m  |  20x = ~2 min  |  40x = ~1 min",
            font=ctk.CTkFont(size=9),
            text_color="#9999bb"
        ).pack(padx=8, pady=(0, 4), anchor="w")

        # ── 3. Multi-Camera Slots ──
        ctk.CTkLabel(
            self, text="CCTV CAMERAS (UP TO 4)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(4, 2), padx=10, anchor="w")

        for i, cam in enumerate(self.cam_ui_slots):
            cframe = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8)
            cframe.pack(fill="x", padx=4, pady=(0, 5))

            hdr = ctk.CTkFrame(cframe, fg_color="transparent")
            hdr.pack(fill="x", padx=8, pady=(3, 2))

            chk = ctk.CTkCheckBox(
                hdr, text=f"Camera {i+1}",
                variable=cam["enabled"],
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=ACCENT_GREEN,
                text_color=TEXT_LIGHT
            )
            chk.pack(side="left")

            cam_badge = ctk.CTkLabel(
                hdr, text="Ready" if cam["enabled"].get() else "OFF",
                font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                text_color=ACCENT_GREEN if cam["enabled"].get() else TEXT_DIM
            )
            cam_badge.pack(side="right")
            cam["progress_badge"] = cam_badge

            src_row = ctk.CTkFrame(cframe, fg_color="transparent")
            src_row.pack(fill="x", padx=8, pady=(1, 2))

            ctk.CTkEntry(
                src_row, textvariable=cam["source_var"],
                placeholder_text="0 | rtsp | video.mp4",
                width=140, height=24, fg_color=DARK_BG, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=(0, 4))

            ctk.CTkButton(
                src_row, text="Browse",
                command=lambda idx=i: self._browse_camera_video(idx),
                width=65, height=24, fg_color=DARK_BG, hover_color=PANEL_BG,
                border_width=1, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(side="left")

            ctk.CTkEntry(
                cframe, textvariable=cam["location_var"],
                placeholder_text="Room / Floor Location",
                width=210, height=24, fg_color=DARK_BG, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(padx=8, pady=(1, 3))

            spd_row = ctk.CTkFrame(cframe, fg_color="transparent")
            spd_row.pack(fill="x", padx=8, pady=(0, 3))
            ctk.CTkLabel(spd_row, text="Speed:", font=ctk.CTkFont(size=9, weight="bold"), text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
            ctk.CTkSegmentedButton(
                spd_row,
                values=["1x", "5x", "10x", "20x", "40x"],
                variable=cam["speed_var"],
                selected_color=ACCENT_BLUE,
                height=20,
                font=ctk.CTkFont(size=9)
            ).pack(side="left", fill="x", expand=True)

            cam_pbar = ctk.CTkProgressBar(
                cframe, height=4, corner_radius=2,
                progress_color=ACCENT_GREEN, fg_color="#181828"
            )
            cam_pbar.set(0.0)
            cam_pbar.pack(fill="x", padx=8, pady=(0, 4))
            cam["progress_bar"] = cam_pbar

        # ── Start / Stop Buttons ──
        self.start_btn = ctk.CTkButton(
            self, text="▶  START SCANNING",
            command=self._on_start_clicked,
            fg_color=ACCENT_GREEN, hover_color="#04a57a",
            text_color="#000000",
            corner_radius=8, height=36,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.start_btn.pack(fill="x", padx=4, pady=(6, 4))

        self.stop_btn = ctk.CTkButton(
            self, text="■  STOP",
            command=self._on_stop_clicked,
            fg_color="#333355", hover_color=ACCENT_RED,
            corner_radius=8, height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled"
        )
        self.stop_btn.pack(fill="x", padx=4, pady=(0, 8))

    def _on_mode_change(self):
        if self.tracking_mode.get() == "class":
            self.photo_frame.pack_forget()
            self.class_entry_frame.pack(fill="x", padx=4, pady=(0, 6))
        else:
            self.class_entry_frame.pack_forget()
            self.photo_frame.pack(fill="x", padx=4, pady=(0, 6))

    def _on_global_speed_change(self, value):
        for cam in self.cam_ui_slots:
            cam["speed_var"].set(value)

    def _on_sensitivity_change(self, value):
        if "Strict" in value or "70" in value:
            self.app.scanner.matcher.match_threshold = 0.70
            self.sens_hint_label.configure(text="Strict: Zero false alarms (Ignores hands & random clothes)")
            self.app.set_status("Target Sensitivity: STRICT (70% - Zero false alarms)", color=ACCENT_GREEN)
        elif "Balanced" in value or "60" in value:
            self.app.scanner.matcher.match_threshold = 0.60
            self.sens_hint_label.configure(text="Balanced: Standard CCTV tracking (Face + Clothes)")
            self.app.set_status("Target Sensitivity: BALANCED (60% threshold)", color=ACCENT_BLUE)
        else:
            self.app.scanner.matcher.match_threshold = 0.50
            self.sens_hint_label.configure(text="Flexible: High recall for distant or blurry cameras")
            self.app.set_status("Target Sensitivity: FLEXIBLE (50% threshold)", color=ACCENT_AMBER)

    def _browse_camera_video(self, slot_idx):
        path = filedialog.askopenfilename(
            title=f"Select Video File for Camera {slot_idx+1}",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if path:
            cam = self.cam_ui_slots[slot_idx]
            cam["source_var"].set(path)
            cam["enabled"].set(True)
            clean_name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
            curr_loc = cam["location_var"].get()
            if not curr_loc or curr_loc.startswith("Camera"):
                cam["location_var"].set(f"Camera {slot_idx+1} ({clean_name})")

    def _upload_photo(self):
        paths = filedialog.askopenfilenames(
            title="Select Target Person Photo(s) - Front, Side, Back (Hold Ctrl to select multiple)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not paths:
            return
        
        name = self.target_name_var.get().strip() or "Target Person"
        self.app.scanner.matcher.clear_target()
        self.app.scanner.matcher.target_name = name
        self.app.tracker.set_target(name)
        self.app.set_status("Analyzing biometric facial landmarks & clothing colors...", color=ACCENT_AMBER)
        self.update_idletasks()

        def _bg_upload():
            for path in paths:
                img_bgr = cv2.imread(path)
                if img_bgr is not None:
                    self.app.scanner.add_person_photo(img_bgr)

            def _finish():
                self.app.scanner.last_sighting["found"] = False
                self.app.scanner.last_sighting["target_name"] = name
                self.app.scanner.sightings_history.clear()
                self._refresh_gallery_display()
            self.app.after(0, _finish)

        threading.Thread(target=_bg_upload, daemon=True).start()

    def _add_angle_photo(self):
        path = filedialog.askopenfilename(
            title="Select Additional Angle / Pose Photo (Side, Back, etc.)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            self.app.set_status("Analyzing pose angle...", color=ACCENT_AMBER)
            self.update_idletasks()

            def _bg_add():
                self.app.scanner.add_person_photo(img_bgr)
                self.app.after(0, self._refresh_gallery_display)

            threading.Thread(target=_bg_add, daemon=True).start()

    def _clear_target_gallery(self):
        self.app.scanner.matcher.clear_target()
        self._refresh_gallery_display()
        self.app.set_status("Target gallery cleared.", color=TEXT_DIM)

    def _refresh_gallery_display(self):
        for widget in self.gallery_container.winfo_children():
            widget.destroy()

        profiles = self.app.scanner.matcher.target_profiles
        if not profiles:
            self.gallery_placeholder = ctk.CTkLabel(
                self.gallery_container,
                text="No Photos Loaded\nUpload Front, Side or Back views",
                height=75, font=ctk.CTkFont(size=10), text_color=TEXT_DIM
            )
            self.gallery_placeholder.pack(pady=8, padx=8)
            return

        strip = ctk.CTkFrame(self.gallery_container, fg_color="transparent")
        strip.pack(fill="x", padx=4, pady=4)

        for i, p in enumerate(profiles[:4]):
            card = ctk.CTkFrame(strip, fg_color="#181828", corner_radius=6)
            card.pack(side="left", padx=2, pady=2, expand=True, fill="both")

            rgb = cv2.cvtColor(p.thumbnail, cv2.COLOR_BGR2RGB)
            pimg = Image.fromarray(rgb)
            pimg.thumbnail((50, 65), Image.BILINEAR)
            cimg = ctk.CTkImage(pimg, size=pimg.size)

            lbl = ctk.CTkLabel(card, image=cimg, text="")
            lbl.image = cimg
            lbl.pack(padx=2, pady=(2, 0))

            badge_text = p.angle_label.split()[0]
            badge_color = ACCENT_GREEN if p.has_face else ACCENT_AMBER
            ctk.CTkLabel(
                card, text=badge_text, font=ctk.CTkFont(size=8, weight="bold"),
                text_color=badge_color
            ).pack(pady=(0, 2))

        count = len(profiles)
        faces = sum(1 for p in profiles if p.has_face)
        name = self.target_name_var.get().strip() or "Target Person"
        self.app.set_status(f"Target: {name} ({count} poses/angles loaded, {faces} with biometric face)", color=ACCENT_GREEN)

    def _on_start_clicked(self):
        # Sync sensitivity threshold
        sens_val = self.match_sensitivity_var.get()
        if "Strict" in sens_val or "70" in sens_val:
            self.app.scanner.matcher.match_threshold = 0.70
        elif "Balanced" in sens_val or "60" in sens_val:
            self.app.scanner.matcher.match_threshold = 0.60
        else:
            self.app.scanner.matcher.match_threshold = 0.50

        # Sync slot configurations to video engine
        for i, cam in enumerate(self.cam_ui_slots):
            self.app.video_engine.configure_slot(
                i,
                enabled=cam["enabled"].get(),
                source=cam["source_var"].get(),
                location=cam["location_var"].get(),
                speed=cam["speed_var"].get()
            )

        if self.tracking_mode.get() == "person":
            if len(self.app.scanner.matcher.target_profiles) == 0:
                messagebox.showwarning("No Target", "Please upload at least one target photo first!")
                return
            name = self.target_name_var.get().strip() or "Target Person"
            self.app.scanner.matcher.target_name = name
            self.app.tracker.set_target(name)
        else:
            class_name = self.class_name_var.get().strip() or "dog"
            self.app.scanner.set_class_target(class_name)
            self.app.tracker.set_target(f"{class_name.capitalize()} Intrusion Detection")

        # Show immediate feedback so GUI never freezes or shows "Not Responding"
        self.start_btn.configure(text="⏳  OPENING FEEDS...", state="disabled")
        self.app.set_status("INITIALIZING CAMERA STREAMS...", color=ACCENT_AMBER)
        self.update_idletasks()

        def _bg_start():
            try:
                active_count = self.app.video_engine.start(
                    global_speed=self.global_speed_var.get(),
                    on_finished_callback=self.app.on_scan_finished
                )
                def _success():
                    self.start_btn.configure(text="▶  START SCANNING", state="disabled")
                    self.stop_btn.configure(state="normal")
                    speed = self.global_speed_var.get()
                    self.app.set_status(f"SYNCHRONIZED SEARCH: {active_count} CAMERAS @ {speed}", color=ACCENT_GREEN)
                self.app.after(0, _success)
            except Exception as e:
                err_msg = str(e)
                def _err():
                    self.start_btn.configure(text="▶  START SCANNING", state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.app.set_status("STREAM CONNECTION ERROR", color=ACCENT_RED)
                    messagebox.showerror("Start Error", err_msg)
                self.app.after(0, _err)

        threading.Thread(target=_bg_start, daemon=True).start()

    def _on_stop_clicked(self):
        self.stop_btn.configure(text="⏳ STOPPING...", state="disabled")
        self.update_idletasks()
        self.app.video_engine.stop()
        self.start_btn.configure(text="▶  START SCANNING", state="normal")
        self.stop_btn.configure(text="■  STOP", state="disabled")
        self.app.set_status("IDLE", color=TEXT_DIM)
        self.app.center_grid.reset_tiles()
        self.reset_camera_progress()

    def update_camera_progress(self, video_engine):
        """Updates individual per-camera progress badges and progress bars."""
        for i, cam in enumerate(self.cam_ui_slots):
            state = video_engine.get_slot_state(i)
            if not state or not state["enabled"]:
                cam["progress_badge"].configure(text="OFF", text_color=TEXT_DIM)
                cam["progress_bar"].set(0.0)
            elif state.get("ended", False):
                cam["progress_badge"].configure(text="🏁 100%", text_color="#04a57a")
                cam["progress_bar"].set(1.0)
                cam["progress_bar"].configure(progress_color="#04a57a")
            elif state.get("is_file", False):
                pct = state.get("progress_pct", 0.0)
                cam["progress_badge"].configure(text=f"{pct:.1f}%", text_color=ACCENT_GREEN)
                cam["progress_bar"].set(pct / 100.0)
                cam["progress_bar"].configure(progress_color=ACCENT_GREEN)
            else:
                cam["progress_badge"].configure(text="LIVE", text_color=ACCENT_BLUE)
                cam["progress_bar"].set(1.0)
                cam["progress_bar"].configure(progress_color=ACCENT_BLUE)

    def reset_camera_progress(self):
        """Resets all camera progress badges and bars."""
        for cam in self.cam_ui_slots:
            if cam["enabled"].get():
                cam["progress_badge"].configure(text="Ready", text_color=ACCENT_GREEN)
            else:
                cam["progress_badge"].configure(text="OFF", text_color=TEXT_DIM)
            cam["progress_bar"].set(0.0)
