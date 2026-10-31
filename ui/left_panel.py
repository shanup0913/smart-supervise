"""
LeftPanel: Target setup, 360-degree multi-angle gallery, camera slots,
per-camera speed selectors, and scan controls.
"""

import os
import sys
import re
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
        super().__init__(parent, fg_color=PANEL_BG, corner_radius=12, width=315)
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

        btn_row2 = ctk.CTkFrame(self.photo_frame, fg_color="transparent")
        btn_row2.pack(fill="x", padx=8, pady=(2, 4))

        ctk.CTkButton(
            btn_row2, text="🎥 Auto 360° from Video Clip",
            command=self._upload_360_video,
            fg_color="#1a4066", hover_color="#24588d",
            corner_radius=6, height=26,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(fill="x", expand=True)

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
            src_row.pack(fill="x", padx=6, pady=(1, 2))

            ctk.CTkEntry(
                src_row, textvariable=cam["source_var"],
                placeholder_text="0 | http://... | file.mp4",
                height=24, fg_color=DARK_BG, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", fill="x", expand=True, padx=(0, 3))

            ctk.CTkButton(
                src_row, text="📁 File",
                command=lambda idx=i: self._browse_camera_video(idx),
                width=52, height=24, fg_color=DARK_BG, hover_color=PANEL_BG,
                border_width=1, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=9, weight="bold")
            ).pack(side="left", padx=(0, 2))

            ctk.CTkButton(
                src_row, text="📱 Phone",
                command=lambda idx=i: self._connect_phone_modal(idx),
                width=62, height=24, fg_color="#142838", hover_color="#1c3c54",
                text_color=ACCENT_BLUE,
                border_width=1, border_color=ACCENT_BLUE,
                font=ctk.CTkFont(size=9, weight="bold")
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

        # Enable smooth recursive mousewheel scrolling across all child widgets
        self._bind_mousewheel_smooth(self)

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

    def _connect_phone_modal(self, slot_idx):
        """Interactive modal dialog to easily connect any Android or iPhone IP camera."""
        modal = ctk.CTkToplevel(self)
        modal.title(f"📱 Connect Phone Camera — Camera {slot_idx + 1}")
        modal.geometry("480x450")
        modal.resizable(False, False)
        modal.configure(fg_color=PANEL_BG)
        modal.transient(self.app)
        modal.grab_set()

        # Center modal over main window
        try:
            self.app.update_idletasks()
            app_x = self.app.winfo_x()
            app_y = self.app.winfo_y()
            app_w = self.app.winfo_width()
            app_h = self.app.winfo_height()
            modal.geometry(f"+{max(50, app_x + (app_w - 480) // 2)}+{max(50, app_y + (app_h - 450) // 2)}")
        except Exception:
            pass

        # Title
        ctk.CTkLabel(
            modal, text=f"📱 LINK WIRELESS PHONE CAMERA (CAM {slot_idx + 1})",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(pady=(12, 4), padx=16, anchor="w")

        ctk.CTkLabel(
            modal, text="Turn any Android or iPhone into a live wireless CCTV camera over Wi-Fi.",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        ).pack(padx=16, anchor="w")

        # Step-by-Step Instruction Card
        card = ctk.CTkFrame(modal, fg_color=CARD_BG, corner_radius=8)
        card.pack(fill="x", padx=16, pady=(10, 8))

        steps_text = (
            "1. Connect phone & laptop to SAME Wi-Fi (or Phone Hotspot)\n"
            "2. Open free app on your phone:\n"
            "   • Android: 'IP Webcam' (Play Store)\n"
            "   • iOS / Android: 'DroidCam' (App Store / Play Store)\n"
            "3. Tap 'Start server' on phone and enter the IP shown on screen:"
        )
        ctk.CTkLabel(
            card, text=steps_text,
            font=ctk.CTkFont(size=10), text_color=TEXT_LIGHT,
            justify="left", anchor="w"
        ).pack(padx=10, pady=8, fill="x")

        # Form variables
        app_type_var = ctk.StringVar(value="ip_webcam")
        ip_var = ctk.StringVar(value="192.168.1.")
        port_var = ctk.StringVar(value="8080")
        loc_var = ctk.StringVar(value=f"Phone Cam {slot_idx + 1}")

        # If camera already has an IP, extract it
        existing_src = self.cam_ui_slots[slot_idx]["source_var"].get().strip()
        if existing_src.startswith("http://"):
            m = re.match(r'http://([^:/]+):?(\d+)?', existing_src)
            if m:
                ip_var.set(m.group(1))
                if m.group(2):
                    port_var.set(m.group(2))

        def _on_app_preset_change(val):
            if val == "IP Webcam (Android)":
                app_type_var.set("ip_webcam")
                port_var.set("8080")
            elif val == "DroidCam (All)":
                app_type_var.set("droidcam")
                port_var.set("4747")
            else:
                app_type_var.set("custom")

        preset_btn = ctk.CTkSegmentedButton(
            modal,
            values=["IP Webcam (Android)", "DroidCam (All)", "Custom IP"],
            command=_on_app_preset_change,
            selected_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=10)
        )
        preset_btn.set("IP Webcam (Android)")
        preset_btn.pack(fill="x", padx=16, pady=(0, 8))

        # Input Form Card
        form = ctk.CTkFrame(modal, fg_color=CARD_BG, corner_radius=8)
        form.pack(fill="x", padx=16, pady=(0, 8))

        row_ip = ctk.CTkFrame(form, fg_color="transparent")
        row_ip.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(row_ip, text="Phone IP Address:", font=ctk.CTkFont(size=11), text_color=TEXT_LIGHT).pack(side="left")
        ctk.CTkLabel(row_ip, text="Port:", font=ctk.CTkFont(size=11), text_color=TEXT_LIGHT).pack(side="right", padx=(0, 24))

        row_fields = ctk.CTkFrame(form, fg_color="transparent")
        row_fields.pack(fill="x", padx=10, pady=(0, 6))

        ip_entry = ctk.CTkEntry(
            row_fields, textvariable=ip_var,
            placeholder_text="e.g. 192.168.1.15",
            fg_color=DARK_BG, border_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=11)
        )
        ip_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        port_entry = ctk.CTkEntry(
            row_fields, textvariable=port_var,
            width=65,
            placeholder_text="8080",
            fg_color=DARK_BG, border_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=11)
        )
        port_entry.pack(side="left")

        ctk.CTkLabel(form, text="Camera Location Tag:", font=ctk.CTkFont(size=10), text_color=TEXT_DIM).pack(padx=10, anchor="w")
        ctk.CTkEntry(
            form, textvariable=loc_var,
            placeholder_text="e.g. Hallway Phone, Entrance Phone",
            fg_color=DARK_BG, border_color=ACCENT_BLUE,
            height=26, font=ctk.CTkFont(size=10)
        ).pack(fill="x", padx=10, pady=(2, 8))

        status_lbl = ctk.CTkLabel(modal, text="", font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_AMBER)
        status_lbl.pack(pady=(0, 4))

        def _apply_url():
            raw_ip = ip_var.get().strip()
            raw_port = port_var.get().strip() or "8080"
            loc = loc_var.get().strip() or f"Phone Cam {slot_idx + 1}"

            if not raw_ip or raw_ip == "192.168.1.":
                status_lbl.configure(text="⚠ Please enter your phone's full IP address!", text_color=ACCENT_RED)
                return

            raw_ip = raw_ip.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in raw_ip:
                parts = raw_ip.split(":")
                raw_ip = parts[0]
                raw_port = parts[1]

            url = f"http://{raw_ip}:{raw_port}/video"

            cam = self.cam_ui_slots[slot_idx]
            cam["source_var"].set(url)
            cam["location_var"].set(loc)
            cam["speed_var"].set("1x")
            cam["enabled"].set(True)
            if "progress_badge" in cam:
                cam["progress_badge"].configure(text="Ready (Phone)", text_color=ACCENT_GREEN)

            self.app.set_status(f"Camera {slot_idx + 1} linked to Phone ({url})", color=ACCENT_GREEN)
            modal.destroy()

        btn_row = ctk.CTkFrame(modal, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkButton(
            btn_row, text="🔗 Connect Phone Camera",
            command=_apply_url,
            fg_color=ACCENT_GREEN, hover_color="#04a57a",
            text_color="#000000",
            corner_radius=8, height=34,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Cancel",
            command=modal.destroy,
            fg_color=DARK_BG, hover_color="#302020",
            border_width=1, border_color="#555577",
            corner_radius=8, height=34,
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(6, 0))

    def _bind_mousewheel_smooth(self, widget):
        """Recursively binds smooth mousewheel scrolling across all child widgets."""
        try:
            widget.bind("<Enter>", self._on_mouse_enter_panel, add="+")
            widget.bind("<Leave>", self._on_mouse_leave_panel, add="+")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._bind_mousewheel_smooth(child)

    def _on_mouse_enter_panel(self, event=None):
        try:
            self._parent_canvas.bind_all("<MouseWheel>", self._on_smooth_mousewheel)
        except Exception:
            pass

    def _on_mouse_leave_panel(self, event=None):
        try:
            self._parent_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass

    def _on_smooth_mousewheel(self, event):
        if sys.platform.startswith("win"):
            # Smooth gliding scroll step (3 units instead of jumpy 20 units)
            delta = -int(event.delta / 40)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            try:
                self._parent_canvas.yview_scroll(delta, "units")
            except Exception:
                pass
            return "break"

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

    def _upload_360_video(self):
        path = filedialog.askopenfilename(
            title="Select Short 360° Video Clip of Target Person (2-10s)",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if not path:
            return

        name = self.target_name_var.get().strip() or "Target Person"
        self.app.tracker.set_target(name)
        self.app.set_status("Scanning video clip & auto-extracting 360° poses (Front, Profile, Back)...", color=ACCENT_AMBER)
        self.update_idletasks()

        def _bg_extract():
            try:
                count = self.app.scanner.extract_360_angles_from_video(path, target_name=name)
                def _success():
                    self.app.scanner.last_sighting["found"] = False
                    self.app.scanner.last_sighting["target_name"] = name
                    self.app.scanner.sightings_history.clear()
                    self._refresh_gallery_display()
                    self.app.set_status(f"Target: {name} (360° Video: {count} poses extracted!)", color=ACCENT_GREEN)
                    messagebox.showinfo(
                        "360° Extractor Complete",
                        f"Successfully extracted {count} distinct body angles (Front, Side, Back) from video clip!\n\nTarget is ready for high-precision search."
                    )
                self.app.after(0, _success)
            except Exception as e:
                err_text = str(e)
                def _error():
                    self.app.set_status("Video 360° Extraction Error", color=ACCENT_RED)
                    messagebox.showerror("360° Video Error", f"Could not extract target from video:\n\n{err_text}")
                self.app.after(0, _error)

        threading.Thread(target=_bg_extract, daemon=True).start()

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

        # Re-bind mousewheel to new gallery cards so scrolling stays smooth
        self._bind_mousewheel_smooth(self.gallery_container)

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
